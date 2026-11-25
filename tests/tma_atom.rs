//! TMA atoms: `make_tiled_tma_atom` and `tma_partition` on the
//! host. Expected descriptors and TMA tensors are derived from C++ CuTe's algorithm
//! (`include/cute/atom/copy_traits_sm90_tma.hpp`: invert the SMEM layout, compose with the
//! CTA tile's map into the global modes, keep the leading unit-step runs, merge runs
//! contiguous in GMEM up to 256 elements, append the other modes with extent 1, fold past
//! five dimensions, split the box over the multicast); each test says how. The placement
//! check (`tma_atom_cases::check_placement`) is the main evidence: the descriptor model
//! puts every GMEM element of a tile where the CuTe layouts put it in shared memory.

mod tma_atom_cases;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;
use tma_atom_cases::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn statics(v: &[usize]) -> Vec<IndexExpr> {
    v.iter().map(|&x| IndexExpr::Static(x)).collect()
}

/// The GMMA tiles over a static `(512, 256)` tensor. With `W` the atom's row in elements
/// (128, 64, 32 or 16 bytes over the element size), C++ CuTe's algorithm gives:
///
/// - K-major (GMEM `(M,K):(K,1)`, SMEM `Sw o (8,W):(W,1)` tiled column-major): the
///   inverted SMEM runs `W` along K, then `BM` along M (the atom's rows and their repeats
///   chain), then `BK/W` along K with a step of `W`, which ends the box: box `(W, BM)`,
///   TMA dimensions `[K, M]` with stride `K` elements, `BK/W` instructions, TMA tensor
///   strides `(1@1, 1@0)`.
/// - M-major (GMEM `(M,K):(1,M)`, SMEM `Sw o (W,8):(1,W)` tiled along K first): runs `W`
///   along M, then `BK` along K: box `(W, BK)`, dimensions `[M, K]` with stride `M`,
///   `BM/W` instructions, strides `(1@0, 1@1)`.
///
/// The swizzle is the atom's (`Swizzle<B,4,3>` on bytes), in cute-rs's element units
/// `Swizzle<B,4-log2(bytes),3>`.
fn gmma_tiles<T: Numeric>() {
    let eb = T::DTYPE.bytes();
    let (m, k) = (512usize, 256usize);
    for atom in Atom::ALL {
        for k_major in [true, false] {
            for (bm, bk) in [(128usize, 64usize), (64, 64)] {
                let what =
                    format!("{} {atom:?} {}-major {bm}x{bk}", T::DTYPE.cute_name(), if k_major { "K" } else { "M" });
                in_kernel(|| {
                    let layout = if k_major { Layout::new((m, k), (k, 1)) } else { Layout::new((m, k), (1, m)) };
                    let a = param_tensor_aligned::<T>("a", layout, 16);
                    let smem = gmma_tile(atom, k_major, eb, bm, bk);
                    // The hand-built atom is the warpgroup helper's.
                    let helper = cute::warpgroup::make_smem_layout_atom::<T>(atom.kind(k_major));
                    let w = atom.row_bytes() / eb;
                    let hand_atom = if k_major {
                        make_composed_layout(element_swizzle(atom, eb), 0, Layout::new((8, w), (w, 1)))
                    } else {
                        make_composed_layout(element_swizzle(atom, eb), 0, Layout::new((w, 8), (1, w)))
                    };
                    assert_eq!(hand_atom, helper, "{what}");
                    let info = cpasync::make_tiled_tma_atom(
                        cpasync::CopyBulkTensorTileG2SOp::default(),
                        &a,
                        smem.clone(),
                        (bm, bk),
                        1,
                    );
                    let ti = info.atom.tma_info().unwrap();
                    let spec = ti.spec();
                    assert_eq!(spec.data_type, TensorMapDataType::for_dtype(T::DTYPE).unwrap(), "{what}");
                    assert_eq!(spec.swizzle, atom.tma_swizzle(), "{what}");
                    assert_eq!(spec.element_strides, vec![1, 1], "{what}");
                    assert_eq!(spec.l2_promotion, TensorMapL2Promotion::l2_128b, "{what}");
                    assert_eq!(spec.base_offset, IndexExpr::Static(0), "{what}");
                    let (box_, dims, stride, gbasis, tma_stride, iters) = if k_major {
                        ([w, bm], [k, m], k, format!("({w},{bm}):(1@1,1@0)"), "(1@1,1@0)", bk / w)
                    } else {
                        ([w, bk], [m, k], m, format!("({w},{bk}):(1@0,1@1)"), "(1@0,1@1)", bm / w)
                    };
                    assert_eq!(spec.box_dims, vec![box_[0] as u32, box_[1] as u32], "{what}");
                    assert_eq!(spec.global_dims, statics(&dims), "{what}");
                    assert_eq!(spec.global_strides, statics(&[stride * eb]), "{what}");
                    assert_eq!(ti.tma_gbasis().to_string(), gbasis, "{what}");
                    assert_eq!(info.tma_tensor.layout().stride().to_string(), tma_stride, "{what}");
                    assert_eq!(info.tma_tensor.layout().shape().to_string(), format!("({m},{k})"), "{what}");
                    assert_eq!(info.atom.num_val_src(), w * box_[1], "{what}");
                    assert_eq!(info.atom.thr_id().to_string(), "1:0", "{what}");
                    // The instructions per tile.
                    let s = make_smem_tensor_aligned::<T>(smem.clone(), 1024);
                    let g = local_tile(&info.tma_tensor, (bm, bk), (1, 2));
                    let (ts, tg) = cpasync::tma_partition(
                        &info.atom,
                        0,
                        &make_layout(1, 0),
                        &s.group_modes(0, 2),
                        &g.group_modes(0, 2),
                    );
                    assert_eq!(ts.layout().mode(0).mode(1).size(), iters, "{what}: {ts:?}");
                    assert_eq!(tg.layout().mode(0).mode(1).size(), iters, "{what}: {tg:?}");
                    // The first instruction starts at the tile's corner.
                    let c0: Coord = Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())])]);
                    let corner = if k_major { vec![2 * bk, bm] } else { vec![bm, 2 * bk] };
                    let got: Vec<usize> = tg.coord(c0).leaves().iter().map(|e| e.as_static().unwrap()).collect();
                    assert_eq!(got, corner, "{what}");
                    check_placement(&info, &a, &IntTuple::from((bm, bk)), (1, 2), &Env::default());
                    check_placement(&info, &a, &IntTuple::from((bm, bk)), (3, 0), &Env::default());
                });
            }
        }
    }
}

#[test]
fn gmma_tiles_f16() {
    gmma_tiles::<F16>();
}

#[test]
fn gmma_tiles_bf16() {
    gmma_tiles::<BF16>();
}

#[test]
fn gmma_tiles_f32() {
    gmma_tiles::<f32>();
}

/// Runtime extents and leading dimension: the descriptor's extents and strides are the
/// scalar parameters, evaluated at each launch.
#[test]
fn gmma_tiles_with_runtime_extents() {
    for case in gmma_desc_cases() {
        let d = case.spec.evaluate(GBASE, &|v| case.env.get(v)).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let k_major = case.name.contains("K-major");
        let eb = d.data_type.bytes();
        assert_eq!(&d.global_dims[..2], if k_major { &[256, 512] } else { &[512, 256] }, "{}", case.name);
        assert_eq!(d.global_strides[0], if k_major { 256 * eb } else { 512 * eb }, "{}", case.name);
        assert!(case.spec.global_dims.iter().all(|e| e.as_static().is_none()), "{}", case.name);
    }
    // And the placement of a tile, f16 128B K-major, with m = 300 (a partial last M tile,
    // which this tile is not) and k = 192.
    in_kernel(|| {
        let m = param_extent("m", 1);
        let k = param_extent("k", 8);
        let a = param_tensor_aligned::<F16>("a", Layout::new((m, k), (k, 1)), 16);
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &a,
            gmma_tile(Atom::SW128, true, 2, 128, 64),
            (128, 64),
            1,
        );
        let env = Env::default().with(&m, 300).with(&k, 192);
        check_placement(&info, &a, &IntTuple::from((128, 64)), (1, 2), &env);
    });
}

/// The swizzle is converted from cute-rs's element units to TMA's bytes for 8-, 16- and
/// 32-bit elements: the 128-byte swizzle is `Swizzle<3,4,3>` on `u8`, `<3,3,3>` on f16 and
/// `<3,2,3>` on f32, and the byte form given as an element swizzle is refused for f16 and
/// f32 (it would be `<3,5,3>` and `<3,6,3>` on bytes).
#[test]
fn swizzle_units_for_8_16_and_32_bit_elements() {
    fn ok<T: Numeric>(sw: Swizzle, width: usize, want: TensorMapSwizzle) {
        in_kernel(|| {
            let a = param_tensor_aligned::<T>("a", Layout::new((256, 256), (256, 1)), 16);
            let s =
                tile_to_shape_swizzled(&make_composed_layout(sw, 0, Layout::new((8, width), (width, 1))), (64, width));
            let info = cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, s, (64, width), 1);
            assert_eq!(info.atom.tma_info().unwrap().swizzle(), want, "{sw} on {}", T::DTYPE);
            check_placement(&info, &a, &IntTuple::from((64, width)), (1, 0), &Env::default());
        });
    }
    ok::<u8>(make_swizzle(3, 4, 3), 128, TensorMapSwizzle::s128b);
    ok::<u8>(make_swizzle(2, 4, 3), 64, TensorMapSwizzle::s64b);
    ok::<u8>(make_swizzle(1, 4, 3), 32, TensorMapSwizzle::s32b);
    ok::<F16>(make_swizzle(3, 3, 3), 64, TensorMapSwizzle::s128b);
    ok::<F16>(make_swizzle(1, 3, 3), 16, TensorMapSwizzle::s32b);
    ok::<f32>(make_swizzle(3, 2, 3), 32, TensorMapSwizzle::s128b);
    ok::<f32>(make_swizzle(2, 2, 3), 16, TensorMapSwizzle::s64b);
    for (bits, sw, width) in [(16, make_swizzle(3, 4, 3), 64), (32, make_swizzle(3, 4, 3), 32)] {
        let msg = build_panic(|| {
            Kernel::build("bad_swizzle", || {
                let s = tile_to_shape_swizzled(
                    &make_composed_layout(sw, 0, Layout::new((8, width), (width, 1))),
                    (64, width),
                );
                if bits == 16 {
                    let a = param_tensor_aligned::<F16>("a", Layout::new((256, 256), (256, 1)), 16);
                    cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, s, (64, width), 1);
                } else {
                    let a = param_tensor_aligned::<f32>("a", Layout::new((256, 256), (256, 1)), 16);
                    cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, s, (64, width), 1);
                }
            });
        });
        let bytes = if bits == 16 { "S<3,5,3>" } else { "S<3,6,3>" };
        assert!(msg.contains(&format!("S<3,4,3> on {bits}-bit elements is {bytes} on bytes")), "{msg}");
        assert!(msg.contains("TMA cannot express"), "{msg}");
    }
}

/// The SM100 128-byte swizzle of 32-byte atoms, `S<2,5,2>`
/// on bytes (`S<2,3,2>` on f32, `S<2,4,2>` on f16), is TMA's `s128b_atom_32b`: the
/// descriptor model places every element where the swizzled layout does. A target before
/// sm_100 refuses it, while the atom is planned and when the kernel is checked.
#[test]
fn the_32_byte_atom_swizzle_on_sm100() {
    fn ok<T: Numeric>(sw: Swizzle, width: usize) {
        in_kernel(|| {
            let a = param_tensor_aligned::<T>("a", Layout::new((256, 256), (256, 1)), 16);
            let s =
                tile_to_shape_swizzled(&make_composed_layout(sw, 0, Layout::new((8, width), (width, 1))), (64, width));
            let info = cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, s, (64, width), 1);
            assert_eq!(info.atom.tma_info().unwrap().swizzle(), TensorMapSwizzle::s128b_atom_32b, "{sw}");
            check_placement(&info, &a, &IntTuple::from((64, width)), (1, 0), &Env::default());
        });
    }
    ok::<f32>(make_swizzle(2, 3, 2), 32);
    ok::<F16>(make_swizzle(2, 4, 2), 64);
    let build = |arch: Arch| {
        Kernel::builder("atom_32b").target(arch).build(|| {
            let a = param_tensor_aligned::<f32>("a", Layout::new((256, 256), (256, 1)), 16);
            let s = tile_to_shape_swizzled(
                &make_composed_layout(make_swizzle(2, 3, 2), 0, Layout::new((8, 32), (32, 1))),
                (64, 32),
            );
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, s, (64, 32), 1);
        })
    };
    let k = build(Arch::Sm100a);
    let e = cute::validate::check_target(k.ir(), Arch::Sm90a).unwrap_err();
    assert!(e.message.contains("s128b_atom_32b") && e.message.contains("sm_100"), "{}", e.message);
    let msg = build_panic(|| {
        build(Arch::Sm90a);
    });
    assert!(msg.contains("s128b_atom_32b mode, which needs sm_100"), "{msg}");
}

/// A staged layout (rank of the tiler plus one) is sliced to its first stage, as CuTe DSL
/// does; `TmaInfo::smem_layout` keeps the staged one. `make_smem_layout_a` for a K-major
/// f16 A of a 128x128x64 tile with 4 stages tiles the 128B atom to `(128,64,4)`.
#[test]
fn staged_layouts_use_one_stage() {
    in_kernel(|| {
        let m = param_extent("m", 1);
        let k = param_extent("k", 8);
        let a = param_tensor_aligned::<F16>("a", Layout::new((m, k), (k, 1)), 16);
        let staged =
            cute::utils::sm90::make_smem_layout_a(LayoutEnum::ROW_MAJOR, (128, 128, 64), cute::ir::DType::F16, 4);
        let info =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &a, staged.clone(), (128, 64), 1);
        assert_eq!(info.smem_layout, staged);
        let ti = info.atom.tma_info().unwrap();
        assert_eq!(ti.smem_layout(), &staged.select(&[0, 1]));
        assert_eq!(ti.spec().box_dims, vec![64, 128]);
        assert_eq!(ti.swizzle(), TensorMapSwizzle::s128b);
        // The partition keeps the stage mode: ((TMA, TMA_Iter), stage).
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let g = local_tile(&info.tma_tensor, (128, 64), (0, ..));
        let (ts, tg) =
            cpasync::tma_partition(&info.atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        // (make_smem_layout_a's stage mode is (1,4): the atom's extent 1 there, tiled 4 times.)
        assert_eq!(ts.layout().to_string(), "((8192,1),(1,4)):((1,0),(0,8192))");
        assert_eq!(tg.layout().shape().mode(0).to_string(), "((64,128),1)");
        // Stage 2's box starts 2 * 8192 elements in.
        let c: Coord = Tree::Branch(vec![
            Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())]),
            Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(2.into())]),
        ]);
        assert_eq!(ts.element_offset(&c), IndexExpr::Static(2 * 8192));
    });
}

/// A batched `(M, K, L)` tensor with runtime extents and batch stride `M*K`: the tiler
/// `(128, 64)` covers M and K, so C++ CuTe drops L from the CTA tile and appends it as a
/// third TMA dimension of box extent 1: dimensions `[K, M, L]`, byte strides
/// `[2K, 2MK]`, box `[64, 128, 1]`, TMA tensor strides `(1@1, 1@0, 1@2)`.
#[test]
fn batched_tensors_append_the_batch_mode() {
    in_kernel(|| {
        let m = param_extent("m", 1);
        let k = param_extent("k", 8);
        let l = param_extent("l", 1);
        let mk = IndexExpr::product(m.into(), k.into());
        let a = param_tensor_aligned::<F16>("a", Layout::new((m, k, l), (IndexExpr::from(k), 1, mk)), 16);
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &a,
            gmma_tile(Atom::SW128, true, 2, 128, 64),
            (128, 64),
            1,
        );
        let ti = info.atom.tma_info().unwrap();
        assert_eq!(ti.spec().box_dims, vec![64, 128, 1]);
        assert_eq!(ti.tma_gbasis().to_string(), "(64,128,1):(1@1,1@0,1@2)");
        assert_eq!(info.tma_tensor.layout().stride().to_string(), "(1@1,1@0,1@2)");
        let env = Env::default().with(&m, 384).with(&k, 320).with(&l, 3);
        let d = ti.spec().evaluate(GBASE, &|v| env.get(v)).unwrap();
        assert_eq!(&d.global_dims[..3], &[320, 384, 3]);
        assert_eq!(&d.global_strides[..2], &[640, 2 * 384 * 320]);
        // The tile (m 2, k 4) of batch 2, and the instruction's coordinates.
        check_placement(&info, &a, &IntTuple::from((128, 64)), (2, 4, 2), &env);
        let g = local_tile(&info.tma_tensor, (128, 64), (2, .., 2));
        let s = make_smem_tensor_aligned::<F16>(ti.smem_layout().clone(), 1024);
        let (_, tg) =
            cpasync::tma_partition(&info.atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let c: Coord =
            Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())]), Tree::Leaf(3.into())]);
        let got: Vec<i64> = tg.coord(c).leaves().iter().map(|e| env.eval(e)).collect();
        assert_eq!(got, vec![3 * 64, 2 * 128, 2]);
    });
}

/// Multicast over 2 and 4 CTAs: the box is split from its outermost dimension (C++ CuTe's
/// `make_tma_copy_desc`), so the K-major 128B f16 box `[64, 128]` becomes `[64, 64]` and
/// `[64, 32]`, while the atom still moves the whole box (`num_val_src` 8192). CTA `c`'s
/// views start at its share, `c * 8192 / n` elements into the box, which is `128/n * c`
/// rows of M. Together the CTAs' boxes place the whole tile.
#[test]
fn multicast_splits_the_box() {
    for n in [2usize, 4] {
        in_kernel(|| {
            let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (256, 1)), 16);
            let info = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SMulticastOp::default(),
                &a,
                gmma_tile(Atom::SW128, true, 2, 128, 64),
                (128, 64),
                n,
            );
            let ti = info.atom.tma_info().unwrap();
            assert_eq!(ti.num_multicast(), n);
            assert_eq!(ti.spec().box_dims, vec![64, 128 / n as u32]);
            assert_eq!(info.atom.num_val_src(), 8192);
            assert_eq!(info.atom.op(), CopyOp::TmaLoad { multicast: true, cta_group: cpasync::CtaGroup::ONE });
            let s = make_smem_tensor_aligned::<F16>(ti.smem_layout().clone(), 1024);
            let g = local_tile(&info.tma_tensor, (128, 64), (1, 3));
            for cta in 0..n {
                let (ts, tg) = cpasync::tma_partition(
                    &info.atom,
                    cta,
                    &make_layout(n, 1),
                    &s.group_modes(0, 2),
                    &g.group_modes(0, 2),
                );
                let c: Coord = Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())])]);
                let got: Vec<usize> = tg.coord(c.clone()).leaves().iter().map(|e| e.as_static().unwrap()).collect();
                assert_eq!(got, vec![3 * 64, 128 + 128 / n * cta], "CTA {cta} of {n}");
                assert_eq!(ts.element_offset(&c), IndexExpr::Static(8192 / n * cta), "CTA {cta} of {n}");
            }
            check_placement(&info, &a, &IntTuple::from((128, 64)), (1, 3), &Env::default());
            // The CTA's coordinate is usually a runtime value (its rank in the cluster).
            let rank = block_in_cluster_idx().y;
            let (ts, tg) = cpasync::tma_partition(
                &info.atom,
                rank,
                &make_layout(n, 1),
                &s.group_modes(0, 2),
                &g.group_modes(0, 2),
            );
            let c: Coord = Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())])]);
            for cta in 0..n {
                let env = Env::default().with(&rank, cta as i64);
                let got: Vec<i64> = tg.coord(c.clone()).leaves().iter().map(|e| env.eval(e)).collect();
                assert_eq!(got, vec![3 * 64, (128 + 128 / n * cta) as i64], "runtime CTA {cta} of {n}");
                assert_eq!(env.eval(&ts.element_offset(&c)), (8192 / n * cta) as i64, "runtime CTA {cta} of {n}");
            }
        });
    }
    // A box that cannot be shared: 3 CTAs and a box of 128 rows.
    let msg = build_panic(|| {
        Kernel::build("mc3", || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (256, 1)), 16);
            cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SMulticastOp::default(),
                &a,
                gmma_tile(Atom::SW128, true, 2, 128, 64),
                (128, 64),
                3,
            );
        });
    });
    assert!(msg.contains("box dimension 1 of 128 elements cannot be split across 3 CTAs"), "{msg}");
}

/// Multicast over 2, 4 and 8 CTAs for every GMMA atom, K- and M-major, f16 and f32: the
/// CTAs' boxes together place the whole tile. And an M-major tile whose atoms repeat
/// along M first (`Sw o (W,8):(1,W)` tiled column-major): its box is `[W, 8]` (the run
/// along M, then the atom's 8 columns of K; the next run steps M by `W`, not 1), so two
/// CTAs share it at half a swizzle pattern. The instruction takes the share's address
/// before the swizzle, as C++ CuTe's `raw_pointer_cast` of a swizzled pointer gives, and
/// the swizzle by address then puts it where the layout says. Unswizzled, the share is
/// 64 bytes, which TMA's 128-byte-aligned destination refuses.
#[test]
fn multicast_places_every_atom() {
    fn one<T: Numeric>(atom: Atom, k_major: bool, n: usize) {
        in_kernel(|| {
            let layout = if k_major { Layout::new((512, 256), (256, 1)) } else { Layout::new((512, 256), (1, 512)) };
            let a = param_tensor_aligned::<T>("a", layout, 16);
            let smem = gmma_tile(atom, k_major, T::DTYPE.bytes(), 128, 64);
            let info = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SMulticastOp::default(),
                &a,
                smem,
                (128, 64),
                n,
            );
            check_placement(&info, &a, &IntTuple::from((128, 64)), (1, 2), &Env::default());
        });
    }
    for atom in Atom::ALL {
        for k_major in [true, false] {
            for n in [2, 4, 8] {
                one::<F16>(atom, k_major, n);
                one::<f32>(atom, k_major, n);
            }
        }
    }
    for atom in [Atom::SW128, Atom::SW64, Atom::SW32] {
        in_kernel(|| {
            let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (1, 512)), 16);
            let w = atom.row_bytes() / 2;
            let atom_layout = make_composed_layout(element_swizzle(atom, 2), 0, Layout::new((w, 8), (1, w)));
            let smem = tile_to_shape_swizzled(&atom_layout, (128, 64));
            let info = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SMulticastOp::default(),
                &a,
                smem,
                (128, 64),
                2,
            );
            assert_eq!(info.atom.tma_info().unwrap().spec().box_dims, vec![w as u32, 4], "{atom:?}");
            check_placement(&info, &a, &IntTuple::from((128, 64)), (1, 2), &Env::default());
        });
    }
    let msg = build_panic(|| {
        Kernel::build("mc_inter", || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (1, 512)), 16);
            let smem = tile_to_shape(&Layout::new((8, 8), (1, 8)), (128, 64));
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SMulticastOp::default(), &a, smem, (128, 64), 2);
        });
    });
    assert!(msg.contains("share of the box [8, 4] is 64 bytes") && msg.contains("128-byte aligned"), "{msg}");
}

/// Runs contiguous in GMEM merge into one TMA dimension (C++ CuTe's `coalesce_256`), across
/// global modes: a GMEM M folded as `((8,16),64):((1,8),128)` with an unswizzled M-major
/// tile `(128,64):(1,128)` gives runs 8 and 16 along modes (0,0) and (0,1), merged into a
/// dimension of extent `(8-1)*1 + (16-1)*8 + 1 = 128` (the recurrence of
/// `fill_tma_gmem_shape_stride`), then 64 along K: box `[128, 64]`, and the TMA tensor's
/// strides `((1@0, 8@0), 1@1)` (mode (0,1) is 8 TMA elements per step).
#[test]
fn contiguous_global_modes_merge() {
    in_kernel(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new(((8, 16), 64), ((1, 8), 128)), 16);
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &a,
            Layout::new((128, 64), (1, 128)),
            (128, 64),
            1,
        );
        let ti = info.atom.tma_info().unwrap();
        assert_eq!(ti.tma_gbasis().to_string(), "((8,16),64):((1@0@0,1@1@0),1@1)");
        assert_eq!(ti.spec().box_dims, vec![128, 64]);
        assert_eq!(ti.spec().global_dims, statics(&[128, 64]));
        assert_eq!(ti.spec().global_strides, statics(&[256]));
        assert_eq!(info.tma_tensor.layout().stride().to_string(), "((1@0,8@0),1@1)");
        check_placement(&info, &a, &IntTuple::from((128, 64)), (0, 0), &Env::default());
    });
}

/// A contiguous run longer than 256 elements: a K-major tile `(16,512):(512,1)` of f16 runs
/// 512 elements along K, which `coalesce_256` keeps as one mode (it only merges up to 256)
/// and C++ CuTe then rejects (its box assertion); so does cute-rs, saying why.
#[test]
fn runs_longer_than_256_are_refused() {
    let msg = build_panic(|| {
        Kernel::build("long_run", || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((64, 1024), (1024, 1)), 16);
            cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileS2GOp,
                &a,
                Layout::new((16, 512), (512, 1)),
                (16, 512),
                1,
            );
        });
    });
    assert!(msg.contains("the box's dimension 0 would hold 512 elements"), "{msg}");
    assert!(msg.contains("more than TMA's 256"), "{msg}");
}

/// More than five TMA dimensions. A 6-mode tensor of extents `(8,2,2,2,2,2)` whose strides
/// leave gaps (`1,16,64,256,1024,4096`) under a tile covering five modes appends the sixth
/// as a box-extent-1 dimension and folds it with the fifth: strides 1024 and 4096 share
/// the gcd 1024, the folded extent is `(2-1)*1 + (2-1)*4 + 1 = 6`, and mode 5's TMA stride
/// is `4096/1024 = 4@4`. Covering all six modes needs six box dimensions, and folding
/// strides 1024 and 4096 into one breaks the box: refused.
#[test]
fn more_than_five_dimensions_fold_or_are_refused() {
    in_kernel(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((8, 2, 2, 2, 2, 2), (1, 16, 64, 256, 1024, 4096)), 16);
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &a,
            Layout::col_major((8, 2, 2, 2, 2)),
            (8, 2, 2, 2, 2),
            1,
        );
        let ti = info.atom.tma_info().unwrap();
        assert_eq!(ti.tma_gbasis().to_string(), "(8,2,2,2,(2,1)):(1@0,1@1,1@2,1@3,(1@4,1@5))");
        assert_eq!(ti.spec().box_dims, vec![8, 2, 2, 2, 2]);
        assert_eq!(ti.spec().global_dims, statics(&[8, 2, 2, 2, 6]));
        assert_eq!(ti.spec().global_strides, statics(&[32, 128, 512, 2048]));
        assert_eq!(info.tma_tensor.layout().stride().to_string(), "(1@0,1@1,1@2,1@3,1@4,4@4)");
        check_placement(&info, &a, &IntTuple::from((8, 2, 2, 2, 2)), (0, 0, 0, 0, 0, 1), &Env::default());
    });
    let msg = build_panic(|| {
        Kernel::build("six", || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((8, 2, 2, 2, 2, 2), (1, 16, 64, 256, 1024, 4096)), 16);
            cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SOp::default(),
                &a,
                Layout::col_major((8, 2, 2, 2, 2, 2)),
                (8, 2, 2, 2, 2, 2),
                1,
            );
        });
    });
    assert!(msg.contains("the tile needs 6 TMA dimensions"), "{msg}");
    assert!(msg.contains("at most 5"), "{msg}");
}

/// dense_gemm.py's epilogue store atom, `make_tiled_tma_atom(CopyBulkTensorTileS2GOp(),
/// tensor_c, epi_smem_layout, epi_tile)`, for a row-major (N-major) f16 C of runtime
/// `(M, N, L)` and an epilogue tile `(64, 32)` with 4 stages (`make_smem_layout_epi`):
/// 32 f16 along N are 512 bits, so the atom is the 64B one, `(8,32):(32,1)`, and the box
/// `[32, 64]` (N, then M) with the 64-byte swizzle, plus L as a third dimension.
#[test]
fn dense_gemm_epilogue_store_atom() {
    in_kernel(|| {
        let m = param_extent("m", 1);
        let n = param_extent("n", 8);
        let l = param_extent("l", 1);
        let mn = IndexExpr::product(m.into(), n.into());
        let c = param_tensor_aligned::<F16>("c", Layout::new((m, n, l), (IndexExpr::from(n), 1, mn)), 16);
        let staged = cute::utils::sm90::make_smem_layout_epi(
            cute::ir::DType::F16,
            LayoutEnum::ROW_MAJOR,
            (64, 32),
            4,
            None,
            None,
        );
        let epi = staged.slice((.., .., 0));
        let info = cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &c, epi, (64, 32), 1);
        assert_eq!(info.atom.op(), CopyOp::TmaStore);
        let ti = info.atom.tma_info().unwrap();
        assert_eq!(ti.swizzle(), TensorMapSwizzle::s64b);
        assert_eq!(ti.spec().box_dims, vec![32, 64, 1]);
        assert_eq!(info.tma_tensor.layout().stride().to_string(), "(1@1,1@0,1@2)");
        let env = Env::default().with(&m, 256).with(&n, 96).with(&l, 2);
        check_placement(&info, &c, &IntTuple::from((64, 32)), (3, 2, 1), &env);
        // The reduce store builds the same descriptor.
        let red = cpasync::make_tiled_tma_atom(
            cpasync::CopyReduceBulkTensorTileS2GOp { reduction_kind: cpasync::ReductionKind::MAX },
            &c,
            staged.slice((.., .., 0)),
            (64, 32),
            1,
        );
        assert_eq!(red.atom.op(), CopyOp::TmaReduce(cpasync::ReductionKind::MAX));
        let rs = red.atom.tma_info().unwrap().spec();
        assert_eq!((rs.box_dims.clone(), rs.swizzle), (ti.spec().box_dims.clone(), ti.spec().swizzle));
    });
}

/// tma_v0.py: a row-major f16 `(M, N)` copied through a `(128,128):(128,1)` SMEM tile with
/// a G2S and an S2G atom (tiler `product_each(smem.shape)`). Unswizzled, the box is the
/// whole tile `[128, 128]` (N, then M), one instruction per tile; after
/// `local_tile(tma_tensor, (128,128), (None,None))`, `group_modes(..., 0, 2)` and
/// `tma_partition(atom, 0, make_layout(1), ...)`, `tAgA[(None, bidx, bidy)]` starts at
/// TMA coordinates `(128*bidy, 128*bidx)`.
#[test]
fn tma_v0_partition() {
    in_kernel(|| {
        let m = param_extent("m", 1);
        let n = param_extent("n", 16);
        let src = param_tensor_aligned::<F16>("src", Layout::new((m, n), (n, 1)), 16);
        let dst = param_tensor_aligned::<F16>("dst", Layout::new((m, n), (n, 1)), 16);
        let smem_layout = Layout::new((128, 128), (128, 1));
        let tiler = smem_layout.shape().clone();
        let load = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &src,
            smem_layout.clone(),
            Tiler::from(&tiler),
            1,
        );
        let store = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileS2GOp,
            &dst,
            smem_layout.clone(),
            Tiler::from(&tiler),
            1,
        );
        let (load_atom, tma_src): (CopyAtom<F16>, Tensor<Ident>) = load.clone().into();
        let (store_atom, tma_dst) = store.into_parts();
        for atom in [&load_atom, &store_atom] {
            let spec = atom.tma_info().unwrap().spec();
            assert_eq!(spec.box_dims, vec![128, 128]);
            assert_eq!(spec.swizzle, TensorMapSwizzle::none);
        }
        assert_ne!(
            load_atom.tma_info().unwrap().descriptor().name(),
            store_atom.tma_info().unwrap().descriptor().name()
        );
        assert_eq!(tma_src.layout().stride().to_string(), "(1@1,1@0)");
        let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
        let g_src = local_tile(&tma_src, (128, 128), (.., ..));
        let g_dst = local_tile(&tma_dst, (128, 128), (.., ..));
        let (ts, tg) =
            cpasync::tma_partition(&load_atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g_src.group_modes(0, 2));
        let (_, tb) =
            cpasync::tma_partition(&store_atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g_dst.group_modes(0, 2));
        assert_eq!(ts.layout().to_string(), "((16384,1)):((1,0))");
        assert_eq!(tg.rank(), 3);
        assert_eq!(tg.layout().shape().mode(0).to_string(), "((128,128),1)");
        for (bx, by) in [(0usize, 0usize), (2, 1), (5, 7)] {
            for t in [&tg, &tb] {
                let cta = t.slice(Tree::Branch(vec![
                    Tree::Leaf(None),
                    Tree::Leaf(Some(bx.into())),
                    Tree::Leaf(Some(by.into())),
                ]));
                let c: Coord = Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())])]);
                let got: Vec<usize> = cta.coord(c).leaves().iter().map(|e| e.as_static().unwrap()).collect();
                assert_eq!(got, vec![128 * by, 128 * bx]);
            }
        }
        let env = Env::default().with(&m, 512).with(&n, 384);
        check_placement(&load, &src, &IntTuple::from((128, 128)), (3, 2), &env);
    });
}

/// dense_gemm.py's A and B loads: `make_smem_layout_a/b` (K-major f16, tile 128x128x64,
/// 4 stages) over runtime `(M, K, L)` and `(N, K, L)`, partitioned with
/// `group_modes(sA, 0, 2)` and `group_modes(gA_mkl, 0, 2)`. With a 1x1 cluster
/// (`cta_layout` `make_layout(1)`) the box is `[64, 128, 1]` and k tile `kt` starts at
/// `(64 kt, 128 m, l)`; with a 1x2 cluster A is multicast over the 2 CTAs along N
/// (`make_layout(2)`): the box halves to `[64, 64, 1]` and CTA `c`'s share starts 64 c rows
/// further along M, 4096 c elements into each stage.
#[test]
fn dense_gemm_operand_partitions() {
    for mcast in [1usize, 2] {
        in_kernel(|| {
            let m = param_extent("m", 1);
            let n = param_extent("n", 1);
            let k = param_extent("k", 8);
            let l = param_extent("l", 1);
            let mk = IndexExpr::product(m.into(), k.into());
            let nk = IndexExpr::product(n.into(), k.into());
            let a = param_tensor_aligned::<F16>("a", Layout::new((m, k, l), (IndexExpr::from(k), 1, mk)), 16);
            let b = param_tensor_aligned::<F16>("b", Layout::new((n, k, l), (IndexExpr::from(k), 1, nk)), 16);
            let tile = (128, 128, 64);
            let sa_staged = cute::utils::sm90::make_smem_layout_a(LayoutEnum::ROW_MAJOR, tile, cute::ir::DType::F16, 4);
            let sb_staged = cute::utils::sm90::make_smem_layout_b(LayoutEnum::ROW_MAJOR, tile, cute::ir::DType::F16, 4);
            let op_a: CopyOp = if mcast == 1 {
                cpasync::CopyBulkTensorTileG2SOp::default().into()
            } else {
                cpasync::CopyBulkTensorTileG2SMulticastOp::default().into()
            };
            let ta = cpasync::make_tiled_tma_atom(op_a, &a, sa_staged.slice((.., .., 0)), (128, 64), mcast);
            let tb = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SOp::default(),
                &b,
                sb_staged.slice((.., .., 0)),
                (128, 64),
                1,
            );
            assert_eq!(ta.atom.tma_info().unwrap().spec().box_dims, vec![64, 128 / mcast as u32, 1]);
            assert_eq!(tb.atom.tma_info().unwrap().spec().box_dims, vec![64, 128, 1]);
            let sa = make_smem_tensor_aligned::<F16>(sa_staged, 1024);
            let (mi, li) = (2usize, 1usize);
            let ga = local_tile(&ta.tma_tensor, (128, 64), (mi, .., li));
            let env = Env::default().with(&m, 1024).with(&n, 512).with(&k, 512).with(&l, 2);
            for c in 0..mcast {
                let (tas, tag) = cpasync::tma_partition(
                    &ta.atom,
                    c,
                    &make_layout(mcast, 1),
                    &sa.group_modes(0, 2),
                    &ga.group_modes(0, 2),
                );
                assert_eq!(tas.layout().to_string(), "((8192,1),(1,4)):((1,0),(0,8192))");
                for (kt, stage) in [(0usize, 0usize), (5, 3)] {
                    let crd = |rest: usize| -> Coord {
                        Tree::Branch(vec![
                            Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())]),
                            Tree::Leaf(rest.into()),
                        ])
                    };
                    let got: Vec<i64> = tag.coord(crd(kt)).leaves().iter().map(|e| env.eval(e)).collect();
                    assert_eq!(got, vec![64 * kt as i64, (128 * mi + 64 * c) as i64, li as i64], "CTA {c} k tile {kt}");
                    let at_stage: Coord = Tree::Branch(vec![
                        Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(0.into())]),
                        Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(stage.into())]),
                    ]);
                    assert_eq!(
                        tas.element_offset(&at_stage),
                        IndexExpr::Static(8192 * stage + 4096 * c),
                        "CTA {c} stage {stage}"
                    );
                }
            }
            check_placement(&ta, &a, &IntTuple::from((128, 64)), (mi, 3, li), &env);
            check_placement(&tb, &b, &IntTuple::from((128, 64)), (1, 7, 0), &env);
        });
    }
}

/// The global tensor may be `make_tensor` on a `param_ptr`, and offset into it (a
/// 16-byte-aligned start): the descriptor's start is the offset.
#[test]
fn tensors_on_param_ptrs_and_offsets() {
    in_kernel(|| {
        let p = param_ptr_aligned::<f32>("p", 16);
        let full = make_tensor(&p, Layout::new((256, 128), (128, 1)));
        let a = domain_offset((8, 32), &full);
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &a,
            gmma_tile(Atom::SW64, true, 4, 64, 32),
            (64, 32),
            1,
        );
        let spec = info.atom.tma_info().unwrap().spec();
        assert_eq!(spec.base_offset, IndexExpr::Static((8 * 128 + 32) * 4));
        assert_eq!(spec.box_dims, vec![16, 64]);
        assert!(info.atom.tma_info().unwrap().descriptor().name().starts_with("p_tma_desc_"));
    });
}

/// The atom's hidden descriptor is a kernel parameter after the kernel's own, encoded at
/// each launch: the simulator encodes it from the tensor's address and the scalars.
#[test]
fn the_atom_descriptor_is_encoded_at_launch() {
    let k = Kernel::builder("tma_atom_desc").block_dim([32, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let kk = param_extent("k", 8);
        let a = param_tensor::<F16>("a", Layout::new((m, kk), (kk, 1)));
        let out = param_tensor::<u64>("out", Layout::col_major((16,)));
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SOp::default(),
            &a,
            gmma_tile(Atom::SW128, true, 2, 128, 64),
            (128, 64),
            1,
        );
        let desc = info.atom.tma_info().unwrap().descriptor().clone();
        if_(thread_idx().x.eq(0), || {
            for w in 0..16 {
                out.store(w, desc.load_word(w));
            }
        });
    });
    let sig = k.signature();
    assert_eq!(sig.user_param_count(), 4);
    assert_eq!(sig.params[4].name, "a_tma_desc_0");
    let mut args = vec![
        KernelArg::scalar(300i32),
        KernelArg::scalar(192i32),
        KernelArg::f16_from_f32(&vec![0.0; 300 * 192]),
        KernelArg::buffer(&[0u64; 16]),
    ];
    k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let bytes: Vec<u8> = args[3].to_vec::<u64>().iter().flat_map(|w| w.to_le_bytes()).collect();
    let d = TmaDesc::from_sim_bytes(&bytes).unwrap();
    assert_eq!(d.rank, 2);
    assert_eq!(&d.global_dims[..2], &[192, 300]);
    assert_eq!(d.global_strides[0], 384);
    assert_eq!(&d.box_dims[..2], &[64, 128]);
    assert_eq!(d.swizzle, TensorMapSwizzle::s128b);
    assert_eq!(d.l2_promotion, TensorMapL2Promotion::l2_128b);
}

fn expect_build_panic(f: impl FnOnce() + Clone, needles: &[&str]) {
    let msg = build_panic(|| {
        Kernel::build("diag", f);
    });
    for n in needles {
        assert!(msg.contains(n), "{msg:?} lacks {n:?}");
    }
}

fn a_kmajor() -> Tensor<F16> {
    param_tensor_aligned::<F16>("a", Layout::new((512, 256), (256, 1)), 16)
}

#[test]
fn diagnostics() {
    let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
    // A non-contiguous SMEM box: the SMEM layout's offset 1 is no element of the tile.
    expect_build_panic(
        move || {
            cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::new((64, 64), (2, 128)), (64, 64), 1);
        },
        &["`a`: the SMEM layout (64,64):(2,128) has no contiguous run along a unit step of a global mode"],
    );
    // A swizzle TMA cannot express.
    expect_build_panic(
        move || {
            let s = make_composed_layout(make_swizzle(2, 3, 2), 0, Layout::new((64, 64), (64, 1)));
            cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), s, (64, 64), 1);
        },
        &["S<2,3,2> on 16-bit elements is S<2,4,2> on bytes, which TMA cannot express"],
    );
    // The SMEM layout's rank against the tiler's.
    expect_build_panic(
        move || {
            cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::col_major((64, 8, 2, 2)), (64, 32), 1);
        },
        &["has rank 4, but the CTA tiler has rank 2"],
    );
    // The SMEM layout's size against the tile's.
    expect_build_panic(
        move || {
            cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 32), 1);
        },
        &["has 4096 elements, but the CTA tile (64,32) has 2048"],
    );
    // Majorness: an M-major tile of a K-major tensor.
    expect_build_panic(
        move || {
            cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::new((64, 64), (1, 64)), (64, 64), 1);
        },
        &["walks global mode [0] whose stride is 256, not 1", "majorness"],
    );
    // A tile walking a broadcast (stride-0) mode.
    expect_build_panic(
        move || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (0, 1)), 16);
            cpasync::make_tiled_tma_atom(g2s, &a, Layout::new((64, 64), (64, 1)), (64, 64), 1);
        },
        &["walks global mode [0] whose stride is 0 (a broadcast mode)"],
    );
    // A global tensor that is not on a kernel parameter.
    expect_build_panic(
        move || {
            let s = make_smem_tensor::<F16>(Layout::new((64, 64), (64, 1)));
            cpasync::make_tiled_tma_atom(g2s, &s, Layout::new((64, 64), (64, 1)), (64, 64), 1);
        },
        &["`smem0` is in smem memory, not global memory"],
    );
    expect_build_panic(
        move || {
            let r = make_rmem_tensor::<F16>(Layout::col_major((64, 64)));
            cpasync::make_tiled_tma_atom(g2s, &r, Layout::new((64, 64), (64, 1)), (64, 64), 1);
        },
        &["`stack0` is in rmem memory, not global memory"],
    );
    // Multicast with the plain load, CTA pairs, the SM100 options, a non-TMA op.
    expect_build_panic(
        move || {
            cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 64), 2);
        },
        &["num_multicast is 1, not 2"],
    );
    // CTA-pair loads exist on SM100 only.
    let msg = build_panic(|| {
        Kernel::builder("diag").target(Arch::Sm90a).build(|| {
            let op = cpasync::CopyBulkTensorTileG2SOp { cta_group: cpasync::CtaGroup::TWO };
            cpasync::make_tiled_tma_atom(op, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 64), 1);
        });
    });
    assert!(msg.contains("CtaGroup::TWO") && msg.contains("sm_90a"), "{msg:?}");
    expect_build_panic(
        move || {
            let opts = cpasync::TmaAtomOptions::new().internal_type(cute::ir::DType::U8);
            cpasync::make_tiled_tma_atom_with(g2s, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 64), 1, &opts);
        },
        &["internal_type=Uint8 is 8 bits", "Float16"],
    );
    expect_build_panic(
        move || {
            let idx = param_tensor::<i32>("idx", Layout::new((64, 64), (0, 1)));
            let opts = cpasync::TmaAtomOptions::new().gmem_coord_tensor(&idx);
            cpasync::make_tiled_tma_atom_with(g2s, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 64), 1, &opts);
        },
        &["gmem_coord_tensor is for the gather4/scatter4 modes"],
    );
    expect_build_panic(
        move || {
            cpasync::make_tiled_tma_atom(
                CopyOp::Universal { bits: 128 },
                &a_kmajor(),
                Layout::new((64, 64), (64, 1)),
                (64, 64),
                1,
            );
        },
        &["not a TMA tile operation"],
    );
    // TMA atoms come from make_tiled_tma_atom only, and copy with their mbarrier.
    expect_build_panic(
        || {
            let _ = make_copy_atom::<F16>(cpasync::CopyBulkTensorTileS2GOp);
        },
        &["make it with cpasync::make_tiled_tma_atom"],
    );
    expect_build_panic(
        move || {
            let a = a_kmajor();
            let info = cpasync::make_tiled_tma_atom(g2s, &a, Layout::new((64, 64), (64, 1)), (64, 64), 1);
            let s = make_smem_tensor::<F16>(Layout::new((64, 64), (64, 1)));
            let g = local_tile(&a, (64, 64), (0, 0));
            copy(&info.atom, &g, &s);
        },
        &["needs `tma_bar_ptr`"],
    );
    // tma_partition: a non-TMA atom, mismatched tiles, the list form.
    expect_build_panic(
        || {
            let s = make_smem_tensor::<F16>(Layout::new((64, 64), (64, 1)));
            let g = make_identity_tensor((64, 64));
            let atom = make_copy_atom::<F16>(CopyOp::Universal { bits: 16 });
            cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        },
        &["is not a TMA atom"],
    );
    expect_build_panic(
        move || {
            let info = cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 64), 1);
            let s = make_smem_tensor::<F16>(Layout::new((64, 64), (64, 1)));
            let g = local_tile(&info.tma_tensor, (64, 64), (0, 0));
            cpasync::tma_partition(&info.atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g);
        },
        &["the GMEM tensor's mode 0 has 64 elements, the SMEM tile 4096", "group the tile's modes"],
    );
    expect_build_panic(
        move || {
            let info = cpasync::make_tiled_tma_atom(g2s, &a_kmajor(), Layout::new((64, 64), (64, 1)), (64, 64), 1);
            let s = make_smem_tensor::<F16>(Layout::new((64, 64), (64, 1)));
            let idx = param_tensor::<i32>("idx", Layout::new((64, 64), (0, 1)));
            cpasync::tma_partition_indexed(&info.atom, 0, &make_layout(1, 0), &s, &info.tma_tensor, &idx);
        },
        &["takes a gather4/scatter4 atom"],
    );
    // tma_partition: a CTA layout other than the multicast's, a rank outside it, a signed
    // rank.
    fn mc2() -> (CopyAtom<F16>, Tensor<F16>, Tensor<Ident>) {
        let info = cpasync::make_tiled_tma_atom(
            cpasync::CopyBulkTensorTileG2SMulticastOp::default(),
            &a_kmajor(),
            gmma_tile(Atom::SW128, true, 2, 128, 64),
            (128, 64),
            2,
        );
        let s = make_smem_tensor_aligned::<F16>(info.atom.tma_info().unwrap().smem_layout().clone(), 1024);
        let g = local_tile(&info.tma_tensor, (128, 64), (1, 3));
        (info.atom, s.group_modes(0, 2), g.group_modes(0, 2))
    }
    expect_build_panic(
        || {
            let (atom, s, g) = mc2();
            cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s, &g);
        },
        &["the CTA layout 1:0 spans 1 CTAs, but the atom's box is split across 2", "make_layout(2)"],
    );
    expect_build_panic(
        || {
            let (atom, s, g) = mc2();
            cpasync::tma_partition(&atom, 2, &make_layout(2, 1), &s, &g);
        },
        &["multicast rank 2, outside the 2 CTAs of 2:1"],
    );
    expect_build_panic(
        || {
            let (atom, s, g) = mc2();
            let r = IndexExpr::signed(block_in_cluster_idx().y);
            cpasync::tma_partition(&atom, r, &make_layout(2, 1), &s, &g);
        },
        &["which may be negative"],
    );
}

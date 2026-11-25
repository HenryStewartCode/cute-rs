//! The SM90 layout helpers: the operand major modes, the GMMA shared-memory
//! layout atoms (`warpgroup.make_smem_layout_atom`), `LayoutEnum`, and
//! `cutlass.utils.sm90`'s `get_smem_layout_atom` and
//! `make_smem_layout_a/b/epi`. Host-side layout code: no kernel runs.

use cute::prelude::*;
use cute::utils::sm90::{get_smem_layout_atom, make_smem_layout_a, make_smem_layout_b, make_smem_layout_epi};
use cute::warpgroup::{SmemLayoutAtomKind, SmemLayoutAtomKind::*, make_smem_layout_atom, make_smem_layout_atom_dtype};

fn panic_text(f: impl FnOnce()) -> String {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

#[test]
fn operand_major_modes() {
    assert_eq!("k".parse::<OperandMajorMode>(), Ok(OperandMajorMode::K));
    assert_eq!("Mn".parse::<OperandMajorMode>(), Ok(OperandMajorMode::MN));
    assert!("m".parse::<OperandMajorMode>().is_err());
    assert_eq!(OperandMajorMode::K.to_string(), "OperandMajorMode.K");
    assert_eq!("n".parse::<OutputMajorMode>(), Ok(OutputMajorMode::N));
    assert_eq!(OutputMajorMode::M.to_string(), "OutputMajorMode.M");
    // warpgroup's (deprecated) name is the same type.
    let _: cute::warpgroup::OperandMajorMode = cute::nvgpu::OperandMajorMode::MN;
}

/// Every atom kind for every element width: `(c,8):(1,c)` (MN) or
/// `(8,c):(c,1)` (K) with `c = bits / width`, under the byte swizzle
/// `Swizzle<B,4,3>` restated on elements.
#[test]
fn atoms_for_every_width() {
    let table: [(DType, [&str; 8]); 5] = [
        (
            DType::F16,
            [
                "S<0,4,3> o 0 o (8,8):(1,8)",
                "S<1,3,3> o 0 o (16,8):(1,16)",
                "S<2,3,3> o 0 o (32,8):(1,32)",
                "S<3,3,3> o 0 o (64,8):(1,64)",
                "S<0,4,3> o 0 o (8,8):(8,1)",
                "S<1,3,3> o 0 o (8,16):(16,1)",
                "S<2,3,3> o 0 o (8,32):(32,1)",
                "S<3,3,3> o 0 o (8,64):(64,1)",
            ],
        ),
        (
            DType::BF16,
            [
                "S<0,4,3> o 0 o (8,8):(1,8)",
                "S<1,3,3> o 0 o (16,8):(1,16)",
                "S<2,3,3> o 0 o (32,8):(1,32)",
                "S<3,3,3> o 0 o (64,8):(1,64)",
                "S<0,4,3> o 0 o (8,8):(8,1)",
                "S<1,3,3> o 0 o (8,16):(16,1)",
                "S<2,3,3> o 0 o (8,32):(32,1)",
                "S<3,3,3> o 0 o (8,64):(64,1)",
            ],
        ),
        (
            DType::TF32,
            [
                "S<0,4,3> o 0 o (4,8):(1,4)",
                "S<1,2,3> o 0 o (8,8):(1,8)",
                "S<2,2,3> o 0 o (16,8):(1,16)",
                "S<3,2,3> o 0 o (32,8):(1,32)",
                "S<0,4,3> o 0 o (8,4):(4,1)",
                "S<1,2,3> o 0 o (8,8):(8,1)",
                "S<2,2,3> o 0 o (8,16):(16,1)",
                "S<3,2,3> o 0 o (8,32):(32,1)",
            ],
        ),
        (
            DType::F8E4M3FN,
            [
                "S<0,4,3> o 0 o (16,8):(1,16)",
                "S<1,4,3> o 0 o (32,8):(1,32)",
                "S<2,4,3> o 0 o (64,8):(1,64)",
                "S<3,4,3> o 0 o (128,8):(1,128)",
                "S<0,4,3> o 0 o (8,16):(16,1)",
                "S<1,4,3> o 0 o (8,32):(32,1)",
                "S<2,4,3> o 0 o (8,64):(64,1)",
                "S<3,4,3> o 0 o (8,128):(128,1)",
            ],
        ),
        // 4-bit elements: the byte swizzle keeps two more low bits.
        (
            DType::I4,
            [
                "S<0,4,3> o 0 o (32,8):(1,32)",
                "S<1,5,3> o 0 o (64,8):(1,64)",
                "S<2,5,3> o 0 o (128,8):(1,128)",
                "S<3,5,3> o 0 o (256,8):(1,256)",
                "S<0,4,3> o 0 o (8,32):(32,1)",
                "S<1,5,3> o 0 o (8,64):(64,1)",
                "S<2,5,3> o 0 o (8,128):(128,1)",
                "S<3,5,3> o 0 o (8,256):(256,1)",
            ],
        ),
    ];
    for (dtype, expected) in table {
        for (kind, want) in SmemLayoutAtomKind::ALL.into_iter().zip(expected) {
            let atom = make_smem_layout_atom_dtype(kind, dtype);
            assert_eq!(atom.to_string(), want, "{kind:?} for {dtype:?}");
            // The element swizzle is the byte swizzle on elements.
            let sw = atom.swizzle().unwrap();
            if !sw.is_identity() {
                assert_eq!(sw.to_bytes(dtype.bits()), kind.byte_swizzle());
            }
            assert_eq!(atom.size() * dtype.bits(), 8 * kind.contiguous_bits());
        }
    }
    // Float32 and Float8E5M2 match TFloat32 and Float8E4M3FN.
    assert_eq!(make_smem_layout_atom::<f32>(K_SW64), make_smem_layout_atom_dtype(K_SW64, DType::TF32));
    assert_eq!(make_smem_layout_atom::<F8E5M2>(MN_SW32), make_smem_layout_atom_dtype(MN_SW32, DType::F8E4M3FN));
    assert_eq!(MN_SW128.byte_swizzle(), make_swizzle(3, 4, 3));
    assert_eq!(K_INTER.major_mode(), OperandMajorMode::K);
}

#[test]
fn atoms_refuse_widths_that_do_not_fit() {
    let six = panic_text(|| {
        make_smem_layout_atom_dtype(K_SW128, DType::F6E2M3FN);
    });
    assert!(six.contains("Float6E2M3FN is 6 bits wide"), "{six}");
    let one = panic_text(|| {
        make_smem_layout_atom::<bool>(MN_INTER);
    });
    assert!(one.contains("1 bits wide"), "{one}");
}

/// The byte offset of `(mn, k)` in a canonical WGMMA shared-memory layout,
/// as the PTX ISA's table of canonical layouts gives them (`T = 128 /
/// bits` elements, `w = 2^B * T` elements per swizzled row), with the
/// swizzle `Swizzle<B,4,3>` on byte addresses. K-major:
/// `((8,m),(T,2^B k)):((wT..,SBO),(1,T))`, written here as row `mn % 8` at
/// `w` elements, groups of 8 rows at `sbo`, and `k < w` contiguous;
/// MN-major: `((T,2^B,m),(8,k)):((1,T,LBO),(w,SBO))`.
fn canonical_bytes(major: OperandMajorMode, b: u32, bits: usize, mn: usize, k: usize, lbo: usize, sbo: usize) -> usize {
    let t = 128 / bits;
    let w = (1 << b) * t;
    let elems = match major {
        OperandMajorMode::K => {
            assert!(k < w);
            (mn % 8) * w + (mn / 8) * sbo + k
        }
        OperandMajorMode::MN => (mn % w) + (mn / w) * lbo + (k % 8) * w + (k / 8) * sbo,
    };
    make_swizzle(b, 4, 3).apply(elems * bits / 8)
}

/// C++ CuTe documents that `tile_to_shape(Layout_K_SW128_Atom<T>, (128,
/// 64))` is accepted by `make_gmma_desc<Major::K>`: the layout is the PTX
/// ISA's canonical K-major 128-byte-swizzled layout with a stride byte
/// offset of 1024. Checked element by element, and the same for the
/// MN-major SW128 tile and the other swizzles.
#[test]
fn tiled_atoms_are_canonical_wgmma_layouts() {
    for (dtype, kind, b) in [
        (DType::F16, K_SW128, 3),
        (DType::BF16, K_SW64, 2),
        (DType::TF32, K_SW32, 1),
        (DType::F8E4M3FN, K_SW128, 3),
        (DType::F16, K_INTER, 0),
    ] {
        let atom = make_smem_layout_atom_dtype(kind, dtype);
        let w = kind.contiguous_bits() / dtype.bits();
        let tile = tile_to_shape_swizzled(&atom, (128, w));
        let bytes = dtype.bits() / 8;
        for m in 0..128 {
            for k in 0..w {
                let ours = tile.call(m + 128 * k) * bytes;
                // SBO: 8 rows of `w` elements, 1024 bytes for SW128.
                let want = canonical_bytes(OperandMajorMode::K, b, dtype.bits(), m, k, 0, 8 * w);
                assert_eq!(ours, want, "{kind:?} {dtype:?} at ({m},{k})");
            }
        }
        if kind == K_SW128 {
            assert_eq!(8 * w * bytes, 1024);
        }
    }
    // MN-major, tiled K first as make_smem_layout_a does for M-major A:
    // atoms along K at SBO = 8 rows of w, then along M at LBO.
    for (dtype, kind, b) in [(DType::F16, MN_SW128, 3), (DType::BF16, MN_SW32, 1), (DType::F16, MN_SW64, 2)] {
        let atom = make_smem_layout_atom_dtype(kind, dtype);
        let w = kind.contiguous_bits() / dtype.bits();
        let tile = tile_to_shape_swizzled_ordered(&atom, (128, 64), &[1, 0]);
        let bytes = dtype.bits() / 8;
        let (sbo, lbo) = (8 * w, 64 * w);
        for m in 0..128 {
            for k in 0..64 {
                let ours = tile.call(m + 128 * k) * bytes;
                assert_eq!(ours, canonical_bytes(OperandMajorMode::MN, b, dtype.bits(), m, k, lbo, sbo));
            }
        }
    }
}

#[test]
fn layout_enum() {
    use LayoutEnum::*;
    assert_eq!(ROW_MAJOR.value(), "row_major");
    assert_eq!(ROW_MAJOR.mma_major_mode(), OperandMajorMode::K);
    assert_eq!(COL_MAJOR.sm90_mma_major_mode(), OperandMajorMode::MN);
    assert!(ROW_MAJOR.is_k_major_a() && ROW_MAJOR.is_k_major_b() && ROW_MAJOR.is_n_major_c());
    assert!(!ROW_MAJOR.is_m_major_a() && !ROW_MAJOR.is_n_major_b() && !ROW_MAJOR.is_m_major_c());
    assert!(COL_MAJOR.is_m_major_a() && COL_MAJOR.is_n_major_b() && COL_MAJOR.is_m_major_c());
    assert!(!COL_MAJOR.is_k_major_a() && !COL_MAJOR.is_k_major_b() && !COL_MAJOR.is_n_major_c());
    assert_eq!(LayoutEnum::from_layout(&Layout::row_major((64, 32))), ROW_MAJOR);
    assert_eq!(LayoutEnum::from_layout(&Layout::col_major((64, 32))), COL_MAJOR);
    // A batched (M, K, L) layout: the first entry of the leading dimension.
    assert_eq!(LayoutEnum::from_layout(&make_layout(((4, 2), 8, 3), ((8, 32), 1, 64))), ROW_MAJOR);
    let _ = Kernel::builder("layout_enum").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((128, 64)));
        let b = param_tensor::<F16>("b", Layout::col_major((128, 64)));
        assert_eq!(LayoutEnum::from_tensor(&a), ROW_MAJOR);
        assert_eq!(LayoutEnum::from_tensor(&b), COL_MAJOR);
    });
}

/// CuTe DSL raises ValueError for a leading dimension that is neither mode
/// 0 nor mode 1.
#[test]
#[should_panic(expected = "LayoutEnum: invalid leading dimension: Some([2])")]
fn layout_enum_from_a_third_mode() {
    LayoutEnum::from_layout(&make_layout((4, 4, 4), (16, 4, 1)));
}

#[test]
fn smem_layout_atom_choice() {
    use LayoutEnum::*;
    assert_eq!(get_smem_layout_atom(ROW_MAJOR, DType::F16, 64), K_SW128);
    assert_eq!(get_smem_layout_atom(ROW_MAJOR, DType::F16, 32), K_SW64);
    assert_eq!(get_smem_layout_atom(ROW_MAJOR, DType::F16, 16), K_SW32);
    assert_eq!(get_smem_layout_atom(ROW_MAJOR, DType::F16, 8), K_INTER);
    assert_eq!(get_smem_layout_atom(COL_MAJOR, DType::F16, 192), MN_SW128);
    assert_eq!(get_smem_layout_atom(COL_MAJOR, DType::F32, 48), MN_SW64);
    assert_eq!(get_smem_layout_atom(COL_MAJOR, DType::F8E4M3FN, 96), MN_SW32);
    assert_eq!(get_smem_layout_atom(ROW_MAJOR, DType::F8E4M3FN, 128), K_SW128);
    let text = panic_text(|| {
        get_smem_layout_atom(ROW_MAJOR, DType::F16, 12);
    });
    assert!(text.contains("12 is not a multiple of 8"), "{text}");
}

/// `dense_gemm.py`'s default configurations: a (128, 128, 64) f16 tile with
/// both majors and 4 to 7 stages, and an fp8 one.
#[test]
fn staged_operand_layouts() {
    use LayoutEnum::*;
    for stages in 4..=7 {
        let k_major = format!("S<3,3,3> o 0 o ((8,16),(64,1),(1,{stages})):((64,512),(1,0),(0,8192))");
        let mn_major = format!("S<3,3,3> o 0 o ((64,2),(8,8),(1,{stages})):((1,4096),(64,512),(0,8192))");
        let a_k = make_smem_layout_a(ROW_MAJOR, (128, 128, 64), DType::F16, stages);
        let a_m = make_smem_layout_a(COL_MAJOR, (128, 128, 64), DType::F16, stages);
        let b_k = make_smem_layout_b(ROW_MAJOR, (128, 128, 64), DType::F16, stages);
        let b_n = make_smem_layout_b(COL_MAJOR, (128, 128, 64), DType::F16, stages);
        assert_eq!(a_k.to_string(), k_major);
        assert_eq!(b_k.to_string(), k_major);
        assert_eq!(a_m.to_string(), mn_major);
        assert_eq!(b_n.to_string(), mn_major);
        assert_eq!(a_k.cosize(), 128 * 64 * stages);
        assert_eq!(a_k.shape().to_string(), format!("((8,16),(64,1),(1,{stages}))"));
    }
    // B of a (128, 256, 64) tile: N = 256, M-sized tiles differ from B's.
    let b = make_smem_layout_b(ROW_MAJOR, (128, 256, 64), DType::BF16, 3);
    assert_eq!(b.to_string(), "S<3,3,3> o 0 o ((8,32),(64,1),(1,3)):((64,512),(1,0),(0,16384))");
    // FP8, K-major (WGMMA's only major mode for 8-bit operands): 128
    // elements per swizzled row.
    let a8 = make_smem_layout_a(ROW_MAJOR, (128, 256, 128), DType::F8E4M3FN, 4);
    assert_eq!(a8.to_string(), "S<3,4,3> o 0 o ((8,16),(128,1),(1,4)):((128,1024),(1,0),(0,16384))");
    // K = 32 f16 is 512 bits: SW64.
    let a64 = make_smem_layout_a(ROW_MAJOR, (64, 64, 32), DType::F16, 2);
    assert_eq!(a64.to_string(), "S<2,3,3> o 0 o ((8,8),(32,1),(1,2)):((32,256),(1,0),(0,2048))");
}

#[test]
fn staged_epilogue_layouts() {
    use LayoutEnum::*;
    // N-major (row-major) C: the atom from N = 32 f16 (512 bits, SW64).
    let n = make_smem_layout_epi(DType::F16, ROW_MAJOR, (64, 32), 4, None, None);
    assert_eq!(n.to_string(), "S<2,3,3> o 0 o ((8,8),(32,1),(1,4)):((32,256),(1,0),(0,2048))");
    // M-major C: the atom from M = 64 (SW128), tiled N first.
    let m = make_smem_layout_epi(DType::F16, COL_MAJOR, (64, 32), 4, None, None);
    assert_eq!(m.to_string(), "S<3,3,3> o 0 o ((64,1),(8,4),(1,4)):((1,0),(64,512),(0,2048))");
    // An explicit target shape and order.
    let t = make_smem_layout_epi(DType::F32, ROW_MAJOR, (64, 32), 2, Some((64, 32, 2).into()), Some(&[1, 0, 2]));
    assert_eq!(t.to_string(), "S<3,2,3> o 0 o ((8,8),(32,1),(1,2)):((32,256),(1,0),(0,2048))");
}

//! The SM100 shared-memory layout atoms
//! (`tcgen05.make_smem_layout_atom`, `make_sparse_smem_layout_atom`),
//! `tcgen05.tile_to_mma_shape`, the UMMA shared-memory descriptors and
//! their decoder, the block-scaled scale-factor layouts, the TMEM
//! accumulator columns, the epilogue tile and the TMEM capacity helpers.
//! Host-side code: no kernel runs on a GPU (one runs in the simulator).
//!
//! The expected values are written out from CuTe C++'s definitions
//! (`include/cute/atom/mma_traits_sm100.hpp`, `include/cute/arch/mma_sm100_desc.hpp`,
//! CUTLASS's `sm100_blockscaled_layout.hpp`) and the PTX ISA, independently
//! of cute-rs's layout algebra.

use cute::prelude::*;
use cute::tcgen05::{
    SmemDescriptor, SmemLayoutAtomKind, SmemLayoutAtomKind::*, UmmaDescEncoding, UmmaLayoutType, UmmaOperandMap,
    make_smem_layout_atom, make_smem_layout_atom_dtype, make_sparse_smem_layout_atom_dtype, make_umma_smem_desc,
    tile_to_mma_shape, umma_desc_fields,
};
use cute::utils::sm100::{compute_acc_tmem_cols_per_stage, compute_epilogue_tile_size};
use cute::utils::{
    BlockScaledBasicChunk, LayoutEnum, Sm103BlockScaledBasicChunk, compute_epilogue_tile_shape, tile_atom_to_shape_sf,
};

fn panic_text(f: impl FnOnce()) -> String {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

/// A byte swizzle `Swizzle<B,M,S>` on a byte address, written out: bits
/// `[M+S, M+S+B)` XORed into `[M, M+B)`.
fn byte_swizzle(b: u32, m: u32, s: u32, addr: usize) -> usize {
    let mask = (1usize << b) - 1;
    addr ^ (((addr >> (m + s)) & mask) << m)
}

/// `(B, M, S)` of each kind's byte swizzle and its rows, contiguous bits and
/// MN-majorness, as CuTe C++ defines the atoms (`GMMA::Layout_*_Atom`,
/// `UMMA::Layout_MN_SW128_32B_Atom`).
fn cpp_atom(kind: SmemLayoutAtomKind) -> ((u32, u32, u32), usize, bool) {
    match kind {
        MN_INTER => ((0, 4, 3), 128, true),
        MN_SW32 => ((1, 4, 3), 256, true),
        MN_SW64 => ((2, 4, 3), 512, true),
        MN_SW128 => ((3, 4, 3), 1024, true),
        MN_SW128_32B => ((2, 5, 2), 1024, true),
        K_INTER => ((0, 4, 3), 128, false),
        K_SW32 => ((1, 4, 3), 256, false),
        K_SW64 => ((2, 4, 3), 512, false),
        K_SW128 => ((3, 4, 3), 1024, false),
    }
}

/// The bit address of element `(i, j)` of a kind's atom over elements of
/// `bits` bits, from the C++ definition: `(c,8):(1,c)` (MN) or
/// `(8,c):(c,1)` (K) in elements, `c = contiguous bits / bits`, then the
/// byte swizzle on the byte address.
fn cpp_atom_bit(kind: SmemLayoutAtomKind, bits: usize, i: usize, j: usize) -> usize {
    let ((b, m, s), cbits, mn) = cpp_atom(kind);
    let c = cbits / bits;
    let elem = if mn { i + j * c } else { i * c + j };
    let bit = elem * bits;
    byte_swizzle(b, m, s, bit / 8) * 8 + bit % 8
}

/// Every kind for 4-, 8-, 16- and 32-bit elements, element by element, and
/// the printed form of a few.
#[test]
fn atoms_match_the_cpp_definitions() {
    for dtype in [DType::F4E2M1FN, DType::I8, DType::F8E4M3FN, DType::F16, DType::BF16, DType::TF32, DType::F32] {
        let bits = dtype.bits();
        for kind in SmemLayoutAtomKind::ALL {
            let atom = make_smem_layout_atom_dtype(kind, dtype);
            let c = kind.contiguous_bits() / bits;
            let (rows, cols) = if kind.major_mode() == OperandMajorMode::MN { (c, kind.rows()) } else { (8, c) };
            assert_eq!(atom.size(), rows * cols, "{kind:?} {dtype:?}");
            for j in 0..cols {
                for i in 0..rows {
                    let got = atom.call(i + rows * j) * bits;
                    assert_eq!(got, cpp_atom_bit(kind, bits, i, j), "{kind:?} {dtype:?} ({i}, {j}): {atom}");
                }
            }
            // The shared kinds are the SM90 atoms.
            if let Some(k) = kind.sm90() {
                assert_eq!(atom, cute::warpgroup::make_smem_layout_atom_dtype(k, dtype));
            }
        }
    }
    let show = |kind, t| make_smem_layout_atom_dtype(kind, t).to_string();
    assert_eq!(show(MN_SW128_32B, DType::F16), "S<2,4,2> o 0 o (64,4):(1,64)");
    assert_eq!(show(MN_SW128_32B, DType::I8), "S<2,5,2> o 0 o (128,4):(1,128)");
    assert_eq!(show(MN_SW128_32B, DType::F32), "S<2,3,2> o 0 o (32,4):(1,32)");
    assert_eq!(show(MN_SW128_32B, DType::F4E2M1FN), "S<2,6,2> o 0 o (256,4):(1,256)");
    assert_eq!(show(K_SW128, DType::F16), "S<3,3,3> o 0 o (8,64):(64,1)");
    assert_eq!(make_smem_layout_atom::<F16>(MN_SW64).to_string(), "S<2,3,3> o 0 o (32,8):(1,32)");
    // 6-bit floats do not divide the atoms' rows.
    let text = panic_text(|| drop(make_smem_layout_atom_dtype(K_SW128, DType::F6E2M3FN)));
    assert!(text.contains("6 bits wide"), "{text}");
}

/// CuTe DSL's 8-row MN_SW128_32B atom is the 4-row atom tiled twice.
#[test]
fn mn_sw128_32b_atom_tiles_to_the_dsl_atom() {
    for dtype in [DType::I8, DType::F16, DType::F32] {
        let atom = make_smem_layout_atom_dtype(MN_SW128_32B, dtype);
        let c = 1024 / dtype.bits();
        let dsl = make_composed_layout(atom.swizzle().unwrap(), 0, Layout::new((c, 8), (1, c)));
        let tiled = tile_to_shape(&atom, (c, 8));
        for i in 0..dsl.size() {
            assert_eq!(dsl.call(i), tiled.call(i), "{dtype:?} index {i}");
        }
    }
}

/// The sparse atoms: `blocked_product((1,S), L)` of the dense outer layout,
/// in units of `width / S` bits, against the offsets written out:
/// MN-major `i S + s + j S c`, K-major `r S c + k` (with `k = s + S j`).
#[test]
fn sparse_atoms_match_the_cpp_aliases() {
    for (dtype, s) in [(DType::I8, 2), (DType::F16, 2), (DType::F32, 2), (DType::F32, 4)] {
        let bits = dtype.bits();
        let unit = bits / s;
        for kind in SmemLayoutAtomKind::ALL {
            let atom = make_sparse_smem_layout_atom_dtype(kind, dtype, s);
            let ((b, m, sh), cbits, mn) = cpp_atom(kind);
            let c = cbits / bits;
            let (rows, cols) = if mn { (c, kind.rows() * s) } else { (8, c * s) };
            assert_eq!(atom.size(), rows * cols, "{kind:?} {dtype:?} S={s}: {atom}");
            for col in 0..cols {
                for row in 0..rows {
                    let units = if mn {
                        let (ss, j) = (col % s, col / s);
                        row * s + ss + j * s * c
                    } else {
                        row * s * c + col
                    };
                    let bit = units * unit;
                    let want = byte_swizzle(b, m, sh, bit / 8) * 8 + bit % 8;
                    let got = atom.call(row + rows * col) * unit;
                    assert_eq!(got, want, "{kind:?} {dtype:?} S={s} ({row}, {col}): {atom}");
                }
            }
        }
    }
    assert_eq!(
        make_sparse_smem_layout_atom_dtype(K_SW128, DType::F16, 2).to_string(),
        "S<3,4,3> o 0 o ((1,8),(2,64)):((1,128),(1,2))"
    );
}

/// The element offset of `(m, k, stage)` in an atom tiled to `(M, K,
/// stages)` in the mode order `order`, written out: the atom's own offset
/// plus the atom's cosize times the repeat's index, counted in `order`.
fn tiled_offset(
    atom: &ComposedLayout,
    (am, ak): (usize, usize),
    (m_total, k_total, stages): (usize, usize, usize),
    order: [usize; 3],
    (m, k, stage): (usize, usize, usize),
) -> usize {
    let repeats = [m_total / am, k_total / ak, stages];
    let q = [m / am, k / ak, stage];
    let mut by_order = [0, 1, 2];
    by_order.sort_by_key(|&i| order[i]);
    let (mut index, mut scale) = (0, 1);
    for i in by_order {
        index += q[i] * scale;
        scale *= repeats[i];
    }
    let outer = atom.outer();
    outer.call((m % am) + am * (k % ak)) + index * am * ak
}

/// `tile_to_mma_shape` of K- and MN-major atoms to `((128,16),1,4,3)`, in
/// the orders `make_smem_layout_a/b` use, against the written-out tiling.
#[test]
fn tile_to_mma_shape_tiles_then_divides() {
    for (kind, order) in [
        (K_SW128, [0, 1, 2]),
        (K_SW128, [1, 0, 2]),
        (K_SW32, [0, 1, 2]),
        (K_SW32, [1, 0, 2]),
        (MN_SW128, [1, 0, 2]),
        (MN_SW128_32B, [1, 0, 2]),
        (MN_INTER, [0, 1, 2]),
    ] {
        let atom = make_smem_layout_atom::<F16>(kind);
        let a = (atom.outer().mode(0).size(), atom.outer().mode(1).size());
        let staged = tile_to_mma_shape(&atom, Shape::from(((128, 16), 1, 4, 3)), Some(&order));
        assert_eq!(staged.inner(), atom.inner(), "the swizzle is kept");
        let s = staged.shape().as_static().unwrap();
        assert_eq!(s.product_each().to_string(), "(2048,1,4,3)", "{kind:?}: {staged}");
        assert_eq!(s.mode(0).product_each().to_string(), "(128,16)", "{kind:?}: {staged}");
        let sw = atom.swizzle().unwrap();
        for stage in 0..3 {
            for kb in 0..4 {
                for kk in 0..16 {
                    for m in 0..128 {
                        let idx = m + 128 * (kk + 16 * (kb + 4 * stage));
                        let want = tiled_offset(&atom, a, (128, 64, 3), order, (m, kb * 16 + kk, stage));
                        assert_eq!(staged.call(idx), sw.apply(want), "{kind:?} {order:?} ({m},{kk},{kb},{stage})");
                    }
                }
            }
        }
    }
    // The orders differ where the atom repeats along both modes.
    let atom = make_smem_layout_atom::<F16>(K_SW32);
    let col = tile_to_mma_shape(&atom, Shape::from(((128, 16), 1, 4, 3)), None);
    let row = tile_to_mma_shape(&atom, Shape::from(((128, 16), 1, 4, 3)), Some(&[1, 0, 2]));
    assert_ne!(col, row);
    assert_eq!(col, tile_to_mma_shape(&atom, Shape::from(((128, 16), 1, 4, 3)), Some(&[0, 1, 2])));
    // A plain layout atom too.
    let plain = tile_to_mma_shape(atom.outer(), Shape::from(((64, 16), 2, 4)), None);
    assert_eq!(plain.shape().as_static().unwrap().product_each().to_string(), "(1024,2,4)");
    let text = panic_text(|| drop(tile_to_mma_shape(&atom, Shape::from((128, 16, 4)), None)));
    assert!(text.contains("((MMA_MN, MMA_K), REST_MN, REST_K, ...)"), "{text}");
}

#[test]
fn descriptor_fields_round_trip() {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for enc in [UmmaDescEncoding::Sm100, UmmaDescEncoding::Sm107] {
        let wide = (1u64 << enc.address_bits()) - 1;
        for _ in 0..1000 {
            let d = SmemDescriptor {
                start_address: (next() & wide) as u32,
                leading_byte_offset: (next() & wide) as u32,
                stride_byte_offset: (next() & 0x3fff) as u32,
                version: (next() & 3) as u8,
                base_offset: (next() & 7) as u8,
                lbo_mode: next() & 1 == 1,
                layout_type: UmmaLayoutType::ALL[(next() % 5) as usize],
            };
            let bits = d.encode(enc);
            assert_eq!(SmemDescriptor::decode(bits, enc), Ok(d));
            // Unused bits stay clear.
            let used = wide | wide << 16 | 0xffff << 32 | 0xf << 49 | 7 << 61;
            assert_eq!(bits & !used, 0, "{bits:#x}");
        }
    }
    // The fields, by hand: start 0x40, LBO 1, SBO 64, version 1, 128-byte swizzle.
    let d = SmemDescriptor {
        start_address: 0x40,
        leading_byte_offset: 1,
        stride_byte_offset: 64,
        version: 1,
        base_offset: 0,
        lbo_mode: false,
        layout_type: UmmaLayoutType::Swizzle128B,
    };
    assert_eq!(d.encode(UmmaDescEncoding::Sm100), 0x4000_4040_0001_0040);
    // 15-bit start and LBO on sm_107 only.
    let big = SmemDescriptor { start_address: 0x7fff, leading_byte_offset: 0x4000, ..d };
    assert_eq!(big.encode(UmmaDescEncoding::Sm107) & 0x7fff_7fff, 0x4000_7fff);
    let text = panic_text(|| {
        big.encode(UmmaDescEncoding::Sm100);
    });
    assert!(text.contains("14-bit field"), "{text}");
    assert_eq!(UmmaDescEncoding::for_arch(Arch::Sm107a), UmmaDescEncoding::Sm107);
    assert_eq!(UmmaDescEncoding::for_arch(Arch::Sm100a), UmmaDescEncoding::Sm100);
    assert!(SmemDescriptor::decode(3 << 61, UmmaDescEncoding::Sm100).is_err());
}

/// Descriptors computed by hand from CuTe C++'s `make_umma_desc` for f16
/// tiles of `tile_to_shape(atom, (128, 64))`, one 128 x 16 MMA tile.
#[test]
fn descriptors_match_hand_computed_values() {
    let fields = |kind: SmemLayoutAtomKind, ty: UmmaLayoutType| {
        let atom = make_smem_layout_atom::<F16>(kind);
        let tiled = tile_to_shape(atom.outer(), (128, 64));
        let tile = tiled_divide(&tiled, (128, 16)).mode(0);
        umma_desc_fields(&tile, kind.major_mode(), ty, 16, UmmaDescEncoding::Sm100)
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"))
            .encode(UmmaDescEncoding::Sm100)
    };
    // K-major SW128: rows of 8 units, 8-row groups 1024 bytes apart (SBO 64), LBO 1.
    assert_eq!(fields(K_SW128, UmmaLayoutType::Swizzle128B), 1 << 16 | 64 << 32 | 1 << 46 | 2 << 61);
    // K-major INTER: 8-row groups 128 bytes apart (SBO 8), K halves 2048 bytes apart (LBO 128).
    assert_eq!(fields(K_INTER, UmmaLayoutType::SwizzleNone), 128 << 16 | 8 << 32 | 1 << 46);
    // MN-major SW128: 64-element MN blocks 1024 bytes apart (LBO 64), 8-row K blocks 2048 (SBO 128).
    assert_eq!(fields(MN_SW128, UmmaLayoutType::Swizzle128B), 64 << 16 | 128 << 32 | 1 << 46 | 2 << 61);
    // MN-major SW128 with 32-byte atomicity is for 32-bit elements (the hardware reads other
    // widths wrongly): for TF32 tiles of `tile_to_shape(atom, (128, 64))`, MN blocks 512 bytes
    // apart (LBO 32), 4-row K blocks 2048 (SBO 128).
    let tf32 = {
        let atom = make_smem_layout_atom_dtype(MN_SW128_32B, DType::TF32);
        let tiled = tile_to_shape(atom.outer(), (128, 64));
        let tile = tiled_divide(&tiled, (128, 8)).mode(0);
        let ty = UmmaLayoutType::Swizzle128BBase32B;
        umma_desc_fields(&tile, OperandMajorMode::MN, ty, 32, UmmaDescEncoding::Sm100)
            .unwrap_or_else(|e| panic!("MN_SW128_32B TF32: {e}"))
            .encode(UmmaDescEncoding::Sm100)
    };
    assert_eq!(tf32, 32 << 16 | 128 << 32 | 1 << 46 | 1 << 61);
    // The pairings the hardware does not take are refused.
    let refused = |kind: SmemLayoutAtomKind, ty: UmmaLayoutType, dtype: DType| {
        let atom = make_smem_layout_atom_dtype(kind, dtype);
        let tiled = tile_to_shape(atom.outer(), (128, 64));
        let tile = tiled_divide(&tiled, (128, 256 / dtype.bits())).mode(0);
        umma_desc_fields(&tile, OperandMajorMode::MN, ty, dtype.bits(), UmmaDescEncoding::Sm100).unwrap_err()
    };
    let e = refused(MN_SW128_32B, UmmaLayoutType::Swizzle128BBase32B, DType::F16);
    assert!(e.contains("32-bit elements, and these are 16-bit"), "{e}");
    let e = refused(MN_SW128, UmmaLayoutType::Swizzle128B, DType::TF32);
    assert!(e.contains("needs the 128-byte swizzle with 32-byte atomicity"), "{e}");
    // MN-major SW32: 16-element MN blocks 256 bytes apart (LBO 16), 8-row K blocks 2048 (SBO 128).
    assert_eq!(fields(MN_SW32, UmmaLayoutType::Swizzle32B), 16 << 16 | 128 << 32 | 1 << 46 | 6 << 61);
}

/// Which descriptor swizzle and atom go together.
fn kinds_for(ty: UmmaLayoutType, major: OperandMajorMode) -> Option<SmemLayoutAtomKind> {
    use OperandMajorMode::*;
    use UmmaLayoutType::*;
    Some(match (ty, major) {
        (SwizzleNone, MN) => MN_INTER,
        (Swizzle32B, MN) => MN_SW32,
        (Swizzle64B, MN) => MN_SW64,
        (Swizzle128B, MN) => MN_SW128,
        (Swizzle128BBase32B, MN) => MN_SW128_32B,
        (SwizzleNone, K) => K_INTER,
        (Swizzle32B, K) => K_SW32,
        (Swizzle64B, K) => K_SW64,
        (Swizzle128B, K) => K_SW128,
        (Swizzle128BBase32B, K) => return None,
    })
}

/// For every swizzle, both majors, 8-, 16- and 32-bit elements and several
/// tile sizes: the descriptor of each MMA tile of a staged layout, decoded
/// as the PTX ISA defines the canonical layouts and swizzles, reaches
/// exactly the bytes the swizzled layout addresses.
#[test]
fn decoded_descriptors_visit_the_swizzled_layouts_bytes() {
    let base = 0x2400usize; // 1024-byte aligned
    let mut checked = 0;
    for ty in UmmaLayoutType::ALL {
        for major in [OperandMajorMode::K, OperandMajorMode::MN] {
            let Some(kind) = kinds_for(ty, major) else { continue };
            for dtype in [DType::I8, DType::F16, DType::TF32] {
                let bits = dtype.bits();
                // An MN-major operand takes the 32-byte-atomicity swizzle exactly when its
                // elements are 32 bits wide (`umma_desc_fields` refuses the other pairings).
                if major == OperandMajorMode::MN && (ty == UmmaLayoutType::Swizzle128BBase32B) != (bits == 32) {
                    continue;
                }
                let mma_k = 256 / bits;
                for mn in [64usize, 128, 256] {
                    let atom = make_smem_layout_atom_dtype(kind, dtype);
                    let block = if major == OperandMajorMode::MN { atom.outer().mode(0).size() } else { 8 };
                    if mn % block != 0 {
                        continue;
                    }
                    let order: &[usize] = if major == OperandMajorMode::MN { &[1, 0, 2] } else { &[0, 1, 2] };
                    let staged = tile_to_mma_shape(&atom, Shape::from(((mn, mma_k), 2, 4, 2)), Some(order));
                    let sw = staged.swizzle().unwrap_or(Swizzle::identity());
                    let outer = staged.outer();
                    let tile = outer.mode(0);
                    let fields = umma_desc_fields(&tile, major, ty, bits, UmmaDescEncoding::Sm100)
                        .unwrap_or_else(|e| panic!("{kind:?} {dtype:?} {mn}: {e}"));
                    for stage in 0..2 {
                        for kb in 0..4 {
                            for rm in 0..2 {
                                // The tile's first element, then each element of the tile.
                                let first = mn * mma_k * (rm + 2 * (kb + 4 * stage));
                                let start = base + outer.call(first) * bits / 8;
                                let desc = fields.with_start_address(start as u32).encode(UmmaDescEncoding::Sm100);
                                let map =
                                    UmmaOperandMap::new(desc, UmmaDescEncoding::Sm100, major, bits, mn, mma_k).unwrap();
                                for kk in 0..mma_k {
                                    for m in 0..mn {
                                        let want = base + sw.apply(outer.call(first + m + mn * kk)) * bits / 8;
                                        assert_eq!(
                                            map.byte_address(m, kk) as usize,
                                            want,
                                            "{kind:?} {dtype:?} mn={mn} ({m},{kk}) tile ({rm},{kb},{stage})"
                                        );
                                    }
                                }
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 300, "{checked}");
}

/// Non-canonical tiles are refused, naming the expected form.
#[test]
fn non_canonical_tiles_are_refused() {
    let k_major = Layout::new((128, 16), (16, 1));
    // A K-major layout read as MN-major.
    let e = umma_desc_fields(&k_major, OperandMajorMode::MN, UmmaLayoutType::Swizzle128B, 16, UmmaDescEncoding::Sm100)
        .unwrap_err();
    assert!(e.contains("expected ((8,8,2),(8,2))"), "{e}");
    // Rows 32 bytes apart are not the 128-byte swizzle's.
    let e = umma_desc_fields(&k_major, OperandMajorMode::K, UmmaLayoutType::Swizzle128B, 16, UmmaDescEncoding::Sm100)
        .unwrap_err();
    assert!(e.contains("expected ((8,16),(8,2)):((64,"), "{e}");
    // K-major tiles cover 32 bytes of K.
    let e = umma_desc_fields(
        &Layout::new((128, 32), (32, 1)),
        OperandMajorMode::K,
        UmmaLayoutType::SwizzleNone,
        16,
        UmmaDescEncoding::Sm100,
    )
    .unwrap_err();
    assert!(e.contains("32 bytes of K"), "{e}");
    let e = umma_desc_fields(
        &Layout::new((128, 16), (64, 1)),
        OperandMajorMode::K,
        UmmaLayoutType::Swizzle128BBase32B,
        16,
        UmmaDescEncoding::Sm100,
    )
    .unwrap_err();
    assert!(e.contains("only for MN-major"), "{e}");
}

/// `make_umma_smem_desc` in a kernel: the build-time fields ORed with the
/// operand's address `>> 4`, simulated.
#[test]
fn make_umma_smem_desc_in_a_kernel() {
    let atom = make_smem_layout_atom::<F16>(K_SW128);
    let staged = tile_to_mma_shape(&atom, Shape::from(((128, 16), 1, 4, 2)), None);
    let tile = staged.outer().mode(0);
    let kernel = Kernel::builder("umma_desc").block_dim([32, 1, 1]).target(Arch::Sm100a).build({
        let staged = staged.clone();
        let tile = tile.clone();
        move || {
            let out = param_tensor::<u64>("out", Layout::col_major((2,)));
            let _pad = alloc_smem::<u8>(1024, 1024);
            let smem = alloc_smem::<F16>(staged.cosize(), 1024);
            let src = recast_ptr::<F16, F16>(&smem, Some(Swizzle::new(3, 4, 3)));
            let t = thread_idx().x;
            if_(t.eq(0), || {
                out.store(0, make_umma_smem_desc(&src, &tile, OperandMajorMode::K, None));
                // The second k-block: 32 bytes on.
                out.store(1, make_umma_smem_desc(&src.offset(16), &tile, OperandMajorMode::K, None));
            });
        }
    });
    let mut args = [KernelArg::buffer::<u64>(&[0, 0])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got = args[0].to_vec::<u64>();
    let fields =
        umma_desc_fields(&tile, OperandMajorMode::K, UmmaLayoutType::Swizzle128B, 16, UmmaDescEncoding::Sm100).unwrap();
    let start = kernel.ir().shared_memory_layout().starts.iter().flatten().copied().max().unwrap() as u32;
    assert_eq!(got[0], fields.with_start_address(start).encode(UmmaDescEncoding::Sm100));
    assert_eq!(got[1], fields.with_start_address(start + 32).encode(UmmaDescEncoding::Sm100));
    assert_eq!(got[0] >> 14 << 14, 0x4000_4040_0001_0000);

    // Misuse is refused when the kernel is built.
    let text = panic_text(|| {
        drop(Kernel::builder("bad").build(|| {
            let smem = alloc_smem::<F16>(2048, 1024);
            let src = recast_ptr::<F16, F16>(&smem, Some(Swizzle::new(3, 3, 3)));
            let _ = make_umma_smem_desc(&src, &Layout::new((128, 16), (64, 1)), OperandMajorMode::K, None);
        }))
    });
    assert!(text.contains("none of UMMA's"), "{text}");
    let text = panic_text(|| {
        drop(Kernel::builder("next").build(|| {
            let smem = alloc_smem::<F16>(2048, 1024);
            let _ = make_umma_smem_desc(&smem, &Layout::new((128, 16), (16, 1)), OperandMajorMode::K, Some(&smem));
        }))
    });
    assert!(text.contains("next_src"), "{text}");
}

/// The scale-factor offset of operand element `(mn, k, l)`, from CuTe C++'s
/// `Sm1xxBlockScaledBasicChunk` written per element: 512-byte atoms of 128
/// rows by 4 scale factors, K atoms innermost, then MN atoms, then L; in an
/// atom, row `r` is at `(r % 32) * 16 + (r / 32) * 4` and its scale factor
/// `j` at `+ j`.
fn sf_offset(mn: usize, k: usize, l: usize, (mn_total, k_total): (usize, usize), vec: usize) -> usize {
    let k_atoms = k_total.div_ceil(vec * 4);
    let mn_atoms = mn_total.div_ceil(128);
    let atom = (k / (vec * 4)) + k_atoms * (mn / 128) + k_atoms * mn_atoms * l;
    let r = mn % 128;
    let j = (k / vec) % 4;
    atom * 512 + (r % 32) * 16 + (r / 32) * 4 + j
}

#[test]
fn scale_factor_layouts() {
    for vec in [16, 32] {
        let k = BlockScaledBasicChunk::new(vec, OperandMajorMode::K);
        assert_eq!(k.layout().to_string(), format!("((32,4),({vec},4)):((16,4),(0,1))"));
        let m = BlockScaledBasicChunk::new(vec, OperandMajorMode::MN);
        assert_eq!(m.layout().to_string(), format!("(({vec},4),(32,4)):((0,1),(16,4))"));
        assert_eq!(k.layout().cosize(), 512);
        let s = Sm103BlockScaledBasicChunk::new(vec, OperandMajorMode::K);
        assert_eq!(s.layout().to_string(), format!("((8,4,4),({vec},4)):((16,128,4),(0,1))"));
        let s = Sm103BlockScaledBasicChunk::new(vec, OperandMajorMode::MN);
        assert_eq!(s.layout().to_string(), format!("(({vec},4),(8,4,4)):((0,1),(16,128,4))"));

        let (mn_total, k_total, l_total) = (256, 8 * vec, 2);
        let sf = tile_atom_to_shape_sf(Shape::from((mn_total, k_total, l_total)), vec);
        let s = sf.shape().as_static().unwrap();
        assert_eq!(s.product_each().to_string(), format!("({mn_total},{k_total},{l_total})"));
        for l in 0..l_total {
            for kk in 0..k_total {
                for r in 0..mn_total {
                    let idx = r + mn_total * (kk + k_total * l);
                    assert_eq!(sf.call(idx), sf_offset(r, kk, l, (mn_total, k_total), vec), "vec {vec} ({r},{kk},{l})");
                }
            }
        }
        // Rank 2: no L.
        let sf2 = tile_atom_to_shape_sf(Shape::from((128, 4 * vec)), vec);
        assert_eq!(sf2.cosize(), 512);
    }
}

#[test]
fn accumulator_tmem_columns() {
    // 2-CTA: a quarter, a half or all of N.
    assert_eq!(compute_acc_tmem_cols_per_stage(32, 256, true, 256, false), 64);
    assert_eq!(compute_acc_tmem_cols_per_stage(64, 256, true, 256, false), 128);
    assert_eq!(compute_acc_tmem_cols_per_stage(128, 256, true, 256, false), 256);
    // 1-CTA, A from shared memory, M 64: interleaved pairs of N tiles.
    assert_eq!(compute_acc_tmem_cols_per_stage(64, 256, false, 64, false), 128);
    assert_eq!(compute_acc_tmem_cols_per_stage(64, 192, false, 64, false), 128);
    assert_eq!(compute_acc_tmem_cols_per_stage(64, 64, false, 64, false), 64);
    // A from TMEM, or M 128: all of N.
    assert_eq!(compute_acc_tmem_cols_per_stage(64, 256, false, 64, true), 256);
    assert_eq!(compute_acc_tmem_cols_per_stage(128, 256, false, 64, false), 256);
}

#[test]
fn epilogue_tiles() {
    use LayoutEnum::*;
    let tiler = |cta: &[usize], two: bool, d: LayoutEnum, dt: DType, c: Option<(LayoutEnum, DType)>| {
        let t = compute_epilogue_tile_shape(cta, two, d, dt, c.map(|x| x.0), c.map(|x| x.1), None);
        format!(
            "({}, {})",
            t.mode(0).as_leaf().unwrap().as_ref().unwrap(),
            t.mode(1).as_leaf().unwrap().as_ref().unwrap()
        )
    };
    // dense_gemm.py's default: MMA tiler (256,128) on 2 CTAs, Float16 D, no C.
    assert_eq!(tiler(&[128, 128, 64], true, ROW_MAJOR, DType::F16, None), "(128:1, 32:1)");
    // The tutorials' 1-CTA (128,256) tile.
    assert_eq!(tiler(&[128, 256], false, ROW_MAJOR, DType::F16, None), "(128:1, 32:1)");
    // 2-CTA with a per-CTA M of 64: two warps along N, each taking its half of N.
    assert_eq!(tiler(&[64, 128], true, ROW_MAJOR, DType::F16, None), "(64:1, (32,2):(1,64))");
    assert_eq!(compute_epilogue_tile_size(64, 128, true, 16, None, false, true, None), (64, 64));
    // With a source C: by width and tile.
    assert_eq!(compute_epilogue_tile_size(128, 128, false, 32, Some(32), false, false, None), (128, 16));
    assert_eq!(compute_epilogue_tile_size(64, 128, false, 32, Some(32), false, false, None), (64, 32));
    assert_eq!(compute_epilogue_tile_size(128, 256, false, 16, Some(16), false, false, None), (128, 64));
    assert_eq!(compute_epilogue_tile_size(128, 128, false, 8, Some(8), true, true, None), (128, 64));
    // 4-bit D: 8192 elements a subtile.
    assert_eq!(compute_epilogue_tile_size(128, 256, false, 4, None, true, true, None), (128, 64));
    // The target is halved until it divides N: 4096 / 128 = 32 does not divide 48.
    assert_eq!(compute_epilogue_tile_size(128, 48, false, 16, None, true, true, None), (128, 16));
    // N-major FP6 needs 128 elements a warp.
    assert_eq!(compute_epilogue_tile_size(128, 256, false, 6, None, false, true, None), (128, 128));
    // An explicit warp shape.
    assert_eq!(compute_epilogue_tile_size(128, 128, false, 16, None, false, true, Some((2, 2))), (64, 64));
    let text = panic_text(|| drop(tiler(&[128, 64], false, ROW_MAJOR, DType::F6E2M3FN, None)));
    assert!(text.contains("CTA tile too small"), "{text}");
}

#[test]
// Tests the deprecated names, which still have to work.
#[allow(deprecated)]
fn tmem_capacity() {
    for (arch, max) in [
        (Arch::Sm100, 512),
        (Arch::Sm100a, 512),
        (Arch::Sm100f, 512),
        (Arch::Sm103a, 512),
        (Arch::Sm107, 576),
        (Arch::Sm107a, 576),
        (Arch::Sm107f, 576),
        (Arch::Sm120a, 512),
    ] {
        assert_eq!(get_max_tmem_alloc_cols(arch), max, "{arch}");
        assert_eq!(get_min_tmem_alloc_cols(arch), 32, "{arch}");
    }
    for arch in [Arch::Sm90a, Arch::Sm110a, Arch::Sm121a] {
        assert!(
            panic_text(|| {
                get_max_tmem_alloc_cols(arch);
            })
            .contains("unsupported")
        );
        assert!(
            panic_text(|| {
                get_min_tmem_alloc_cols(arch);
            })
            .contains("unsupported")
        );
    }
    assert!(is_tmem_allocation_exclusive(576, Arch::Sm107a));
    assert!(!is_tmem_allocation_exclusive(512, Arch::Sm107));
    assert!(!is_tmem_allocation_exclusive(576, Arch::Sm100a));
    assert_eq!((SM100_TMEM_CAPACITY_COLUMNS, SM100_TMEM_MIN_ALLOC_COLUMNS), (512, 32));
}

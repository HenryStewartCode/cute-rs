// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! The `mma.sync` cases, dense and sparse, and the ldmatrix and tiled-copy kernels beside them.

use super::*;
/// The `mma.sync` kind of an SM80 MMA operation.
fn sm80_kind(op: impl Into<MmaOp>) -> MmaKind {
    match op.into() {
        MmaOp::Sm80(kind) => kind,
        other => panic!("{other:?} is not an mma.sync"),
    }
}

/// One warp, one `mma.sync` per tile of K, `K_TILES` tiles.
fn mma_kernel<TA: Numeric, TC: Numeric>(name: &'static str, op: MmaOp, zero: TC) -> Kernel {
    let (m, n, k_atom) = sm80_kind(op).shape_mnk();
    let k = k_atom * K_TILES;
    Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<TA>("a", Layout::col_major((m, k)));
        let b = param_tensor::<TA>("b", Layout::col_major((n, k)));
        let c = param_tensor::<TC>("c", Layout::col_major((m, n)));
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let (ta, tb, tc) = (thr.partition_a(&a), thr.partition_b(&b), thr.partition_c(&c));
        let (fa, fb) = (thr.partition_fragment_a::<TA, _>(&a), thr.partition_fragment_b::<TA, _>(&b));
        let acc = thr.partition_fragment_c::<TC, _>(&c);
        for i in 0..fa.size() {
            fa.store(i, ta.load(i));
        }
        for i in 0..fb.size() {
            fb.store(i, tb.load(i));
        }
        acc.fill(zero);
        gemm(&mma, &acc, &fa, &fb, &acc);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    })
}

const K_TILES: usize = 2;

macro_rules! mma_case {
    ($name:ident, $op:expr, $ta:ty, $tc:ty, $zero:expr, $as_f64:expr) => {
        pub fn $name() -> Case {
            let kernel = mma_kernel::<$ta, $tc>(stringify!($name), $op.into(), $zero);
            fn check(args: &[KernelArg]) {
                let as_a: fn($ta) -> f64 = $as_f64.0;
                let as_c: fn($tc) -> f64 = $as_f64.1;
                let (m, n, k_atom) = sm80_kind($op).shape_mnk();
                let k = k_atom * K_TILES;
                let (a, b, c) = (args[0].to_vec::<$ta>(), args[1].to_vec::<$ta>(), args[2].to_vec::<$tc>());
                for i in 0..m {
                    for j in 0..n {
                        let want: f64 = (0..k).map(|l| as_a(a[i + m * l]) * as_a(b[j + n * l])).sum();
                        assert_eq!(as_c(c[i + m * j]), want, "C[{i},{j}]");
                    }
                }
            }
            Case {
                name: stringify!($name),
                kernel,
                cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]),
                small_only: true,
                check,
            }
        }
    };
}

mma_case!(mma_k8, MmaKind::M16N8K8F32F16, F16, f32, 0.0f32, (|x: F16| x.to_f32() as f64, |x: f32| x as f64));
mma_case!(
    mma_f16_accumulate,
    MmaKind::M16N8K16F16F16,
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
mma_case!(mma_tf32, MmaKind::M16N8K4Tf32, f32, f32, 0.0f32, (|x: f32| x as f64, |x: f32| x as f64));
mma_case!(mma_f64, MmaKind::M8N8K4F64, f64, f64, 0.0f64, (|x: f64| x, |x: f64| x));
mma_case!(
    mma_bf16,
    MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 16)),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);
mma_case!(
    mma_bf16_k8,
    MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 8)),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);
mma_case!(
    mma_f16_accumulate_k8,
    MmaF16BF16Op::new(DType::F16, DType::F16, (16, 8, 8)),
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
mma_case!(mma_tf32_k8, MmaTF32Op::new((16, 8, 8)), f32, f32, 0.0f32, (|x: f32| x as f64, |x: f32| x as f64));

mma_case!(
    mma_fp8_e4m3_f32_k32,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)),
    F8E4M3FN,
    f32,
    0.0f32,
    (|x: F8E4M3FN| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e4m3_f16_k32,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F16, (16, 8, 32)),
    F8E4M3FN,
    F16,
    F16::ZERO,
    (|x: F8E4M3FN| x.to_f64(), |x: F16| x.to_f32() as f64)
);
mma_case!(
    mma_fp8_e4m3_f32_k16,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 16)),
    F8E4M3FN,
    f32,
    0.0f32,
    (|x: F8E4M3FN| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e4m3_f16_k16,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F16, (16, 8, 16)),
    F8E4M3FN,
    F16,
    F16::ZERO,
    (|x: F8E4M3FN| x.to_f64(), |x: F16| x.to_f32() as f64)
);
mma_case!(
    mma_fp8_e5m2_f32_k32,
    MmaFP8Op::new(DType::F8E5M2, DType::F32, (16, 8, 32)),
    F8E5M2,
    f32,
    0.0f32,
    (|x: F8E5M2| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e5m2_f16_k32,
    MmaFP8Op::new(DType::F8E5M2, DType::F16, (16, 8, 32)),
    F8E5M2,
    F16,
    F16::ZERO,
    (|x: F8E5M2| x.to_f64(), |x: F16| x.to_f32() as f64)
);
mma_case!(
    mma_fp8_e5m2_f32_k16,
    MmaFP8Op::new(DType::F8E5M2, DType::F32, (16, 8, 16)),
    F8E5M2,
    f32,
    0.0f32,
    (|x: F8E5M2| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e5m2_f16_k16,
    MmaFP8Op::new(DType::F8E5M2, DType::F16, (16, 8, 16)),
    F8E5M2,
    F16,
    F16::ZERO,
    (|x: F8E5M2| x.to_f64(), |x: F16| x.to_f32() as f64)
);

/// Every FP8 `mma.sync` (sm_89): both formats, both accumulators, both
/// shapes.
pub fn mma_fp8() -> Vec<Case> {
    vec![
        mma_fp8_e4m3_f32_k32(),
        mma_fp8_e4m3_f16_k32(),
        mma_fp8_e4m3_f32_k16(),
        mma_fp8_e4m3_f16_k16(),
        mma_fp8_e5m2_f32_k32(),
        mma_fp8_e5m2_f16_k32(),
        mma_fp8_e5m2_f32_k16(),
        mma_fp8_e5m2_f16_k16(),
    ]
}

/// Metadata words whose nibbles are all valid index pairs (`0b0100`,
/// `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110`); every rotation by a
/// whole number of nibbles keeps them valid.
const SPARSE_META: u32 = 0x489C_DE49;

/// One warp, `K_TILES` sparse MMAs: A is the compressed `(M, K/2)` tile,
/// the metadata of lane `l` and tile `t` is [`SPARSE_META`] rotated left by
/// `4 * ((l + t) % 7 + 1)` bits, which the kernel writes to `meta` for the
/// check.
fn sparse_mma_kernel<TA: Numeric, TC: Numeric>(name: &'static str, op: MmaF16BF16SparseOp, zero: TC) -> Kernel {
    let (m, n, k_atom) = op.shape_mnk();
    let k = k_atom * K_TILES;
    Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<TA>("a", Layout::col_major((m, k / 2)));
        let b = param_tensor::<TA>("b", Layout::col_major((n, k)));
        let c = param_tensor::<TC>("c", Layout::col_major((m, n)));
        let meta = param_tensor::<u32>("meta", Layout::col_major((32, K_TILES)));
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let lane = lane_idx();
        let thr = mma.get_slice(lane);
        let (ta, tb, tc) = (thr.partition_a(&a), thr.partition_b(&b), thr.partition_c(&c));
        let (fa, fb) = (thr.partition_fragment_a::<TA, _>(&a), thr.partition_fragment_b::<TA, _>(&b));
        let fe = thr.partition_fragment_e(&a);
        assert_eq!(fe.size(), K_TILES);
        let acc = thr.partition_fragment_c::<TC, _>(&c);
        for i in 0..fa.size() {
            fa.store(i, ta.load(i));
        }
        for i in 0..fb.size() {
            fb.store(i, tb.load(i));
        }
        let lane_u = lane.to::<u32>();
        for t in 0..K_TILES {
            let s = ((lane_u + t as u32) % 7u32 + 1u32) * 4u32;
            let word = Uint32::from(SPARSE_META);
            let e = (word << s) | (word >> (32u32 - s));
            fe.store(t, e);
            meta.store((lane, t), e);
        }
        acc.fill(zero);
        gemm_sparse(&mma, &acc, &fa, &fe, &fb, &acc);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    })
}

/// The dense `m x k` A (column-major) of a sparse MMA kernel's compressed A
/// and metadata words (`meta[lane + 32 * tile]`), from the PTX ISA's
/// metadata layout: lane `4 * (r % 8)` (plus `chunk / 4` for m16n8k32)
/// holds the nibble of chunk `chunk % 4` of row `r` at `4 * (r / 8) + chunk
/// % 4`; the nibble's low two bits place the chunk's first kept value.
pub fn sparse_dense_a(a: &[f64], meta: &[u32], m: usize, k_atom: usize, tiles: usize) -> Vec<f64> {
    let k = k_atom * tiles;
    let mut dense = vec![0.0; m * k];
    for tile in 0..tiles {
        for r in 0..m {
            for chunk in 0..k_atom / 4 {
                let lane = 4 * (r % 8) + if k_atom == 32 { chunk / 4 } else { 0 };
                let nibble = 4 * (r / 8) + chunk % 4;
                let bits = meta[lane + 32 * tile] >> (4 * nibble) & 0xf;
                let (i0, i1) = ((bits & 3) as usize, (bits >> 2) as usize);
                assert!(i0 < i1, "invalid metadata nibble {bits:#06b}");
                let (dense_col, comp_col) = (tile * k_atom + 4 * chunk, tile * k_atom / 2 + 2 * chunk);
                dense[r + m * (dense_col + i0)] = a[r + m * comp_col];
                dense[r + m * (dense_col + i1)] = a[r + m * (comp_col + 1)];
            }
        }
    }
    dense
}

macro_rules! sparse_mma_case {
    ($name:ident, $op:expr, $ta:ty, $tc:ty, $zero:expr, $as_f64:expr) => {
        pub fn $name() -> Case {
            let kernel = sparse_mma_kernel::<$ta, $tc>(stringify!($name), $op, $zero);
            fn check(args: &[KernelArg]) {
                let as_a: fn($ta) -> f64 = $as_f64.0;
                let as_c: fn($tc) -> f64 = $as_f64.1;
                let (m, n, k_atom) = $op.shape_mnk();
                let k = k_atom * K_TILES;
                let a: Vec<f64> = args[0].to_vec::<$ta>().into_iter().map(as_a).collect();
                let (b, c, meta) = (args[1].to_vec::<$ta>(), args[2].to_vec::<$tc>(), args[3].to_vec::<u32>());
                let dense = sparse_dense_a(&a, &meta, m, k_atom, K_TILES);
                for i in 0..m {
                    for j in 0..n {
                        let want: f64 = (0..k).map(|l| dense[i + m * l] * as_a(b[j + n * l])).sum();
                        assert_eq!(as_c(c[i + m * j]), want, "C[{i},{j}]");
                    }
                }
            }
            Case {
                name: stringify!($name),
                kernel,
                cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]),
                small_only: true,
                check,
            }
        }
    };
}

sparse_mma_case!(
    mma_sparse_f16_f32_k16,
    MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid),
    F16,
    f32,
    0.0f32,
    (|x: F16| x.to_f32() as f64, |x: f32| x as f64)
);
sparse_mma_case!(
    mma_sparse_f16_f16_k16,
    MmaF16BF16SparseOp::new(DType::F16, DType::F16, (16, 8, 16), SparseMetadataFormat::Tid),
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
sparse_mma_case!(
    mma_sparse_bf16_f32_k16,
    MmaF16BF16SparseOp::new(DType::BF16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);
sparse_mma_case!(
    mma_sparse_f16_f32_k32,
    MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 32), SparseMetadataFormat::Tid),
    F16,
    f32,
    0.0f32,
    (|x: F16| x.to_f32() as f64, |x: f32| x as f64)
);
sparse_mma_case!(
    mma_sparse_f16_f16_k32,
    MmaF16BF16SparseOp::new(DType::F16, DType::F16, (16, 8, 32), SparseMetadataFormat::Tid),
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
sparse_mma_case!(
    mma_sparse_bf16_f32_k32,
    MmaF16BF16SparseOp::new(DType::BF16, DType::F32, (16, 8, 32), SparseMetadataFormat::Tid),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);

/// Every sparse `mma.sp` (sm_80): F16 with F16 and F32 accumulators, BF16
/// with F32, both shapes.
pub fn mma_sparse() -> Vec<Case> {
    vec![
        mma_sparse_f16_f32_k16(),
        mma_sparse_f16_f16_k16(),
        mma_sparse_bf16_f32_k16(),
        mma_sparse_f16_f32_k32(),
        mma_sparse_f16_f16_k32(),
        mma_sparse_bf16_f32_k32(),
    ]
}

/// A `2 x 1`-warp FP8 m16n8k32 MMA fed from shared memory by
/// `LdMatrix8x16x8bOp` tiled copies built with `make_tiled_copy_a` (four
/// matrices) and `make_tiled_copy_b` (two), as CuTe's Ampere GEMM loads
/// K-major FP8 operands on SM89. The fragments the copies fill must be the
/// MMA's: the product must be exact.
pub fn ldmatrix_8x16x8b_fp8() -> Case {
    const M: usize = 32;
    const N: usize = 8;
    const K: usize = 64;
    let kernel = Kernel::builder("ldmatrix_8x16x8b_fp8").block_dim([64, 1, 1]).build(|| {
        let a = param_tensor::<F8E4M3FN>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F8E4M3FN>("b", Layout::row_major((N, K)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));
        let frag_a = param_tensor::<F8E4M3FN>("frag_a", Layout::row_major((64, 32)));
        let t = thread_idx().x;
        let (sa, sb) = (
            make_smem_tensor::<F8E4M3FN>(Layout::row_major((M, K))),
            make_smem_tensor::<F8E4M3FN>(Layout::row_major((N, K))),
        );
        let bytes16 = make_copy_atom::<F8E4M3FN>(CopyOp::Universal { bits: 128 });
        let g2s = make_tiled_copy_tv(&bytes16, &Layout::row_major((16, 4)), &Layout::row_major((1, 16)));
        let (gs, ss) = (g2s.get_slice(t).partition_s(&a), g2s.get_slice(t).partition_d(&sa));
        copy(&g2s, &gs, &ss);
        let bytes8 = make_copy_atom::<F8E4M3FN>(CopyOp::Universal { bits: 64 });
        let g2s_b = make_tiled_copy_tv(&bytes8, &Layout::row_major((8, 8)), &Layout::row_major((1, 8)));
        copy(&g2s_b, &g2s_b.get_slice(t).partition_s(&b), &g2s_b.get_slice(t).partition_d(&sb));
        sync_threads();
        let mma = make_tiled_mma(MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)), Layout::col_major((2, 1, 1)));
        let ldsm_a =
            make_tiled_copy_a(&make_copy_atom::<F8E4M3FN>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None }), &mma);
        let ldsm_b =
            make_tiled_copy_b(&make_copy_atom::<F8E4M3FN>(CopyOp::LdMatrix8x16x8b { num: 2, unpack_bits: None }), &mma);
        let thr_mma = mma.get_slice(t);
        let (fa, fb) =
            (thr_mma.partition_fragment_a::<F8E4M3FN, _>(&sa), thr_mma.partition_fragment_b::<F8E4M3FN, _>(&sb));
        copy(&ldsm_a, &ldsm_a.get_slice(t).partition_s(&sa), &ldsm_a.retile(&fa));
        copy(&ldsm_b, &ldsm_b.get_slice(t).partition_s(&sb), &ldsm_b.retile(&fb));
        for v in 0..fa.size() {
            frag_a.store((t, v), fa.load(v));
        }
        let acc = thr_mma.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0);
        gemm(&mma, &acc, &fa, &fb, &acc);
        let tc = thr_mma.partition_c(&c);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, b, c, frag_a) = (
            args[0].to_vec::<F8E4M3FN>(),
            args[1].to_vec::<F8E4M3FN>(),
            args[2].to_vec::<f32>(),
            args[3].to_vec::<F8E4M3FN>(),
        );
        // The A fragment of thread `t` (warp `w`, lane `l`) holds, for each
        // k-block `kb` of 32, a_i at row 16w + g + 8 ((i >> 2) & 1) and
        // column 32kb + 4t' + (i & 3) + 16 (i >> 3) (the PTX ISA's
        // m16n8k32 FP8 fragment, g = l / 4, t' = l % 4).
        for t in 0..64 {
            let (w, l) = (t / 32, t % 32);
            let (g, tq) = (l / 4, l % 4);
            for kb in 0..K / 32 {
                for i in 0..16 {
                    let row = 16 * w + g + 8 * ((i >> 2) & 1);
                    let col = 32 * kb + 4 * tq + (i & 3) + 16 * (i >> 3);
                    assert_eq!(frag_a[t * 32 + 16 * kb + i], a[row * K + col], "thread {t} k-block {kb} a{i}");
                }
            }
        }
        for i in 0..M {
            for j in 0..N {
                let want: f64 = (0..K).map(|l| a[i * K + l].to_f64() * b[j * K + l].to_f64()).sum();
                assert_eq!(c[i * N + j] as f64, want, "C[{i},{j}]");
            }
        }
    }
    Case {
        name: "ldmatrix_8x16x8b_fp8",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [64, 1, 1]),
        small_only: true,
        check,
    }
}

/// `make_tiled_copy_tv` with a warp-level atom: `ldmatrix.x4` over a
/// `32 x 8` tile, 8 x 4 threads each receiving a `4 x 2` block. The
/// destination is a `(2, 4)` register tensor (columns fastest), whose values
/// only CuTe's general `retile` gathers in the copy's order. The output must
/// be the input.
pub fn tiled_copy_tv_ldmatrix() -> Case {
    let kernel = Kernel::builder("tiled_copy_tv_ldmatrix").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((32, 8)));
        let out = param_tensor::<F16>("out", Layout::row_major((32, 8)));
        let t = thread_idx().x;
        let s = make_smem_tensor::<F16>(Layout::row_major((32, 8)));
        for j in 0..8 {
            s.store((t, j), a.load((t, j)));
        }
        sync_threads();
        let atom = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
        let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 4)), &Layout::row_major((4, 2)));
        assert_eq!((tiled.size(), tiled.tile_shape()), (32, IntTuple::from((32, 8))));
        let frag = make_fragment::<F16>((2, 4));
        copy(&tiled, &tiled.get_slice(t).partition_s(&s), &tiled.retile(&frag));
        // Thread `4 * tm + tk` owns rows `4 * tm + i` and columns `2 * tk + j`.
        let (tm, tk) = (t / 4, t % 4);
        for i in 0..4 {
            for j in 0..2 {
                out.store((tm * 4 + i, tk * 2 + j), frag.load((j, i)));
            }
        }
    });
    fn check(args: &[KernelArg]) {
        assert_eq!(args[1].to_vec::<F16>(), args[0].to_vec::<F16>());
    }
    Case {
        name: "tiled_copy_tv_ldmatrix",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]),
        small_only: false,
        check,
    }
}

/// The tiled-copy constructors around a `2 x 2`-warp m16n8k16 MMA: the A and
/// B fragments are loaded from global memory by `make_tiled_copy_d` copies
/// that partition like the `ldmatrix` copies feeding the MMA, C is stored
/// with `make_tiled_copy_c_atom` and `retile`, and `make_tiled_copy_s`
/// copies out the row each thread would hand `ldmatrix`.
pub fn mma_tiled_copies() -> Case {
    const M: usize = 32;
    const N: usize = 16;
    const K: usize = 16;
    let kernel = Kernel::builder("mma_tiled_copies").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F16>("b", Layout::row_major((N, K)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));
        let rows = param_tensor::<F16>("rows", Layout::row_major((128, 8)));
        let t = thread_idx().x;
        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)));
        let ldsm_a = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
        let ldsm_b = make_tiled_copy_b(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 2, trans: false }), &mma);
        let pairs = make_copy_atom::<F16>(CopyOp::Universal { bits: 32 });
        let g2r_a = make_tiled_copy_d(&pairs, &ldsm_a);
        let g2r_b = make_tiled_copy_d(&pairs, &ldsm_b);
        let thr_mma = mma.get_slice(t);
        let (fa, fb) = (thr_mma.partition_fragment_a::<F16, _>(&a), thr_mma.partition_fragment_b::<F16, _>(&b));
        copy(&g2r_a, &g2r_a.get_slice(t).partition_s(&a), &g2r_a.retile(&fa));
        copy(&g2r_b, &g2r_b.get_slice(t).partition_s(&b), &g2r_b.retile(&fb));
        let acc = thr_mma.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0);
        gemm(&mma, &acc, &fa, &fb, &acc);
        let r2g = make_tiled_copy_c_atom(&make_copy_atom::<f32>(CopyOp::Universal { bits: 64 }), &mma);
        copy(&r2g, &r2g.retile(&acc), &r2g.get_slice(t).partition_d(&c));
        let s_copy = make_tiled_copy_s(&make_copy_atom::<F16>(CopyOp::Universal { bits: 128 }), &ldsm_a);
        let src = s_copy.get_slice(t).partition_s(&a);
        let row = make_fragment_like::<F16, F16>(&src);
        copy(&s_copy, &src, &row);
        for v in 0..8 {
            rows.store((t, v), row.load(v));
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, b, c, rows) =
            (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>(), args[3].to_vec::<F16>());
        for i in 0..M {
            for j in 0..N {
                let want: f32 = (0..K).map(|l| a[i * K + l].to_f32() * b[j * K + l].to_f32()).sum();
                assert_eq!(c[i * N + j], want, "C[{i},{j}]");
            }
        }
        // ldmatrix.x4 for an m16n8k16 A fragment loads matrices A[0:8, 0:8],
        // A[8:16, 0:8], A[0:8, 8:16] and A[8:16, 8:16] (registers a0a1, a2a3,
        // a4a5, a6a7 of the PTX fragment table); lane `l` supplies row `l % 8`
        // of matrix `l / 8`. Warp `w` covers rows `16 * (w % 2)` on.
        for t in 0..128 {
            let (w, l) = (t / 32, t % 32);
            let (row, col) = (16 * (w % 2) + 8 * ((l / 8) % 2) + l % 8, 8 * (l / 16));
            for v in 0..8 {
                assert_eq!(rows[t * 8 + v], a[row * K + col + v], "thread {t} value {v}");
            }
        }
    }
    Case { name: "mma_tiled_copies", kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: true, check }
}

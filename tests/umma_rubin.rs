// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernels' structure (scale factors staged from shared memory into TMEM with a
// Cp4x32x128bOp copy, then a k-block loop of block-scaled MMAs taking `[a, sfa]` and
// `[b, sfb]`, a commit to an mbarrier and a tcgen05.ld epilogue) follows NVIDIA
// CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py and
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py (the pair),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The Rubin (sm_107) tcgen05 MMAs: GEMM tiles on the
//! simulator for SM107MmaFP8Op (K = 64 and 32), SM107BlockScaledMmaMXF8F6F4Op (K = 64, one
//! CTA and a pair) and SM107MmaMXF4NVF4Op (K = 128 with 16-element UE4M3 and UE5M3 scales,
//! the UE5M3 codes made by the host codec, and 32-element UE8M0 scales), exact against plain
//! Rust; a B collector-reuse sequence (FILL, USE, LASTUSE) equal to the same MMAs without
//! reuse; the simulator's reports of a USE with nothing filled and of a B changed between
//! FILL and USE; the build-time checks; the targets (sm_107f and sm_107a only); and the MLIR
//! text, verified by mlir-opt when one is found. No installed LLVM or ptxas has sm_107, so
//! nothing here is assembled.

use CollectorOp::{DISCARD, FILL, LASTUSE, USE};
use OperandMajorMode::{K, MN};
use SmemLayoutAtomKind::{K_SW64, K_SW128, MN_SW128};
use cute::block::block_copy;
use cute::prelude::*;
use cute::tcgen05::{
    self, CollectorOp, Cp4x32x128bOp, CtaGroup, Field, Ld32x32bOp, OperandSource, Repetition,
    SM107BlockScaledMmaMXF8F6F4Op, SM107MmaFP8Op, SM107MmaMXF4NVF4Op, SmemLayoutAtomKind,
};

const LANE: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Case {
    m: usize,
    n: usize,
    k: usize,
    pair: bool,
    a_kind: SmemLayoutAtomKind,
    b_kind: SmemLayoutAtomKind,
    a_fmt: Fmt,
    b_fmt: Fmt,
    /// FP4 packed two per byte (mxf4nvf4) rather than the padded image.
    packed: bool,
    sf: Sf,
    /// Elements per scale factor; 0 for a dense MMA.
    vec: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    E4M3,
    E5M2,
    E2M1,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Sf {
    Ue8m0,
    Ue4m3,
    Ue5m3,
}

impl Fmt {
    fn code(self, x: f32) -> u8 {
        match self {
            Fmt::E4M3 => F8E4M3FN::from_f32(x).0,
            Fmt::E5M2 => F8E5M2::from_f32(x).0,
            Fmt::E2M1 => F4E2M1FN::from_f32(x).0,
        }
    }
}

fn row_bytes(fmt: Fmt, packed: bool, k: usize) -> usize {
    if fmt == Fmt::E2M1 && packed { k / 2 } else { k }
}

/// The global bytes of the row-major values `v`: FP8 codes; FP4 packed (low nibble first) or
/// the padded image (16 values in the first 8 bytes of each 16-byte slot).
fn image(fmt: Fmt, packed: bool, v: &[f32]) -> Vec<u8> {
    let codes: Vec<u8> = v.iter().map(|&x| fmt.code(x)).collect();
    if fmt != Fmt::E2M1 {
        return codes;
    }
    if packed {
        return codes.chunks(2).map(|p| p[0] | p[1] << 4).collect();
    }
    codes
        .chunks(16)
        .flat_map(|g| {
            let mut slot = [0u8; 16];
            for (i, &x) in g.iter().enumerate() {
                slot[i / 2] |= x << (4 * (i % 2));
            }
            slot
        })
        .collect()
}

fn operand_layout<T: Numeric>(kind: SmemLayoutAtomKind, mn: usize, k: usize) -> ComposedLayout {
    let atom = tcgen05::make_smem_layout_atom::<T>(kind);
    if kind.major_mode() == OperandMajorMode::K {
        tile_to_shape_swizzled(&atom, (mn, k))
    } else {
        tile_to_shape_swizzled_ordered(&atom, (mn, k), &[1, 0])
    }
}

/// Rows `row0..row0 + mn` of the row-major global bytes `g` into the `(mn, kb)` byte view
/// `s`, by all 128 threads (MN-major: `g` holds the transpose).
fn fill(g: &Tensor<u8>, s: &Tensor<u8>, row0: Int32, mn: usize, kb: usize, mn_major: bool) {
    let t = thread_idx().x;
    for_range(0, (mn * kb / 128) as i32, 1, |j| {
        let idx = j * 128 + t;
        let (r, c) = (idx % mn as i32, idx / mn as i32);
        if mn_major {
            s.store((r, c), g.load((c, row0 + r)));
        } else {
            s.store((r, c), g.load((row0 + r, c)));
        }
    });
}

/// Shared-memory scale factors: `kt` chunks of 128 rows by 4 factors.
fn sf_smem_layout(vec: usize, kt: usize) -> Layout {
    let chunk = utils::BlockScaledBasicChunk::new(vec, OperandMajorMode::K);
    Layout::from_modes([chunk.layout().clone(), Layout::new(kt, 512)])
}

fn sf_smem_offset(r: usize, s: usize) -> usize {
    16 * (r % 32) + 4 * (r / 32) + s % 4 + 512 * (s / 4)
}

/// TMEM scale factors as the copy writes them: row `32 q + l` in lane `l` of each replica,
/// column `q`, factor `s` in byte `s % 4` of block `s / 4` (four columns per block). A
/// K = 128, 16-element MMA reads two consecutive blocks.
fn sf_tmem_layout(vec: usize, kt: usize) -> Layout {
    let dp = 4 * LANE;
    let mn = Layout::new(((32, 4), 4), ((dp, 4), 32 * dp));
    let k = Layout::new((vec, 4), (0, 1));
    Layout::from_modes([Layout::from_modes([mn, k]), Layout::new(kt, 16)])
}

/// Copy a TMEM accumulator tile `(rows, n)` of this CTA (Layout D: row by lane) to `gc`.
fn epilogue(tmem: &Ptr<f32>, gc: &Tensor<f32>, row0: Int32, col0: usize, n: usize) {
    let warp = make_warp_uniform(warp_idx());
    let row = row0 + (warp % 4) * 32 + lane_idx();
    let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
    for chunk in 0..n / 32 {
        let layout = Layout::from_modes([Layout::new((32, 32), (1, LANE))]);
        let view = make_tensor(tmem.clone() + (warp % 4) * (32 * LANE) as i32 + (32 * chunk) as i32, layout);
        let frag = make_fragment::<f32>(32);
        copy(&ld, &view, &frag);
        fence_view_async_tmem_load();
        for e in 0..32 {
            gc.store((row, (col0 + 32 * chunk + e) as i32), frag.load(e));
        }
    }
    fence_view_async_tmem_load();
}

/// The tile `C = (A sfa) (B sfb)^T` (or `A B^T` when dense), C `(m, n)` row-major Float32,
/// by one CTA or a CTA pair of four warps each, for `target`.
fn tile_kernel<TA: Numeric, TB: Numeric, S: Numeric>(
    name: &str,
    op: impl Into<MmaAtom>,
    c: Case,
    target: Arch,
) -> Kernel {
    let atom: MmaAtom = op.into();
    let kb = atom.shape_mnk()[2];
    let Case { m, n, k, pair, a_kind, b_kind, a_fmt, b_fmt, packed, vec, .. } = c;
    let scaled = vec > 0;
    let v = if pair { 2 } else { 1 };
    let (ma, nb) = (m / v, n / v);
    let (kab, kbb) = (row_bytes(a_fmt, packed, k), row_bytes(b_fmt, packed, k));
    let nsf = if scaled { kb / vec } else { 0 };
    let kt = if scaled { k / vec / 4 } else { 0 };
    let group = if pair { CtaGroup::TWO } else { CtaGroup::ONE };
    Kernel::builder(name).block_dim([128, 1, 1]).target(target).build(move || {
        let a_mn = a_kind.major_mode() == OperandMajorMode::MN;
        let b_mn = b_kind.major_mode() == OperandMajorMode::MN;
        let ga =
            param_tensor::<u8>("a", if a_mn { Layout::new((kab, m), (m, 1)) } else { Layout::new((m, kab), (kab, 1)) });
        let gb =
            param_tensor::<u8>("b", if b_mn { Layout::new((kbb, n), (n, 1)) } else { Layout::new((n, kbb), (kbb, 1)) });
        let gsfa = param_tensor::<u8>("sfa", Layout::col_major((v * kt * 512).max(1)));
        let gsfb = param_tensor::<u8>("sfb", Layout::col_major((kt * 512).max(1)));
        let gc = param_tensor::<f32>("c", Layout::new((m, n), (n, 1)));
        let mut mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let sa = make_smem_tensor_aligned::<TA>(operand_layout::<TA>(a_kind, ma, k), 1024);
        let sb = make_smem_tensor_aligned::<TB>(operand_layout::<TB>(b_kind, nb, k), 1024);
        let ssfa = make_smem_tensor_aligned::<u8>(sf_smem_layout(16, kt.max(1)), 128);
        let ssfb = make_smem_tensor_aligned::<u8>(sf_smem_layout(16, kt.max(1)), 128);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        let rank = if pair { block_idx_in_cluster() } else { Int32::from(0) };
        let leader = rank.eq(0);
        let (sfa_col, sfb_col) = (n, n + 4 * kt);
        let cols = (n + 8 * kt).next_power_of_two().max(32) as i32;
        if_(warp.eq(0), || alloc_tmem(cols, &slot, pair.then_some(true), Arch::Sm107a));
        let (sa8, sb8) = (sa.recast::<u8>(), sb.recast::<u8>());
        fill(&ga, &sa8, rank * ma as i32, ma, kab, a_mn);
        fill(&gb, &sb8, rank * nb as i32, nb, kbb, b_mn);
        if scaled {
            let raw_a = make_tensor(ssfa.ptr(), Layout::col_major(kt * 512));
            let raw_b = make_tensor(ssfb.ptr(), Layout::col_major(kt * 512));
            for_range(0, (kt * 4) as i32, 1, |j| {
                raw_a.store(j * 128 + t, gsfa.load(rank * (kt * 512) as i32 + j * 128 + t));
                raw_b.store(j * 128 + t, gsfb.load(j * 128 + t));
            });
        }
        fence_view_async_shared();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        if pair {
            cluster_arrive();
            cluster_wait();
        } else {
            sync_threads();
        }
        let tmem = retrieve_tmem_ptr::<f32>(16, &slot);
        let tmem8 = retrieve_tmem_ptr::<u8>(16, &slot);
        let tmem_sf = retrieve_tmem_ptr::<S>(16, &slot);
        let acc =
            make_tensor(tmem.clone(), mma.make_fragment_c::<f32>(&mma.partition_shape_c((m, n))).layout().clone());
        let fa = mma.make_fragment_a(&tiled_divide(&sa, (ma, kb)));
        let fb = mma.make_fragment_b(&tiled_divide(&sb, (nb, kb)));
        if_(warp.eq(0) & leader, || {
            if scaled {
                let catom = make_copy_atom::<u8>(Cp4x32x128bOp::new(group));
                for (s, col) in [(&ssfa, sfa_col), (&ssfb, sfb_col)] {
                    for blk in 0..kt {
                        let tsf = make_tensor(tmem8.clone() + (4 * col + 16 * blk) as i32, sf_tmem_layout(vec, 1));
                        let src = make_tensor(s.ptr() + (512 * blk) as i32, sf_smem_layout(vec, 1));
                        let tiled = tcgen05::make_s2t_copy(&catom, &tsf.with_layout(tsf.layout().filter_zeros()));
                        block_copy(&tiled, &src, &tsf, CopyArgs::new());
                    }
                }
            }
            mma.set(Field::ACCUMULATE, false);
            for j in 0..k / kb {
                let (aj, bj) = (fa.slice((.., .., j)), fb.slice((.., .., j)));
                if scaled {
                    // The factors of k-block j: a row's stream from factor j * nsf.
                    let s0 = j * nsf;
                    let off = (16 * (s0 / 4) + s0 % 4) as i32;
                    let sfa = make_tensor(tmem_sf.clone() + (4 * sfa_col) as i32 + off, Layout::col_major(1));
                    let sfb = make_tensor(tmem_sf.clone() + (4 * sfb_col) as i32 + off, Layout::col_major(1));
                    gemm(&mma, &acc, (&aj, &sfa), (&bj, &sfb), &acc);
                } else {
                    gemm(&mma, &acc, &aj, &bj, &acc);
                }
                mma.set(Field::ACCUMULATE, true);
            }
            let mask = pair.then(|| Int16::from(0b11i16));
            if_(elect_one(), || tcgen05::commit(&mbar, mask, group));
        });
        mbarrier_wait(&mbar, 0);
        epilogue(&tmem, &gc, rank * 128, 0, n);
        if pair {
            cluster_arrive();
            cluster_wait();
        } else {
            sync_threads();
        }
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(pair.then_some(true));
            dealloc_tmem(&tmem, cols, pair.then_some(true), Arch::Sm107a);
        });
    })
}

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// Values from -2 to 2 in steps of 0.5 (exact in E2M1, E4M3 and E5M2).
fn small_halves(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| (r() % 9) as f32 * 0.5 - 2.0).collect()
}

/// Scale factors of 0.5, 1, 2 or 4 and their codes (UE5M3 through the host codec).
fn scales(n: usize, seed: u64, sf: Sf) -> Vec<(f32, u8)> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            let e = (r() % 4) as i32 - 1;
            let x = 2f32.powi(e);
            let code = match sf {
                Sf::Ue8m0 => (127 + e) as u8,
                Sf::Ue4m3 => (0x38 + 8 * e) as u8,
                Sf::Ue5m3 => F8E5M3FNU::from_f32(x).0,
            };
            (x, code)
        })
        .collect()
}

fn run<TA: Numeric, TB: Numeric, S: Numeric>(name: &str, op: impl Into<MmaAtom>, c: Case) {
    let atom: MmaAtom = op.into();
    let kb = atom.shape_mnk()[2];
    let Case { m, n, k, pair, a_fmt, b_fmt, packed, sf, vec, a_kind, b_kind, .. } = c;
    let scaled = vec > 0;
    let s_per_row = if scaled { k / vec } else { 0 };
    let kt = s_per_row / 4;
    let (av, bv) = (small_halves(m * k, 3 + m as u64 + k as u64), small_halves(n * k, 11 + n as u64));
    let (sa, sb) = (scales(m * s_per_row, 5 + n as u64, sf), scales(n * s_per_row, 7 + m as u64, sf));
    let transpose = |img: Vec<u8>, rows: usize, cols: usize| -> Vec<u8> {
        (0..rows * cols).map(|i| img[(i % rows) * cols + i / rows]).collect()
    };
    let a_img = image(a_fmt, packed, &av);
    let b_img = image(b_fmt, packed, &bv);
    let a_img = if a_kind.major_mode() == OperandMajorMode::MN { transpose(a_img, m, k) } else { a_img };
    let b_img = if b_kind.major_mode() == OperandMajorMode::MN { transpose(b_img, n, k) } else { b_img };
    let v = if pair { 2 } else { 1 };
    let ma = m / v;
    let mut sfa = vec![0u8; (v * kt * 512).max(1)];
    for r in 0..m {
        for s in 0..s_per_row {
            sfa[(r / ma) * kt * 512 + sf_smem_offset(r % ma, s)] = sa[r * s_per_row + s].1;
        }
    }
    let mut sfb = vec![0u8; (kt * 512).max(1)];
    for r in 0..n {
        for s in 0..s_per_row {
            sfb[sf_smem_offset(r, s)] = sb[r * s_per_row + s].1;
        }
    }
    let kernel = tile_kernel::<TA, TB, S>(name, atom, c, Arch::Sm107a);
    let mut args = vec![
        KernelArg::buffer(&a_img),
        KernelArg::buffer(&b_img),
        KernelArg::buffer(&sfa),
        KernelArg::buffer(&sfb),
        KernelArg::buffer(&vec![0f32; m * n]),
    ];
    let cfg = if pair {
        LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1])
    } else {
        LaunchConfig::new([1, 1, 1], [128, 1, 1])
    };
    let report = kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.tcgen05_mma_instructions, (k / kb) as u64, "{name}");
    let got = args[4].to_vec::<f32>();
    let scale = |s: &[(f32, u8)], row: usize, kk: usize| if scaled { s[row * s_per_row + kk / vec].0 } else { 1.0 };
    for r in 0..m {
        for col in 0..n {
            let want: f32 =
                (0..k).map(|kk| av[r * k + kk] * scale(&sa, r, kk) * bv[col * k + kk] * scale(&sb, col, kk)).sum();
            assert_eq!(got[r * n + col], want, "{name}: C({r}, {col})");
        }
    }
}

fn case(m: usize, n: usize, k: usize, fmts: (Fmt, Fmt), packed: bool, vec: usize, sf: Sf) -> Case {
    Case { m, n, k, pair: false, a_kind: K_SW128, b_kind: K_SW128, a_fmt: fmts.0, b_fmt: fmts.1, packed, sf, vec }
}

#[test]
fn fp8_k64_and_k32_tiles_match_the_rust_reference() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let f8 = |a, b, shape: (usize, usize, usize), bm| {
        SM107MmaFP8Op::new(a, b, DType::F32, shape, one, s, K, bm, DISCARD, DISCARD)
    };
    // K = 64 (the default), E4M3 x E5M2, two MMAs.
    let op = f8(DType::F8E4M3FN, DType::F8E5M2, (128, 64, 64), K);
    assert_eq!(op.shape_mnk(), (128, 64, 64));
    let c = case(128, 64, 128, (Fmt::E4M3, Fmt::E5M2), false, 0, Sf::Ue8m0);
    run::<F8E4M3FN, F8E5M2, u8>("rubin_fp8_k64", op, c);
    // K = 64 with an MN-major B (N step 16).
    let op = f8(DType::F8E5M2, DType::F8E4M3FN, (128, 128, 64), MN);
    let c = Case { b_kind: MN_SW128, ..case(128, 128, 128, (Fmt::E5M2, Fmt::E4M3), false, 0, Sf::Ue8m0) };
    run::<F8E5M2, F8E4M3FN, u8>("rubin_fp8_k64_mn", op, c);
    // K = 32, four MMAs.
    let op = f8(DType::F8E4M3FN, DType::F8E4M3FN, (128, 64, 32), K);
    let c = case(128, 64, 128, (Fmt::E4M3, Fmt::E4M3), false, 0, Sf::Ue8m0);
    run::<F8E4M3FN, F8E4M3FN, u8>("rubin_fp8_k32", op, c);
}

#[test]
fn mxf8f6f4_k64_tiles_match_the_rust_reference() {
    let s = OperandSource::SMEM;
    let mx = |a, b, shape, group| {
        SM107BlockScaledMmaMXF8F6F4Op::new(a, b, DType::F8E8M0FNU, 32, shape, group, s, K, K, DISCARD, DISCARD)
    };
    // E2M1 (padded) x E4M3, two factors per MMA: the IDs walk bytes 0 and 2.
    let op = mx(DType::F4E2M1FN, DType::F8E4M3FN, (128, 64), CtaGroup::ONE);
    let c = case(128, 64, 128, (Fmt::E2M1, Fmt::E4M3), false, 32, Sf::Ue8m0);
    run::<u8, F8E4M3FN, F8E8M0FNU>("rubin_mxf8f6f4", op, c);
    // A pair: M = 256.
    let op = mx(DType::F8E4M3FN, DType::F8E5M2, (256, 128), CtaGroup::TWO);
    let c = Case { pair: true, ..case(256, 128, 128, (Fmt::E4M3, Fmt::E5M2), false, 32, Sf::Ue8m0) };
    run::<F8E4M3FN, F8E5M2, F8E8M0FNU>("rubin_mxf8f6f4_pair", op, c);
}

#[test]
fn mxf4nvf4_k128_tiles_match_the_rust_reference() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let fp4 = (Fmt::E2M1, Fmt::E2M1);
    let nv = |sf, vec, n| SM107MmaMXF4NVF4Op::new(sf, vec, (128, n), one, s, DISCARD, DISCARD);
    // 16-element vectors: eight factors per MMA, two blocks of four columns.
    let c = case(128, 64, 256, fp4, true, 16, Sf::Ue4m3);
    run::<F4E2M1FN, F4E2M1FN, F8E4M3FN>("rubin_nvf4_ue4m3", nv(DType::F8E4M3FN, 16, 64), c);
    let c = case(128, 128, 256, fp4, true, 16, Sf::Ue5m3);
    run::<F4E2M1FN, F4E2M1FN, F8E5M3FNU>("rubin_nvf4_ue5m3", nv(DType::F8E5M3FNU, 16, 128), c);
    // 32-element vectors (.block32): four factors per MMA.
    let c = case(128, 64, 256, fp4, true, 32, Sf::Ue8m0);
    run::<F4E2M1FN, F4E2M1FN, F8E8M0FNU>("rubin_mxf4_block32", nv(DType::F8E8M0FNU, 32, 64), c);
}

/// What the reuse kernel does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reuse {
    /// FILL, USE, LASTUSE of B.
    Collect,
    /// The same MMAs without the collector.
    Discard,
    /// USE first.
    UseFirst,
    /// FILL, then (after the MMA completed) a store into B's tile, then USE.
    ChangeB,
}

/// Three MMAs `D_i = A_i B^T` (FP8 K = 64, 128 x 64 each) sharing one B tile.
fn reuse_kernel(name: &str, mode: Reuse) -> Kernel {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let base =
        SM107MmaFP8Op::new(DType::F8E4M3FN, DType::F8E4M3FN, DType::F32, (128, 64), one, s, K, K, DISCARD, DISCARD);
    let ops = match mode {
        Reuse::Collect | Reuse::ChangeB => [FILL, USE, LASTUSE],
        Reuse::Discard => [DISCARD; 3],
        Reuse::UseFirst => [USE, USE, LASTUSE],
    };
    Kernel::builder(name).block_dim([128, 1, 1]).target(Arch::Sm107a).build(move || {
        let ga = param_tensor::<u8>("a", Layout::new((384, 64), (64, 1)));
        let gb = param_tensor::<u8>("b", Layout::new((64, 64), (64, 1)));
        let gc = param_tensor::<f32>("c", Layout::new((128, 192), (192, 1)));
        let mmas: Vec<TiledMma> = ops
            .iter()
            .map(|&o| make_tiled_mma(base.with_collectors(DISCARD, o), Layout::col_major((1, 1, 1))))
            .collect();
        let sas: Vec<_> = (0..3)
            .map(|_| make_smem_tensor_aligned::<F8E4M3FN>(operand_layout::<F8E4M3FN>(K_SW64, 128, 64), 1024))
            .collect();
        let sb = make_smem_tensor_aligned::<F8E4M3FN>(operand_layout::<F8E4M3FN>(K_SW64, 64, 64), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(2, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(0), || alloc_tmem(256, &slot, None, Arch::Sm107a));
        for (i, sa) in sas.iter().enumerate() {
            fill(&ga, &sa.recast::<u8>(), Int32::from(128 * i as i32), 128, 64, false);
        }
        let sb8 = sb.recast::<u8>();
        fill(&gb, &sb8, Int32::from(0), 64, 64, false);
        fence_view_async_shared();
        if_(t.eq(0), || {
            mbarrier_init(&mbar, 1);
            mbarrier_init(&(mbar.clone() + 1), 1);
        });
        mbarrier_init_fence();
        sync_threads();
        let tmem = retrieve_tmem_ptr::<f32>(16, &slot);
        let fb = mmas[0].make_fragment_b(&tiled_divide(&sb, (64, 64)));
        if_(warp.eq(0), || {
            for (i, mma) in mmas.iter().enumerate() {
                let mut mma = mma.clone();
                let fa = mma.make_fragment_a(&tiled_divide(&sas[i], (128, 64)));
                let layout = mma.make_fragment_c::<f32>(&mma.partition_shape_c((128, 64))).layout().clone();
                let acc = make_tensor(tmem.clone() + (64 * i) as i32, layout);
                mma.set(Field::ACCUMULATE, false);
                gemm(&mma, &acc, &fa, &fb, &acc);
                if mode == Reuse::ChangeB && i == 0 {
                    // Let the FILL complete, then change one byte of B's tile.
                    if_(elect_one(), || tcgen05::commit(&(mbar.clone() + 1), None, one));
                    mbarrier_wait(&(mbar.clone() + 1), 0);
                    if_(lane_idx().eq(0), || sb8.store((0, 0), 0x7fu8));
                    fence_view_async_shared();
                }
            }
            if_(elect_one(), || tcgen05::commit(&mbar, None, one));
        });
        mbarrier_wait(&mbar, 0);
        epilogue(&tmem, &gc, Int32::from(0), 0, 192);
        sync_threads();
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem, 256, None, Arch::Sm107a);
        });
    })
}

fn simulate_reuse(mode: Reuse, a: &[u8], b: &[u8]) -> Result<Vec<f32>, String> {
    let kernel = reuse_kernel("rubin_reuse", mode);
    let mut args = vec![KernelArg::buffer(a), KernelArg::buffer(b), KernelArg::buffer(&vec![0f32; 128 * 192])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).map_err(|e| e.to_string())?;
    Ok(args[2].to_vec::<f32>())
}

#[test]
fn b_reuse_through_the_collector_equals_the_mmas_without_it() {
    let (av, bv) = (small_halves(384 * 64, 21), small_halves(64 * 64, 22));
    let (a, b) = (image(Fmt::E4M3, false, &av), image(Fmt::E4M3, false, &bv));
    let reused = simulate_reuse(Reuse::Collect, &a, &b).unwrap();
    let plain = simulate_reuse(Reuse::Discard, &a, &b).unwrap();
    assert_eq!(reused, plain);
    for i in 0..3 {
        for r in (0..128).step_by(5) {
            for col in 0..64 {
                let want: f32 = (0..64).map(|kk| av[(128 * i + r) * 64 + kk] * bv[col * 64 + kk]).sum();
                assert_eq!(reused[r * 192 + 64 * i + col], want, "D{i}({r}, {col})");
            }
        }
    }
}

#[test]
fn collector_misuse_is_reported() {
    let (a, b) = (vec![0x38u8; 384 * 64], vec![0x38u8; 64 * 64]);
    let e = simulate_reuse(Reuse::UseFirst, &a, &b).unwrap_err();
    assert!(e.contains(".collector::b::use") && e.contains("FILL"), "{e}");
    let e = simulate_reuse(Reuse::ChangeB, &a, &b).unwrap_err();
    assert!(e.contains(".collector::b::use") && e.contains("changed since"), "{e}");
}

fn panic_text(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn illegal_rubin_operations_panic_when_built() {
    let (one, two, s, tm) = (CtaGroup::ONE, CtaGroup::TWO, OperandSource::SMEM, OperandSource::TMEM);
    let (e4, f32_) = (DType::F8E4M3FN, DType::F32);
    // K the op does not have.
    let e = panic_text(|| {
        SM107MmaFP8Op::new(e4, e4, f32_, (128, 64, 128), one, s, K, K, DISCARD, DISCARD);
    });
    assert!(e.contains("K is 128"), "{e}");
    let e = panic_text(|| {
        SM107MmaMXF4NVF4Op::new(DType::F8E4M3FN, 16, (128, 64, 64), one, s, DISCARD, DISCARD);
    });
    assert!(e.contains("K = 128"), "{e}");
    let e = panic_text(|| {
        SM107BlockScaledMmaMXF8F6F4Op::new(e4, e4, DType::F8E8M0FNU, 32, (128, 64, 32), one, s, K, K, DISCARD, DISCARD);
    });
    assert!(e.contains("K = 64"), "{e}");
    // Types and vectors.
    let e = panic_text(|| {
        SM107MmaFP8Op::new(DType::F8E5M3FNU, e4, f32_, (128, 64), one, s, K, K, DISCARD, DISCARD);
    });
    assert!(e.contains("a_dtype"), "{e}");
    let e = panic_text(|| {
        SM107MmaMXF4NVF4Op::new(DType::F8E5M2, 16, (128, 64), one, s, DISCARD, DISCARD);
    });
    assert!(e.contains("sf_dtype"), "{e}");
    let e = panic_text(|| {
        SM107MmaMXF4NVF4Op::new(DType::F8E8M0FNU, 64, (128, 64), one, s, DISCARD, DISCARD);
    });
    assert!(e.contains("sf_vec_size"), "{e}");
    let e = panic_text(|| {
        SM107BlockScaledMmaMXF8F6F4Op::new(e4, e4, DType::F8E4M3FN, 32, (128, 64), one, s, K, K, DISCARD, DISCARD);
    });
    assert!(e.contains("sf_dtype"), "{e}");
    // M: 128 on one CTA, 256 on a pair, for the block-scaled forms.
    let e = panic_text(|| {
        SM107MmaMXF4NVF4Op::new(DType::F8E8M0FNU, 32, (128, 64), two, s, DISCARD, DISCARD);
    });
    assert!(e.contains("M is 128") && e.contains("M = 256"), "{e}");
    // A's collector with A in TMEM.
    let e = panic_text(|| {
        SM107MmaFP8Op::new(e4, e4, f32_, (128, 64), one, tm, K, K, FILL, DISCARD);
    });
    assert!(e.contains("a_collector_op") && e.contains("TMEM"), "{e}");
    // NV8E5M3 operands are accepted by the op but not issued.
    let op = SM107BlockScaledMmaMXF8F6F4Op::new(
        DType::F8E5M3FNU,
        e4,
        DType::F8E8M0FNU,
        32,
        (128, 64),
        one,
        s,
        K,
        K,
        DISCARD,
        DISCARD,
    );
    assert_eq!((op.a_dtype(), op.shape_mnk().2), (DType::F8E5M3FNU, 64));
}

#[test]
fn rubin_mmas_target_the_sm107f_family_only() {
    let check = |k: &Kernel, a: Arch| cute::validate::check_target(k.ir(), a);
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let fp8 = tile_kernel::<F8E4M3FN, F8E4M3FN, u8>(
        "t_rubin_fp8",
        SM107MmaFP8Op::new(DType::F8E4M3FN, DType::F8E4M3FN, DType::F32, (128, 64), one, s, K, K, DISCARD, DISCARD),
        case(128, 64, 128, (Fmt::E4M3, Fmt::E4M3), false, 0, Sf::Ue8m0),
        Arch::Sm107a,
    );
    let nvf4 = tile_kernel::<F4E2M1FN, F4E2M1FN, F8E5M3FNU>(
        "t_rubin_nvf4",
        SM107MmaMXF4NVF4Op::new(DType::F8E5M3FNU, 16, (128, 64), one, s, DISCARD, DISCARD),
        case(128, 64, 256, (Fmt::E2M1, Fmt::E2M1), true, 16, Sf::Ue5m3),
        Arch::Sm107a,
    );
    for k in [&fp8, &nvf4] {
        for a in [Arch::Sm107a, Arch::Sm107f] {
            check(k, a).unwrap_or_else(|e| panic!("{} {a:?}: {}", k.ir().name, e.message));
        }
        // An sm_107 op compiled for sm_100a.
        let e = check(k, Arch::Sm100a).unwrap_err().message;
        assert!(e.contains("sm_107f"), "{e}");
        // The baseline sm_107 has no tcgen05 at all.
        assert!(check(k, Arch::Sm107).is_err());
    }
    // K = 64 on an arch that has tcgen05 but not the Rubin K.
    let e = check(&fp8, Arch::Sm103a).unwrap_err().message;
    assert!(e.contains("K = 64") && e.contains("sm_107f"), "{e}");
    let e = check(&nvf4, Arch::Sm110a).unwrap_err().message;
    assert!(e.contains("K = 128"), "{e}");
}

#[test]
fn rubin_mlir() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let fp8 = SM107MmaFP8Op::new(DType::F8E4M3FN, DType::F8E5M2, DType::F32, (128, 64), one, s, K, K, DISCARD, DISCARD);
    let idesc = fp8.kind().idesc();
    let kernels = [
        (
            tile_kernel::<F8E4M3FN, F8E5M2, u8>(
                "mlir_rubin_fp8",
                fp8,
                case(128, 64, 128, (Fmt::E4M3, Fmt::E5M2), false, 0, Sf::Ue8m0),
                Arch::Sm107a,
            ),
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [",
        ),
        (
            tile_kernel::<u8, F8E4M3FN, F8E8M0FNU>(
                "mlir_rubin_mxf8f6f4",
                SM107BlockScaledMmaMXF8F6F4Op::new(
                    DType::F4E2M1FN,
                    DType::F8E4M3FN,
                    DType::F8E8M0FNU,
                    32,
                    (128, 64),
                    one,
                    s,
                    K,
                    K,
                    DISCARD,
                    DISCARD,
                ),
                case(128, 64, 128, (Fmt::E2M1, Fmt::E4M3), false, 32, Sf::Ue8m0),
                Arch::Sm107a,
            ),
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [",
        ),
        (
            tile_kernel::<F4E2M1FN, F4E2M1FN, F8E4M3FN>(
                "mlir_rubin_nvf4",
                SM107MmaMXF4NVF4Op::new(DType::F8E4M3FN, 16, (128, 64), one, s, DISCARD, DISCARD),
                case(128, 64, 256, (Fmt::E2M1, Fmt::E2M1), true, 16, Sf::Ue4m3),
                Arch::Sm107a,
            ),
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [",
        ),
        (reuse_kernel("mlir_rubin_reuse", Reuse::Collect), "kind::f8f6f4.collector::b::lastuse ["),
    ];
    let tc = Toolchain::discover();
    if tc.is_none() {
        eprintln!("skipping the mlir-opt check: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    for (kernel, text) in &kernels {
        let mlir = kernel.to_mlir();
        assert!(mlir.contains(text), "{}: no `{text}` in\n{mlir}", kernel.ir().name);
        if let Some(tc) = &tc {
            tc.verify(&mlir).unwrap_or_else(|e| panic!("{}: {e}", kernel.ir().name));
        }
    }
    // The FP8 MMA's constant descriptor carries the K = 64 bit.
    let mlir = kernels[0].0.to_mlir();
    assert!(idesc >> 29 & 1 == 1 && mlir.contains(&format!("arith.constant {} : i32", idesc as i32)), "{mlir}");
    let reuse = kernels[3].0.to_mlir();
    assert!(reuse.contains("collector::b::fill") && reuse.contains("collector::b::use"), "{reuse}");
}

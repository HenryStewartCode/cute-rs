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

//! The block-scaled tcgen05 MMAs: GEMM tiles on the
//! simulator for MXF8F6F4, MXF4, NVF4 (UE4M3 and UE8M0 scales) and the sm_103a K = 96
//! forms, with M = 128 on one CTA and M = 256 and 128 on a CTA pair, several N, the scale
//! factors staged into TMEM by `tcgen05.cp` in the layout written out below (row `32 c + l`
//! of a 128-row block in lane `l` of every 32-lane sub-partition, column `c`, the factors
//! of a row in consecutive bytes, four columns per four factors); exact against plain Rust.
//! Then the build-time checks, the dispatchers, the targets, the simulator's report of an
//! unordered copy, and the PTX (sm_100a through `ptxas` when it is on PATH; sm_103a only
//! through mlir-opt).

use cute::block::block_copy;
use cute::prelude::*;
use cute::tcgen05::{
    self, Cp4x32x128bOp, CtaGroup, Field, Ld32x32bOp, MmaMXF4NVF4Op, MmaMXF4Op, MmaMXF8F6F4Op, OperandSource,
    Repetition, SM103MmaMXF4NVF4Op, SM103MmaMXF4Op, SmemLayoutAtomKind,
};
use cute::utils::sm100::{make_blockscaled_trivial_tiled_mma, sm103_make_blockscaled_trivial_tiled_mma};

const LANE: usize = 1 << 16;

/// What a kernel gets wrong on purpose.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    /// Warp 1 issues the scale-factor copies and warp 0 the MMAs, after a barrier only.
    OtherWarpCopies,
}

/// One test: the operation, operand types, shape and layouts.
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
    /// Whether FP4 is packed two per byte (mxf4, mxf4nvf4) rather than the padded image.
    packed: bool,
    sf_e4m3: bool,
    vec: usize,
    /// Pass the scale factors through set_tensor rather than as gemm operand pairs.
    fields: bool,
    target: Arch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    E4M3,
    E5M2,
    E2M1,
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

/// The bytes per row of a `k`-element operand in shared memory.
fn row_bytes(fmt: Fmt, packed: bool, k: usize) -> usize {
    if fmt == Fmt::E2M1 && packed { k / 2 } else { k }
}

/// The global bytes of the row-major `(rows, k)` values `v`: FP8 codes; FP4 packed two per
/// byte (low nibble first), or the padded image (16 values in the first 8 bytes of each
/// 16-byte slot).
fn image(fmt: Fmt, packed: bool, v: &[f32], k: usize) -> Vec<u8> {
    let codes: Vec<u8> = v.iter().map(|&x| fmt.code(x)).collect();
    if fmt != Fmt::E2M1 {
        return codes;
    }
    if packed {
        return codes.chunks(2).map(|p| p[0] | p[1] << 4).collect();
    }
    let _ = k;
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

/// Copy rows `row0..row0 + mn` of the row-major global bytes `g` into the `(mn, kb)` byte
/// view `s` with plain stores by all 128 threads (MN-major tiles: `g` holds the transpose).
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

/// The shared-memory scale factors: `kt` blocks of 128 rows by 4 factors (CUTLASS's
/// BlockScaledBasicChunk), `(((32,4),(vec,4)), KT):(((16,4),(0,1)), 512)`.
fn sf_smem_layout(vec: usize, kt: usize) -> Layout {
    let chunk = utils::BlockScaledBasicChunk::new(vec, OperandMajorMode::K);
    Layout::from_modes([chunk.layout().clone(), Layout::new(kt, 512)])
}

/// The byte offset in shared memory of factor `s` of row `r`.
fn sf_smem_offset(r: usize, s: usize) -> usize {
    16 * (r % 32) + 4 * (r / 32) + s % 4 + 512 * (s / 4)
}

/// The TMEM scale factors as the copy writes them, in bytes: row `32 q + l` in lane `l` of
/// replica `p` (lane `l + 32 p`) and column `q`, factor `s` in byte `s % 4` of the
/// columns of block `s / 4`, four columns per block.
fn sf_tmem_layout(vec: usize, kt: usize) -> Layout {
    let dp = 4 * LANE;
    let mn = Layout::new(((32, 4), 4), ((dp, 4), 32 * dp));
    let k = Layout::new((vec, 4), (0, 1));
    Layout::from_modes([Layout::from_modes([mn, k]), Layout::new(kt, 16)])
}

/// The tile `C = (A sfa) (B sfb)^T`, C `(m, n)` row-major Float32, by one CTA or a CTA pair
/// of four warps each.
fn bs_kernel<TA: Numeric, TB: Numeric, S: Numeric>(
    name: &str,
    op: impl Into<MmaAtom>,
    c: Case,
    fault: Fault,
) -> Kernel {
    let atom: MmaAtom = op.into();
    let kb = atom.shape_mnk()[2];
    let Case { m, n, k, pair, a_kind, b_kind, a_fmt, b_fmt, packed, vec, fields, target, .. } = c;
    let v = if pair { 2 } else { 1 };
    let (ma, nb) = (m / v, n / v);
    let (kab, kbb) = (row_bytes(a_fmt, packed, k), row_bytes(b_fmt, packed, k));
    let nsf = kb / vec;
    let kt = k / vec / 4;
    let group = if pair { CtaGroup::TWO } else { CtaGroup::ONE };
    Kernel::builder(name).block_dim([128, 1, 1]).target(target).build(move || {
        let a_mn = a_kind.major_mode() == OperandMajorMode::MN;
        let b_mn = b_kind.major_mode() == OperandMajorMode::MN;
        let ga =
            param_tensor::<u8>("a", if a_mn { Layout::new((kab, m), (m, 1)) } else { Layout::new((m, kab), (kab, 1)) });
        let gb =
            param_tensor::<u8>("b", if b_mn { Layout::new((kbb, n), (n, 1)) } else { Layout::new((n, kbb), (kbb, 1)) });
        let gsfa = param_tensor::<u8>("sfa", Layout::col_major(v * kt * 512));
        let gsfb = param_tensor::<u8>("sfb", Layout::col_major(kt * 512));
        let gc = param_tensor::<f32>("c", Layout::new((m, n), (n, 1)));
        let mut mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let sa = make_smem_tensor_aligned::<TA>(operand_layout::<TA>(a_kind, ma, k), 1024);
        let sb = make_smem_tensor_aligned::<TB>(operand_layout::<TB>(b_kind, nb, k), 1024);
        let ssfa = make_smem_tensor_aligned::<u8>(sf_smem_layout(vec, kt), 128);
        let ssfb = make_smem_tensor_aligned::<u8>(sf_smem_layout(vec, kt), 128);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        let rank = if pair { block_idx_in_cluster() } else { Int32::from(0) };
        let leader = rank.eq(0);
        let acc_cols = if pair && m == 128 { n / 2 } else { n };
        let (sfa_col, sfb_col) = (acc_cols, acc_cols + 4 * kt);
        let cols = (acc_cols + 8 * kt).next_power_of_two().max(32) as i32;
        if_(warp.eq(0), || alloc_tmem(cols, &slot, pair.then_some(true), Arch::Sm100a));
        // Operands (this CTA's rows) and scale factors (SFA of this CTA's rows, SFB of all N).
        let (sa8, sb8) = (sa.recast::<u8>(), sb.recast::<u8>());
        fill(&ga, &sa8, rank * ma as i32, ma, kab, a_mn);
        fill(&gb, &sb8, rank * nb as i32, nb, kbb, b_mn);
        let raw_a = make_tensor(ssfa.ptr(), Layout::col_major(kt * 512));
        let raw_b = make_tensor(ssfb.ptr(), Layout::col_major(kt * 512));
        for_range(0, (kt * 4) as i32, 1, |j| {
            raw_a.store(j * 128 + t, gsfa.load(rank * (kt * 512) as i32 + j * 128 + t));
            raw_b.store(j * 128 + t, gsfb.load(j * 128 + t));
        });
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
        let copy_sf = || {
            let catom = make_copy_atom::<u8>(Cp4x32x128bOp::new(group));
            // One 128-row block of four factors at a time (three blocks do not tile).
            for (s, col) in [(&ssfa, sfa_col), (&ssfb, sfb_col)] {
                for blk in 0..kt {
                    let tsf = make_tensor(tmem8.clone() + (4 * col + 16 * blk) as i32, sf_tmem_layout(vec, 1));
                    let src = make_tensor(s.ptr() + (512 * blk) as i32, sf_smem_layout(vec, 1));
                    let tiled = tcgen05::make_s2t_copy(&catom, &tsf.with_layout(tsf.layout().filter_zeros()));
                    block_copy(&tiled, &src, &tsf, CopyArgs::new());
                }
            }
        };
        if fault == Fault::OtherWarpCopies {
            if_(warp.eq(1), copy_sf);
            sync_threads();
        }
        let acc =
            make_tensor(tmem.clone(), mma.make_fragment_c::<f32>(&mma.partition_shape_c((m, n))).layout().clone());
        let fa = mma.make_fragment_a(&tiled_divide(&sa, (ma, kb)));
        let fb = mma.make_fragment_b(&tiled_divide(&sb, (nb, kb)));
        if_(warp.eq(0) & leader, || {
            if fault == Fault::None {
                copy_sf();
            }
            mma.set(Field::ACCUMULATE, false);
            for j in 0..k / kb {
                // The factors of k-block j: the stream of a row's factors from factor j * nsf.
                let s0 = j * nsf;
                let off = (16 * (s0 / 4) + s0 % 4) as i32;
                let sfa = make_tensor(tmem_sf.clone() + (4 * sfa_col) as i32 + off, Layout::col_major(1));
                let sfb = make_tensor(tmem_sf.clone() + (4 * sfb_col) as i32 + off, Layout::col_major(1));
                let (aj, bj) = (fa.slice((.., .., j)), fb.slice((.., .., j)));
                if fields {
                    mma.set_tensor(Field::SFA, &sfa);
                    mma.set_tensor(Field::SFB, &sfb);
                    gemm(&mma, &acc, &aj, &bj, &acc);
                } else {
                    gemm(&mma, &acc, (&aj, &sfa), (&bj, &sfb), &acc);
                }
                mma.set(Field::ACCUMULATE, true);
            }
            let mask = pair.then(|| Int16::from(0b11i16));
            if_(elect_one(), || tcgen05::commit(&mbar, mask, group));
        });
        mbarrier_wait(&mbar, 0);
        // This CTA's half of D: rows by TMEM lane, or (pair M = 128) Layout B.
        let l = lane_idx();
        let tmem_lane = (warp % 4) * 32 + l;
        let (row, col0, chunks) = if !pair || m == 256 {
            (rank * 128 + tmem_lane, Int32::from(0), n / 32)
        } else {
            (rank * 64 + tmem_lane % 64, (tmem_lane / 64) * (n / 2) as i32, n / 64)
        };
        let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
        for chunk in 0..chunks {
            let layout = Layout::from_modes([Layout::new((32, 32), (1, LANE))]);
            let view = make_tensor(tmem.clone() + (warp % 4) * (32 * LANE) as i32 + (32 * chunk) as i32, layout);
            let frag = make_fragment::<f32>(32);
            copy(&ld, &view, &frag);
            fence_view_async_tmem_load();
            for e in 0..32 {
                gc.store((row, col0 + (32 * chunk + e) as i32), frag.load(e));
            }
        }
        fence_view_async_tmem_load();
        if pair {
            cluster_arrive();
            cluster_wait();
        } else {
            sync_threads();
        }
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(pair.then_some(true));
            dealloc_tmem(&tmem, cols, pair.then_some(true), Arch::Sm100a);
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

/// Scale factors: a value of 0.5, 1, 2 or 4 and its code.
fn scales(n: usize, seed: u64, e4m3: bool) -> Vec<(f32, u8)> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            let e = (r() % 4) as i32 - 1;
            let code = if e4m3 { (0x38 + 8 * e) as u8 } else { (127 + e) as u8 };
            (2f32.powi(e), code)
        })
        .collect()
}

/// The host operands: arguments and the reference.
fn run<TA: Numeric, TB: Numeric, S: Numeric>(name: &str, op: impl Into<MmaAtom>, c: Case) {
    let atom: MmaAtom = op.into();
    let kb = atom.shape_mnk()[2];
    let Case { m, n, k, pair, a_fmt, b_fmt, packed, sf_e4m3, vec, a_kind, b_kind, .. } = c;
    let s_per_row = k / vec;
    let kt = s_per_row / 4;
    let (av, bv) = (small_halves(m * k, 3 + m as u64 + k as u64), small_halves(n * k, 11 + n as u64));
    let (sa, sb) = (scales(m * s_per_row, 5 + n as u64, sf_e4m3), scales(n * s_per_row, 7 + m as u64, sf_e4m3));
    let transpose = |img: Vec<u8>, rows: usize, cols: usize| -> Vec<u8> {
        (0..rows * cols).map(|i| img[(i % rows) * cols + i / rows]).collect()
    };
    let a_img = image(a_fmt, packed, &av, k);
    let b_img = image(b_fmt, packed, &bv, k);
    let a_img = if a_kind.major_mode() == OperandMajorMode::MN { transpose(a_img, m, k) } else { a_img };
    let b_img = if b_kind.major_mode() == OperandMajorMode::MN { transpose(b_img, n, k) } else { b_img };
    // SFA: each CTA's rows in its 128-row block; SFB: all N rows (at most 128).
    let v = if pair { 2 } else { 1 };
    let ma = m / v;
    let mut sfa = vec![0u8; v * kt * 512];
    for r in 0..m {
        for s in 0..s_per_row {
            sfa[(r / ma) * kt * 512 + sf_smem_offset(r % ma, s)] = sa[r * s_per_row + s].1;
        }
    }
    let mut sfb = vec![0u8; kt * 512];
    for r in 0..n {
        for s in 0..s_per_row {
            sfb[sf_smem_offset(r, s)] = sb[r * s_per_row + s].1;
        }
    }
    let kernel = bs_kernel::<TA, TB, S>(name, atom, c, Fault::None);
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
    for r in 0..m {
        for col in 0..n {
            let want: f32 = (0..k)
                .map(|kk| {
                    av[r * k + kk]
                        * sa[r * s_per_row + kk / vec].0
                        * bv[col * k + kk]
                        * sb[col * s_per_row + kk / vec].0
                })
                .sum();
            assert_eq!(got[r * n + col], want, "{name}: C({r}, {col})");
        }
    }
}

use SmemLayoutAtomKind::{K_INTER, K_SW128, MN_SW128};

fn case(m: usize, n: usize, k: usize, fmts: (Fmt, Fmt), packed: bool, vec: usize) -> Case {
    Case {
        m,
        n,
        k,
        pair: false,
        a_kind: K_SW128,
        b_kind: K_SW128,
        a_fmt: fmts.0,
        b_fmt: fmts.1,
        packed,
        sf_e4m3: false,
        vec,
        fields: false,
        target: Arch::Sm100a,
    }
}

#[test]
fn mxf8f6f4_tiles_match_the_rust_reference() {
    use OperandMajorMode::{K, MN};
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    // E2M1 (padded) x E4M3, M = 128, N = 64; one factor per MMA, so the ID walks bytes 0 to 3.
    let op = MmaMXF8F6F4Op::new(DType::F4E2M1FN, DType::F8E4M3FN, (128, 64), one, s, K, K);
    run::<u8, F8E4M3FN, F8E8M0FNU>("bs_mxf8f6f4_e2m1_e4m3", op, case(128, 64, 128, (Fmt::E2M1, Fmt::E4M3), false, 32));
    // E5M2 with an MN-major B, through the SFA/SFB fields, N = 128.
    let op = MmaMXF8F6F4Op::new(DType::F8E5M2, DType::F8E5M2, (128, 128), one, s, K, MN);
    let c = Case { b_kind: MN_SW128, fields: true, ..case(128, 128, 128, (Fmt::E5M2, Fmt::E5M2), false, 32) };
    run::<F8E5M2, F8E5M2, F8E8M0FNU>("bs_mxf8f6f4_e5m2_mn", op, c);
}

#[test]
fn mxf4_and_nvf4_tiles_match_the_rust_reference() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let fp4 = (Fmt::E2M1, Fmt::E2M1);
    run::<F4E2M1FN, F4E2M1FN, F8E8M0FNU>(
        "bs_mxf4",
        MmaMXF4Op::new((128, 128), one, s),
        case(128, 128, 256, fp4, true, 32),
    );
    let c = Case { sf_e4m3: true, ..case(128, 64, 256, fp4, true, 16) };
    run::<F4E2M1FN, F4E2M1FN, F8E4M3FN>("bs_nvf4_ue4m3", MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 64), one, s), c);
    let c = Case { a_kind: K_INTER, ..case(128, 128, 128, fp4, true, 16) };
    run::<F4E2M1FN, F4E2M1FN, F8E8M0FNU>("bs_nvf4_ue8m0", MmaMXF4NVF4Op::new(DType::F8E8M0FNU, (128, 128), one, s), c);
}

#[test]
fn pair_tiles_match_the_rust_reference() {
    use OperandMajorMode::K;
    let (two, s) = (CtaGroup::TWO, OperandSource::SMEM);
    // M = 256 (Layout A): each CTA its half of A and B, SFA of its rows and all of SFB in its
    // own TMEM. A B200 gives the same.
    let op = MmaMXF8F6F4Op::new(DType::F8E4M3FN, DType::F4E2M1FN, (256, 128), two, s, K, K);
    let c = Case { pair: true, ..case(256, 128, 128, (Fmt::E4M3, Fmt::E2M1), false, 32) };
    run::<F8E4M3FN, u8, F8E8M0FNU>("bs_pair_mxf8f6f4", op, c);
}

/// A pair's M = 128 MMA uses the 2x2 data path (Layout B), which reads B's scale factors from
/// somewhere else: staged as for M = 256, a B200 gave the columns of the second half of B the
/// factors of the first. Until that layout is known the simulator does not run it.
#[test]
fn a_pairs_m128_scale_factors_are_not_modelled() {
    let (two, s) = (CtaGroup::TWO, OperandSource::SMEM);
    let c = Case { pair: true, sf_e4m3: true, ..case(128, 128, 256, (Fmt::E2M1, Fmt::E2M1), true, 16) };
    let refused = std::panic::catch_unwind(|| {
        run::<F4E2M1FN, F4E2M1FN, F8E4M3FN>("bs_pair_nvf4", MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 128), two, s), c)
    })
    .unwrap_err();
    let message = refused.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(message.contains("the 2x2 data path") && message.contains("not modelled"), "{message}");
}

/// The K = 96 forms: three or six factors per MMA (the IDs and the second group of four
/// columns), on non-swizzled operands (whose 48-byte K steps stay canonical).
#[test]
fn sm103_k96_tiles_match_the_rust_reference() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let fp4 = (Fmt::E2M1, Fmt::E2M1);
    let base = Case { a_kind: K_INTER, b_kind: K_INTER, target: Arch::Sm103a, ..case(128, 64, 384, fp4, true, 32) };
    run::<F4E2M1FN, F4E2M1FN, F8E8M0FNU>("bs_k96_mxf4", SM103MmaMXF4Op::new((128, 64), one, s), base);
    let c = Case { n: 128, k: 192, vec: 16, sf_e4m3: true, ..base };
    run::<F4E2M1FN, F4E2M1FN, F8E4M3FN>("bs_k96_nvf4", SM103MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 128), one, s), c);
}

#[test]
fn a_copy_from_another_warp_not_ordered_before_the_mma_is_reported() {
    let op = MmaMXF4Op::new((128, 64), CtaGroup::ONE, OperandSource::SMEM);
    let c = case(128, 64, 128, (Fmt::E2M1, Fmt::E2M1), true, 32);
    let kernel = bs_kernel::<F4E2M1FN, F4E2M1FN, F8E8M0FNU>("bs_unordered_cp", op, c, Fault::OtherWarpCopies);
    let mut args = vec![
        KernelArg::buffer(&vec![0u8; 128 * 64]),
        KernelArg::buffer(&vec![0u8; 64 * 64]),
        KernelArg::buffer(&vec![127u8; 512]),
        KernelArg::buffer(&vec![127u8; 512]),
        KernelArg::buffer(&vec![0f32; 128 * 64]),
    ];
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap_err().to_string();
    assert!(e.contains("data race") && e.contains("SFA") && e.contains("warp 1"), "{e}");
}

fn panic_text(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
#[allow(deprecated)]
fn illegal_block_scaled_operations_panic_when_built() {
    use OperandMajorMode::{K, MN};
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    // M = 64 on one CTA.
    let e = panic_text(|| {
        MmaMXF4Op::new((64, 64), one, s);
    });
    assert!(e.contains("M is 64") && e.contains("M = 128"), "{e}");
    // An MN-major FP4 operand.
    let e = panic_text(|| {
        MmaMXF8F6F4Op::new(DType::F4E2M1FN, DType::F8E4M3FN, (128, 64), one, s, MN, K);
    });
    assert!(e.contains("MN-major") && e.contains("K-major"), "{e}");
    // A scale type NVF4 does not take, a wrong K, and the FP8-only alias.
    let e = panic_text(|| {
        MmaMXF4NVF4Op::new(DType::F8E5M2, (128, 64), one, s);
    });
    assert!(e.contains("sf_dtype"), "{e}");
    let e = panic_text(|| {
        SM103MmaMXF4Op::new((128, 64, 64), one, s);
    });
    assert!(e.contains("K = 96"), "{e}");
    let e = panic_text(|| {
        tcgen05::MmaMXF8Op::new(DType::F4E2M1FN, (128, 64), one, s, K, K);
    });
    assert!(e.contains("MmaMXF8Op"), "{e}");
    let mxf8 = tcgen05::MmaMXF8Op::new(DType::F8E4M3FN, (128, 64), one, s, K, K);
    assert_eq!((mxf8.sf_dtype(), mxf8.sf_vec_size()), (DType::F8E8M0FNU, 32));
    // disable_output_lane on a block-scaled MMA.
    let e = panic_text(|| {
        let mut atom: MmaAtom = MmaMXF4Op::new((128, 64), one, s).into();
        atom.set_disable_output_lane(&[0i32; 4]);
    });
    assert!(e.contains("DISABLE_OUTPUT_LANE") || e.contains("disable"), "{e}");
    // Scale factors of the wrong type for the MMA (UE4M3 for MXF4's UE8M0).
    let e = panic_text(|| {
        Kernel::builder("bs_wrong_sf").block_dim([128, 1, 1]).build(|| {
            let mma = make_tiled_mma(MmaMXF4Op::new((128, 64), one, s), Layout::col_major((1, 1, 1)));
            let sa = make_smem_tensor_aligned::<F4E2M1FN>(operand_layout::<F4E2M1FN>(K_SW128, 128, 256), 1024);
            let sb = make_smem_tensor_aligned::<F4E2M1FN>(operand_layout::<F4E2M1FN>(K_SW128, 64, 256), 1024);
            let fa = mma.make_fragment_a(&tiled_divide(&sa, (128, 64)));
            let fb = mma.make_fragment_b(&tiled_divide(&sb, (64, 64)));
            let acc_ptr = make_ptr::<f32>(Int32::from(0), AddressSpace::Tmem, Some(16));
            let acc =
                make_tensor(acc_ptr, mma.make_fragment_c::<f32>(&mma.partition_shape_c((128, 64))).layout().clone());
            let sf = make_tensor(make_ptr::<F8E4M3FN>(Int32::from(64), AddressSpace::Tmem, Some(16)), 1);
            gemm(&mma, &acc, (&fa.slice((.., .., 0)), &sf), (&fb.slice((.., .., 0)), &sf), &acc);
        });
    });
    assert!(e.contains("SFA holds f8E4M3FN"), "{e}");
}

#[test]
fn the_dispatchers_pick_the_block_scaled_operations() {
    use OperandMajorMode::{K, MN};
    let (one, two, s) = (CtaGroup::ONE, CtaGroup::TWO, OperandSource::SMEM);
    let kind = |t: &TiledMma| match t.atom().op() {
        cute::MmaOp::Sm100(k) => k,
        other => panic!("{other:?}"),
    };
    let fp4 = DType::F4E2M1FN;
    let mxf4 = make_blockscaled_trivial_tiled_mma(fp4, fp4, K, K, DType::F8E8M0FNU, 32, one, (128, 128), s);
    assert_eq!(kind(&mxf4), MmaMXF4Op::new((128, 128, 64), one, s).kind());
    let nvf4 = make_blockscaled_trivial_tiled_mma(fp4, fp4, K, K, DType::F8E4M3FN, 16, two, (256, 64), s);
    assert_eq!(kind(&nvf4), MmaMXF4NVF4Op::new(DType::F8E4M3FN, (256, 64, 64), two, s).kind());
    let f8 = make_blockscaled_trivial_tiled_mma(DType::F8E4M3FN, fp4, MN, K, DType::F8E8M0FNU, 32, one, (128, 64), s);
    let want = MmaMXF8F6F4Op::new(DType::F8E4M3FN, fp4, (128, 64, 32), one, s, MN, K).kind();
    assert_eq!(kind(&f8), want);
    assert_eq!(f8.atom().shape_mnk(), [128, 64, 32]);
    let k96 = sm103_make_blockscaled_trivial_tiled_mma(DType::F8E8M0FNU, 32, one, (128, 256), s);
    assert_eq!(kind(&k96), SM103MmaMXF4Op::new((128, 256), one, s).kind());
    let k96 = sm103_make_blockscaled_trivial_tiled_mma(DType::F8E4M3FN, 16, two, (256, 128), s);
    assert_eq!(kind(&k96).k(), 96);
    // Refusals: another vector size, a mismatched scale type and size, other types.
    let e = panic_text(|| {
        make_blockscaled_trivial_tiled_mma(fp4, fp4, K, K, DType::F8E8M0FNU, 64, one, (128, 128), s);
    });
    assert!(e.contains("ValueError") && e.contains("64"), "{e}");
    let e = panic_text(|| {
        make_blockscaled_trivial_tiled_mma(fp4, fp4, K, K, DType::F8E4M3FN, 32, one, (128, 128), s);
    });
    assert!(e.contains("ValueError") && e.contains("Float8E8M0FNU"), "{e}");
    let e = panic_text(|| {
        make_blockscaled_trivial_tiled_mma(DType::F8E5M2, fp4, K, K, DType::F8E8M0FNU, 16, one, (128, 128), s);
    });
    assert!(e.contains("ValueError") && e.contains("16"), "{e}");
    let e = panic_text(|| {
        make_blockscaled_trivial_tiled_mma(DType::F16, DType::F16, K, K, DType::F8E8M0FNU, 32, one, (128, 128), s);
    });
    assert!(e.contains("TypeError"), "{e}");
    let e = panic_text(|| {
        sm103_make_blockscaled_trivial_tiled_mma(DType::F8E8M0FNU, 64, one, (128, 128), s);
    });
    assert!(e.contains("ValueError"), "{e}");
}

/// A small kernel of each op for the target checks and the PTX.
fn ptx_kernel(name: &str, which: usize, target: Arch) -> Kernel {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    let fp4 = (Fmt::E2M1, Fmt::E2M1);
    match which {
        0 => bs_kernel::<u8, F8E4M3FN, F8E8M0FNU>(
            name,
            MmaMXF8F6F4Op::new(
                DType::F4E2M1FN,
                DType::F8E4M3FN,
                (128, 64),
                one,
                s,
                OperandMajorMode::K,
                OperandMajorMode::K,
            ),
            Case { target, ..case(128, 64, 128, (Fmt::E2M1, Fmt::E4M3), false, 32) },
            Fault::None,
        ),
        1 => bs_kernel::<F4E2M1FN, F4E2M1FN, F8E8M0FNU>(
            name,
            MmaMXF4Op::new((128, 64), one, s),
            Case { target, ..case(128, 64, 256, fp4, true, 32) },
            Fault::None,
        ),
        2 => bs_kernel::<F4E2M1FN, F4E2M1FN, F8E4M3FN>(
            name,
            MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 64), one, s),
            Case { target, ..case(128, 64, 256, fp4, true, 16) },
            Fault::None,
        ),
        _ => bs_kernel::<F4E2M1FN, F4E2M1FN, F8E4M3FN>(
            name,
            SM103MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 64), one, s),
            Case { a_kind: K_INTER, b_kind: K_INTER, target, ..case(128, 64, 192, fp4, true, 16) },
            Fault::None,
        ),
    }
}

/// Two K = 96 NVF4 MMAs on non-swizzled shared tiles and fixed TMEM addresses (no allocation).
fn k96_mma_only(name: &str) -> Kernel {
    let op = SM103MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 64), CtaGroup::ONE, OperandSource::SMEM);
    Kernel::builder(name).block_dim([128, 1, 1]).target(Arch::Sm103a).build(move || {
        let mut mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let sa = make_smem_tensor_aligned::<F4E2M1FN>(operand_layout::<F4E2M1FN>(K_INTER, 128, 192), 1024);
        let sb = make_smem_tensor_aligned::<F4E2M1FN>(operand_layout::<F4E2M1FN>(K_INTER, 64, 192), 1024);
        let fa = mma.make_fragment_a(&tiled_divide(&sa, (128, 96)));
        let fb = mma.make_fragment_b(&tiled_divide(&sb, (64, 96)));
        let acc_ptr = make_ptr::<f32>(Int32::from(0), AddressSpace::Tmem, Some(16));
        let acc = make_tensor(acc_ptr, mma.make_fragment_c::<f32>(&mma.partition_shape_c((128, 64))).layout().clone());
        let sf = make_ptr::<F8E4M3FN>(Int32::from(64), AddressSpace::Tmem, Some(16));
        mma.set(Field::ACCUMULATE, false);
        for j in 0..2 {
            let off: i32 = 16 * (6 * j / 4) + 6 * j % 4;
            let sfa = make_tensor(sf.clone() + off, Layout::col_major(1));
            let sfb = make_tensor(sf.clone() + 64 + off, Layout::col_major(1));
            gemm(&mma, &acc, (&fa.slice((.., .., j)), &sfa), (&fb.slice((.., .., j)), &sfb), &acc);
            mma.set(Field::ACCUMULATE, true);
        }
    })
}

#[test]
fn targets_of_the_block_scaled_kinds() {
    let check = |k: &Kernel, a: Arch| cute::validate::check_target(k.ir(), a);
    let mxf8 = ptx_kernel("t_mxf8f6f4", 0, Arch::Sm100a);
    let mxf4 = ptx_kernel("t_mxf4", 1, Arch::Sm100a);
    let nvf4 = ptx_kernel("t_nvf4", 2, Arch::Sm100a);
    let k96 = ptx_kernel("t_k96", 3, Arch::Sm103a);
    for a in [Arch::Sm100a, Arch::Sm100f, Arch::Sm103a, Arch::Sm110a] {
        check(&mxf8, a).unwrap_or_else(|e| panic!("mxf8f6f4 {a:?}: {e}"));
    }
    for k in [&mxf4, &nvf4] {
        for a in [Arch::Sm100a, Arch::Sm103a, Arch::Sm110a] {
            check(k, a).unwrap_or_else(|e| panic!("{} {a:?}: {e}", k.ir().name));
        }
        // Not the f families.
        for a in [Arch::Sm100f, Arch::Sm103f, Arch::Sm110f] {
            let e = check(k, a).unwrap_err();
            assert!(e.message.contains("scale_vec") && e.message.contains("sm_100a"), "{a:?}: {}", e.message);
        }
    }
    check(&k96, Arch::Sm103a).unwrap();
    for a in [Arch::Sm100a, Arch::Sm100f, Arch::Sm103f, Arch::Sm110a] {
        let e = check(&k96, a).unwrap_err();
        assert!(e.message.contains("K = 96") && e.message.contains("sm_103a"), "{a:?}: {}", e.message);
    }
}

#[test]
fn block_scaled_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let cases = [
        (0, Arch::Sm100a, "kind::mxf8f6f4.block_scale ["),
        (1, Arch::Sm100a, "kind::mxf4.block_scale.scale_vec::2X ["),
        (2, Arch::Sm100a, "kind::mxf4nvf4.block_scale.scale_vec::4X ["),
        (3, Arch::Sm103a, "kind::mxf4nvf4.block_scale.block16 ["),
    ];
    for (which, arch, text) in cases {
        let name = format!("ptx_bs_{which}");
        // LLVM 22's NVVM dialect refuses tcgen05.alloc on sm_103a, so the K = 96 form is checked
        // in a kernel of just the MMAs (which are inline PTX).
        let kernel = if which == 3 { k96_mma_only(&name) } else { ptx_kernel(&name, which, arch) };
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        assert!(ptx.contains(&format!("tcgen05.mma.cta_group::1.{text}")), "{name}: no `{text}` in the PTX");
        // The sm_103a forms need PTX 8.8, which the pod's ptxas 12.8 does not assemble.
        let (Some(ptxas), Arch::Sm100a) = (&ptxas, arch) else { continue };
        let dir = std::env::temp_dir().join(format!("cute-umma-bs-{}-{name}", std::process::id()));
        let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// The SM100 shared-memory and TMEM scale-factor layouts of
// cute::utils (make_smem_layout_sfa/sfb, make_tmem_layout_sfa/sfb), checked here against
// the placements written out by hand; examples/blackwell_nvfp4_gemm.rs checks them on
// the simulator end to end.

/// The value of `l` at `((mn, k), rest_mn, rest_k)`, with `mn_size` the extent of mode 0's
/// first mode (TMEM: the replicas included, replica 0 here).
fn sf_at(l: &Layout, mn: usize, k: usize, rest_mn: usize, rest_k: usize) -> usize {
    let mn_size = l.mode(0).mode(0).size();
    let s0 = l.mode(0).size();
    l.call(mn + mn_size * k + s0 * rest_mn + s0 * l.mode(1).size() * rest_k)
}

#[test]
fn sm100_scale_factor_smem_and_tmem_layouts() {
    let (one, s) = (CtaGroup::ONE, OperandSource::SMEM);
    // nvfp4_gemm_0's tile: 128x256x256, MMAs of 128x256x64, 16-element vectors.
    let mma = make_tiled_mma(MmaMXF4NVF4Op::new(DType::F8E4M3FN, (128, 256, 64), one, s), Layout::col_major((1, 1, 1)));
    let tiler = (128, 256, 256);
    for (sfb, rows) in [(false, 128), (true, 256)] {
        let smem = if sfb {
            utils::make_smem_layout_sfb(&mma, tiler, 16, 4)
        } else {
            utils::make_smem_layout_sfa(&mma, tiler, 16, 4)
        };
        // (((128, MN / 128), (16, 4)), MMA_MN 1, MMA_K 4, STAGE 4), each stage rows x 16 bytes.
        assert_eq!(smem.rank(), 4, "{smem}");
        assert_eq!((smem.mode(0).mode(0).size(), smem.mode(0).mode(1).size()), (rows, 64), "{smem}");
        assert_eq!((smem.mode(1).size(), smem.mode(2).size(), smem.mode(3).size()), (1, 4, 4), "{smem}");
        assert_eq!(smem.mode(3).static_stride().flat_leaves(), vec![rows * 16], "{smem}");
        let stage = smem.select(&[0, 1, 2]);
        let tmem = if sfb {
            utils::make_tmem_layout_sfb(&mma, tiler, 16, &stage)
        } else {
            utils::make_tmem_layout_sfa(&mma, tiler, 16, &stage)
        };
        for r in (0..rows).step_by(7) {
            for kb in 0..4 {
                for f in 0..4 {
                    let sf = 4 * kb + f;
                    // SMEM: 512-byte chunks of 128 rows x 4 factors, K-chunks first.
                    let want = 16 * (r % 32) + 4 * (r / 32 % 4) + sf % 4 + 512 * (sf / 4 + 4 * (r / 128));
                    assert_eq!(sf_at(&stage, r, 16 * f + 3, 0, kb), want, "smem row {r} factor {sf}");
                    // TMEM: lane r % 32, column r / 32 of the k-block's rows / 32 columns, byte f.
                    let (lane, col) = (r % 32, r / 32 + rows / 32 * kb);
                    assert_eq!(
                        sf_at(&tmem, r, 16 * f + 3, 0, kb),
                        (lane << 18) + 4 * col + f,
                        "tmem row {r} factor {sf}"
                    );
                }
            }
        }
        // The replicas: the same row in lanes + 32, 64, 96.
        let mn_size = tmem.mode(0).mode(0).size();
        assert_eq!(mn_size, 4 * rows);
        assert_eq!(tmem.call(5 + 3 * rows), (5 + 96) << 18);
    }
    // MXF8F6F4 (one factor of 32 per MMA, K = 32): tmem_sf_frg's tiled_product puts the K tiles
    // in the atom's free bytes, MMA j's factor in byte j of the row's column (its scale-factor ID).
    let mma8 = make_tiled_mma(
        MmaMXF8F6F4Op::new(
            DType::F8E4M3FN,
            DType::F8E4M3FN,
            (128, 128),
            one,
            s,
            OperandMajorMode::K,
            OperandMajorMode::K,
        ),
        Layout::col_major((1, 1, 1)),
    );
    let smem = utils::make_smem_layout_sfa(&mma8, (128, 128, 128), 32, 2);
    assert_eq!((smem.mode(0).mode(1).size(), smem.mode(2).size()), (32, 4), "{smem}");
    let tmem = utils::make_tmem_layout_sfa(&mma8, (128, 128, 128), 32, &smem.select(&[0, 1, 2]));
    for j in 0..4 {
        assert_eq!(sf_at(&tmem, 33, 0, 0, j), (1 << 18) + 4 + j, "k-block {j}");
    }
    // The generic form and the SM103 forms (sm103_dense_blockscaled_gemm_persistent.py).
    let l = utils::make_smem_layout_sf((256, 128), 16, 3);
    assert_eq!((l.rank(), l.mode(2).static_stride().flat_leaves()), (3, vec![256 * 8]), "{l}");
    let k96 = sm103_make_blockscaled_trivial_tiled_mma(DType::F8E4M3FN, 16, one, (128, 256), s);
    let a = utils::sm103_make_smem_layout_sfa(&k96, (128, 256, 768), 16, 2);
    let b = utils::sm103_make_smem_layout_sfb(&k96, (128, 256, 768), 16, 2);
    for l in [&a, &b] {
        // ((atom), rest_MN, rest_K, STAGE)
        assert_eq!(l.rank(), 4, "{l}");
        let stage = l.select(&[0, 1, 2]).filter_zeros();
        assert_eq!(l.mode(3).static_stride().flat_leaves(), vec![stage.size()], "{l}");
        assert_eq!(stage.cosize(), stage.size(), "{l}");
    }
    // A pair's SFB is not laid out yet.
    let pair = make_tiled_mma(
        MmaMXF4NVF4Op::new(DType::F8E4M3FN, (256, 256, 64), CtaGroup::TWO, s),
        Layout::col_major((1, 1, 1)),
    );
    let sfb = utils::make_smem_layout_sfb(&pair, (256, 256, 256), 16, 1);
    let one_stage = sfb.select(&[0, 1, 2]);
    let e = panic_text(std::panic::AssertUnwindSafe(|| {
        utils::make_tmem_layout_sfb(&pair, (256, 256, 256), 16, &one_stage);
    }));
    assert!(e.contains("2x2"), "{e}");
}

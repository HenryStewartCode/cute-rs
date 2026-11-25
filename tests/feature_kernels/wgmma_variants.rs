// SPDX-License-Identifier: MIT OR Apache-2.0

//! The warpgroup MMA variants, held to plain Rust:
//! single-warpgroup `64 x N x 128` tiles of FP8 (every E4M3/E5M2 pair, accumulated in
//! Float32 and Float16) and 8-bit integers (every S8/U8 pair, bit-exact), built with
//! `make_trivial_tiled_mma`; and the register-sourced A form: A loaded from shared
//! memory by `ldmatrix` tiled copies (Float16, E4M3 and U8/S8, one and two
//! warpgroups), and A converted from the Float32 accumulators of an earlier MMA (the
//! `P V` step of an attention kernel: `S = Q K^T`, `P = Float16(S)`, `O = P V`).
//!
//! FP8 inputs are small integers, so Float32 sums are exact; Float16 accumulators follow
//! the simulator's documented order (each MMA adds its 32 products to C in Float32 and
//! rounds once), which the reference repeats.

use cute::prelude::*;
use cute::utils::sm90::make_trivial_tiled_mma;
use cute::warpgroup::{self, OperandSource, SmemLayoutAtomKind};

use super::{Case, operand_layout};

/// K of every tile: four k-blocks of 32 8-bit elements (eight of 16 for Float16).
pub const V_K: usize = 128;

/// Host values of the operand and accumulator types.
pub trait HostVal: Numeric + Copy {
    fn val(self) -> f64;
    /// `x` rounded to this (accumulator) type.
    fn round(x: f32) -> f32 {
        x
    }
}

impl HostVal for F8E4M3FN {
    fn val(self) -> f64 {
        self.to_f64()
    }
}

impl HostVal for F8E5M2 {
    fn val(self) -> f64 {
        self.to_f64()
    }
}

impl HostVal for F16 {
    fn val(self) -> f64 {
        self.to_f32() as f64
    }
    fn round(x: f32) -> f32 {
        F16::from_f32(x).to_f32()
    }
}

impl HostVal for f32 {
    fn val(self) -> f64 {
        self as f64
    }
}

impl HostVal for i8 {
    fn val(self) -> f64 {
        self as f64
    }
}

impl HostVal for u8 {
    fn val(self) -> f64 {
        self as f64
    }
}

impl HostVal for i32 {
    fn val(self) -> f64 {
        self as f64
    }
}

/// Copy the `(rows, V_K)` global tile `g` into the shared tile `s`, 128 threads.
fn fill<T: Numeric>(s: &Tensor<T>, g: &Tensor<T>, rows: usize, threads: usize) {
    let t = thread_idx().x;
    for_range(0, (rows * V_K / threads) as i32, 1, |j| {
        let idx = j * threads as i32 + t;
        let (r, k) = (idx % rows as i32, idx / rows as i32);
        s.store((r, k), g.load((r, k)));
    });
}

/// One `(64, N)` tile of `C = A B^T` (A `(64, V_K)`, B `(N, V_K)`, both K-major in
/// global memory, C row-major) from shared memory in `a_kind` and `b_kind` atoms, by
/// the MMA `make_trivial_tiled_mma` gives for the types.
fn ss_kernel<TA: Numeric, TB: Numeric, TC: Numeric>(
    name: &'static str,
    n: usize,
    a_kind: SmemLayoutAtomKind,
    b_kind: SmemLayoutAtomKind,
) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).build(move || {
        let a = param_tensor_aligned::<TA>("a", Layout::new((64, V_K), (V_K, 1)), 16);
        let b = param_tensor_aligned::<TB>("b", Layout::new((n, V_K), (V_K, 1)), 16);
        let c = param_tensor::<TC>("c", Layout::new((64, n), (n, 1)));
        let k = OperandMajorMode::K;
        let mut mma =
            make_trivial_tiled_mma(TA::DTYPE, TB::DTYPE, k, k, TC::DTYPE, (1, 1, 1), (64, n), OperandSource::SMEM);
        let kb_count = V_K / mma.atom().shape_mnk()[2];
        let sa = make_smem_tensor_aligned::<TA>(operand_layout::<TA>(a_kind, 64, V_K, 1), 1024);
        let sb = make_smem_tensor_aligned::<TB>(operand_layout::<TB>(b_kind, n, V_K, 1), 1024);
        fill(&sa, &a, 64, 128);
        fill(&sb, &b, n, 128);
        fence_view_async_shared();
        sync_threads();
        let thr = mma.get_slice(thread_idx().x);
        let (tcra, tcrb) = (mma.make_fragment_a(&thr.partition_a(&sa)), mma.make_fragment_b(&thr.partition_b(&sb)));
        let tcgc = thr.partition_c(&c);
        let acc = mma.make_fragment_c::<TC>(&tcgc);
        mma.set(MmaField::ACCUMULATE, false);
        warpgroup::fence();
        for kb in 0..kb_count {
            gemm(&mma, &acc, &tcra.slice((.., .., kb)), &tcrb.slice((.., .., kb)), &acc);
            mma.set(MmaField::ACCUMULATE, true);
        }
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        for i in 0..acc.size() {
            tcgc.store(i, acc.load(i));
        }
    })
}

/// The float reference: `C = A B^T` over `V_K`, each k-block of `k_inst` products added
/// to C in Float32 in order of K and rounded to `TC` (the simulator's order). A is
/// `(M, V_K)` and B `(N, V_K)`, row-major.
fn check_float<TA: HostVal, TB: HostVal, TC: HostVal>(args: &[KernelArg]) {
    let (a, b, c) = (args[0].to_vec::<TA>(), args[1].to_vec::<TB>(), args[2].to_vec::<TC>());
    let (m, n) = (a.len() / V_K, b.len() / V_K);
    let k_inst = 256 / TA::DTYPE.bits();
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for kb in 0..V_K / k_inst {
                let mut s = acc;
                for kk in kb * k_inst..(kb + 1) * k_inst {
                    s += (a[r * V_K + kk].val() * b[col * V_K + kk].val()) as f32;
                }
                acc = TC::round(s);
            }
            assert_eq!(c[r * n + col].val() as f32, acc, "C({r}, {col}) of a {m}x{n} tile");
        }
    }
}

/// The integer reference: exact sums, wrapping modulo 2^32.
fn check_int<TA: HostVal, TB: HostVal>(args: &[KernelArg]) {
    let (a, b, c) = (args[0].to_vec::<TA>(), args[1].to_vec::<TB>(), args[2].to_vec::<i32>());
    let (m, n) = (a.len() / V_K, b.len() / V_K);
    for r in 0..m {
        for col in 0..n {
            let want = (0..V_K).fold(0i32, |acc, kk| {
                acc.wrapping_add((a[r * V_K + kk].val() as i64 * b[col * V_K + kk].val() as i64) as i32)
            });
            assert_eq!(c[r * n + col], want, "C({r}, {col}) of a {m}x{n} tile");
        }
    }
}

fn f8_case<TA: HostVal, TB: HostVal, TC: HostVal>(
    name: &'static str,
    n: usize,
    a: SmemLayoutAtomKind,
    b: SmemLayoutAtomKind,
) -> Case {
    let kernel = ss_kernel::<TA, TB, TC>(name, n, a, b);
    Case {
        name,
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]),
        small_only: true,
        check: check_float::<TA, TB, TC>,
    }
}

fn i8_case<TA: HostVal, TB: HostVal>(
    name: &'static str,
    n: usize,
    a: SmemLayoutAtomKind,
    b: SmemLayoutAtomKind,
) -> Case {
    let kernel = ss_kernel::<TA, TB, i32>(name, n, a, b);
    Case { name, kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: false, check: check_int::<TA, TB> }
}

/// The FP8 and 8-bit integer tiles from shared memory.
pub fn wgmma_8bit_cases() -> Vec<Case> {
    use SmemLayoutAtomKind::*;
    type E4 = F8E4M3FN;
    type E5 = F8E5M2;
    vec![
        f8_case::<E4, E4, f32>("wgmma_f8_e4m3_e4m3_f32_n64", 64, K_SW128, K_SW128),
        f8_case::<E4, E5, f32>("wgmma_f8_e4m3_e5m2_f32_n128", 128, K_SW64, K_SW32),
        f8_case::<E5, E4, f32>("wgmma_f8_e5m2_e4m3_f32_n24", 24, K_INTER, K_SW128),
        f8_case::<E5, E5, f32>("wgmma_f8_e5m2_e5m2_f32_n256", 256, K_SW128, K_SW64),
        f8_case::<E4, E4, F16>("wgmma_f8_e4m3_e4m3_f16_n8", 8, K_SW32, K_INTER),
        f8_case::<E4, E5, F16>("wgmma_f8_e4m3_e5m2_f16_n64", 64, K_SW128, K_SW128),
        f8_case::<E5, E4, F16>("wgmma_f8_e5m2_e4m3_f16_n96", 96, K_SW64, K_SW64),
        f8_case::<E5, E5, F16>("wgmma_f8_e5m2_e5m2_f16_n32", 32, K_INTER, K_SW32),
        i8_case::<i8, i8>("wgmma_i8_s8_s8_n8", 8, K_SW128, K_SW128),
        i8_case::<i8, u8>("wgmma_i8_s8_u8_n24", 24, K_SW64, K_INTER),
        i8_case::<u8, i8>("wgmma_i8_u8_s8_n64", 64, K_SW32, K_SW128),
        i8_case::<u8, u8>("wgmma_i8_u8_u8_n256", 256, K_SW128, K_SW64),
    ]
}

/// An RS GEMM: `(64 W, N)` tile of `C = A B^T` by `W` warpgroups, A `(64 W, V_K)` loaded
/// from row-major shared memory into the MMA's register A fragment by the `ldmatrix`
/// tiled copy `make_tiled_copy_a` builds from `ldsm`, B from shared memory in a K_SW128
/// atom.
fn rs_ldmatrix_kernel<TA: Numeric, TB: Numeric, TC: Numeric>(
    name: &'static str,
    n: usize,
    warpgroups: usize,
    ldsm: CopyOp,
) -> Kernel {
    let (m, threads) = (64 * warpgroups, 128 * warpgroups);
    Kernel::builder(name).block_dim([threads as u32, 1, 1]).build(move || {
        let a = param_tensor_aligned::<TA>("a", Layout::new((m, V_K), (V_K, 1)), 16);
        let b = param_tensor_aligned::<TB>("b", Layout::new((n, V_K), (V_K, 1)), 16);
        let c = param_tensor::<TC>("c", Layout::new((m, n), (n, 1)));
        let k = OperandMajorMode::K;
        let mut mma = make_trivial_tiled_mma(
            TA::DTYPE,
            TB::DTYPE,
            k,
            k,
            TC::DTYPE,
            (warpgroups, 1, 1),
            (64, n),
            OperandSource::RMEM,
        );
        let kb_count = V_K / mma.atom().shape_mnk()[2];
        let sa = make_smem_tensor_aligned::<TA>(Layout::new((m, V_K), (V_K, 1)), 128);
        let sb = make_smem_tensor_aligned::<TB>(operand_layout::<TB>(SmemLayoutAtomKind::K_SW128, n, V_K, 1), 1024);
        fill(&sa, &a, m, threads);
        fill(&sb, &b, n, threads);
        fence_view_async_shared();
        sync_threads();
        let t = thread_idx().x;
        let thr = mma.get_slice(t);
        let fa = thr.partition_fragment_a::<TA, _>(&sa);
        let ldsm = make_tiled_copy_a(&make_copy_atom::<TA>(ldsm), &mma);
        copy(&ldsm, &ldsm.get_slice(t).partition_s(&sa), &ldsm.retile(&fa));
        let tcrb = mma.make_fragment_b(&thr.partition_b(&sb));
        let tcgc = thr.partition_c(&c);
        let acc = mma.make_fragment_c::<TC>(&tcgc);
        mma.set(MmaField::ACCUMULATE, false);
        warpgroup::fence();
        for kb in 0..kb_count {
            gemm(&mma, &acc, &fa.slice((.., .., kb)), &tcrb.slice((.., .., kb)), &acc);
            mma.set(MmaField::ACCUMULATE, true);
        }
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        for i in 0..acc.size() {
            tcgc.store(i, acc.load(i));
        }
    })
}

/// `S = Q K^T` (Q `(64, 16)`, K `(64, 16)`, from shared memory), `P = Float16(S)` in
/// registers as the A fragment of a second, register-sourced MMA, and `O = P V^T`
/// (V^T `(32, 64)`, K-major in shared memory): the `P V` step of an attention kernel,
/// where the accumulator fragment of the first MMA is, value for value, the A fragment
/// of the second.
pub fn wgmma_rs_pv() -> Case {
    const N1: usize = 64;
    const N2: usize = 32;
    const D: usize = 16;
    let kernel = Kernel::builder("wgmma_rs_pv").block_dim([128, 1, 1]).build(|| {
        let q = param_tensor_aligned::<F16>("q", Layout::new((64, D), (D, 1)), 16);
        let kt = param_tensor_aligned::<F16>("k", Layout::new((N1, D), (D, 1)), 16);
        let vt = param_tensor_aligned::<F16>("vt", Layout::new((N2, N1), (N1, 1)), 16);
        let o = param_tensor::<f32>("o", Layout::new((64, N2), (N2, 1)));
        let k = OperandMajorMode::K;
        let mut qk =
            make_trivial_tiled_mma(DType::F16, DType::F16, k, k, DType::F32, (1, 1, 1), (64, N1), OperandSource::SMEM);
        let mut pv =
            make_trivial_tiled_mma(DType::F16, DType::F16, k, k, DType::F32, (1, 1, 1), (64, N2), OperandSource::RMEM);
        use SmemLayoutAtomKind::{K_SW32, K_SW128};
        let sq = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_SW32, 64, D, 1), 1024);
        let sk = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_SW32, N1, D, 1), 1024);
        let sv = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(K_SW128, N2, N1, 1), 1024);
        let t = thread_idx().x;
        for_range(0, (64 * D / 128) as i32, 1, |j| {
            let idx = j * 128 + t;
            let (r, c) = (idx % 64, idx / 64);
            sq.store((r, c), q.load((r, c)));
            sk.store((r, c), kt.load((r, c)));
        });
        for_range(0, (N2 * N1 / 128) as i32, 1, |j| {
            let idx = j * 128 + t;
            let (r, c) = (idx % N2 as i32, idx / N2 as i32);
            sv.store((r, c), vt.load((r, c)));
        });
        fence_view_async_shared();
        sync_threads();
        // S = Q K^T.
        let thr = qk.get_slice(t);
        let (tsq, tsk) = (qk.make_fragment_a(&thr.partition_a(&sq)), qk.make_fragment_b(&thr.partition_b(&sk)));
        let acc_s = thr.partition_fragment_c::<f32, _>(&make_fragment::<f32>((64, N1)));
        qk.set(MmaField::ACCUMULATE, false);
        warpgroup::fence();
        gemm(&qk, &acc_s, &tsq, &tsk, &acc_s);
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        // P = Float16(S), as the A fragment of P V: value i of the accumulators is
        // value i of the A fragment (8 per k-block of 16).
        let thr_pv = pv.get_slice(t);
        let pa = thr_pv.partition_fragment_a::<F16, _>(&make_fragment::<F16>((64, N1)));
        assert_eq!(pa.size(), acc_s.size());
        for i in 0..pa.size() {
            pa.store(i, acc_s.load(i).to::<F16>());
        }
        let tsv = pv.make_fragment_b(&thr_pv.partition_b(&sv));
        let tgo = thr_pv.partition_c(&o);
        let acc_o = pv.make_fragment_c::<f32>(&tgo);
        pv.set(MmaField::ACCUMULATE, false);
        warpgroup::fence();
        for kb in 0..N1 / 16 {
            gemm(&pv, &acc_o, &pa.slice((.., .., kb)), &tsv.slice((.., .., kb)), &acc_o);
            pv.set(MmaField::ACCUMULATE, true);
        }
        warpgroup::commit_group();
        warpgroup::wait_group(0);
        for i in 0..acc_o.size() {
            tgo.store(i, acc_o.load(i));
        }
    });
    fn check(args: &[KernelArg]) {
        const N1: usize = 64;
        const N2: usize = 32;
        const D: usize = 16;
        let (q, k, vt, o) =
            (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<F16>(), args[3].to_vec::<f32>());
        for r in 0..64 {
            let p: Vec<f32> = (0..N1)
                .map(|j| {
                    let s: f32 = (0..D).map(|d| q[r * D + d].to_f32() * k[j * D + d].to_f32()).sum();
                    F16::from_f32(s).to_f32()
                })
                .collect();
            for col in 0..N2 {
                let want: f32 = (0..N1).map(|j| p[j] * vt[col * N1 + j].to_f32()).sum();
                assert_eq!(o[r * N2 + col], want, "O({r}, {col})");
            }
        }
    }
    Case { name: "wgmma_rs_pv", kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: true, check }
}

/// The register-sourced A cases.
pub fn wgmma_rs_cases() -> Vec<Case> {
    let ldsm16 = CopyOp::LdMatrix { num: 4, trans: false };
    let ldsm8 = CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None };
    let cfg = |w: u32| LaunchConfig::new([1, 1, 1], [128 * w, 1, 1]);
    vec![
        Case {
            name: "wgmma_rs_ldmatrix_f16_n64",
            kernel: rs_ldmatrix_kernel::<F16, F16, f32>("wgmma_rs_ldmatrix_f16_n64", 64, 1, ldsm16),
            cfg: cfg(1),
            small_only: true,
            check: check_float::<F16, F16, f32>,
        },
        Case {
            name: "wgmma_rs_ldmatrix_f16acc_two_warpgroups_n32",
            kernel: rs_ldmatrix_kernel::<F16, F16, F16>("wgmma_rs_ldmatrix_f16acc_two_warpgroups_n32", 32, 2, ldsm16),
            cfg: cfg(2),
            small_only: true,
            check: check_float::<F16, F16, F16>,
        },
        Case {
            name: "wgmma_rs_ldmatrix_e4m3_e5m2_n32",
            kernel: rs_ldmatrix_kernel::<F8E4M3FN, F8E5M2, f32>("wgmma_rs_ldmatrix_e4m3_e5m2_n32", 32, 1, ldsm8),
            cfg: cfg(1),
            small_only: true,
            check: check_float::<F8E4M3FN, F8E5M2, f32>,
        },
        Case {
            name: "wgmma_rs_ldmatrix_u8_s8_n16",
            kernel: rs_ldmatrix_kernel::<u8, i8, i32>("wgmma_rs_ldmatrix_u8_s8_n16", 16, 1, ldsm8),
            cfg: cfg(1),
            small_only: false,
            check: check_int::<u8, i8>,
        },
        wgmma_rs_pv(),
    ]
}

/// Every variant case.
pub fn wgmma_variant_cases() -> Vec<Case> {
    wgmma_8bit_cases().into_iter().chain(wgmma_rs_cases()).collect()
}

/// A sample of [`wgmma_variant_cases`] for the sm_90a feature list: FP8 with both
/// accumulators, a mixed-sign integer MMA (PTX ISA 8.4), and the register-sourced A
/// from `ldmatrix` and from an earlier accumulator.
pub fn wgmma_variant_cases_sampled() -> Vec<Case> {
    const SAMPLE: [&str; 5] = [
        "wgmma_f8_e4m3_e5m2_f32_n128",
        "wgmma_f8_e5m2_e4m3_f16_n96",
        "wgmma_i8_u8_s8_n64",
        "wgmma_rs_ldmatrix_u8_s8_n16",
        "wgmma_rs_pv",
    ];
    wgmma_variant_cases().into_iter().filter(|c| SAMPLE.contains(&c.name)).collect()
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The pipelined main loop follows the design of NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is
// cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Warpgroup MMAs (roadmap part m8-wgmma-core), held to a plain Rust GEMM: single
//! warpgroup tiles `64 x N x 64` (four k-blocks of 16) for N of 8, 64, 128 and 256, in
//! Float16 accumulated in Float32 or Float16 and BFloat16 accumulated in Float32, with
//! K-major and MN-major operands in every SM90 shared-memory layout atom; shared
//! memory filled by plain stores (then `fence_view_async_shared`) or by TMA; two
//! warpgroups; and a multi-stage TMA main loop that keeps `k_pipe_mmas` groups of MMAs
//! in flight (`wait_group(k)`), as CuTe DSL's Hopper dense GEMM does. The tests' inputs
//! are small integers, so Float32 sums are exact; Float16 accumulators follow the
//! simulator's documented order (each MMA adds its 16 products to C in Float32 and
//! rounds once), which the reference repeats.

use cute::prelude::*;
use cute::warpgroup::{self, OperandSource, SmemLayoutAtomKind, make_smem_layout_atom};

use super::Case;

/// K of every tile: four k-blocks of 16.
pub const WG_K: usize = 64;

/// How shared memory is filled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fill {
    /// Plain stores by every thread, then `fence_view_async_shared` and a barrier.
    Stores,
    /// One TMA load per operand, waited for on an mbarrier.
    Tma,
}

/// The shared-memory layout of an `(mn, k)` operand (`stages` of them) tiled from
/// `kind`'s atom: K-major atoms along M/N first, MN-major ones along K first (CuTe
/// DSL's `make_smem_layout_a/b` orders).
pub fn operand_layout<T: Numeric>(kind: SmemLayoutAtomKind, mn: usize, k: usize, stages: usize) -> ComposedLayout {
    let atom = make_smem_layout_atom::<T>(kind);
    let k_major = kind.major_mode() == OperandMajorMode::K;
    match (stages, k_major) {
        (1, true) => tile_to_shape_swizzled(&atom, (mn, k)),
        (1, false) => tile_to_shape_swizzled_ordered(&atom, (mn, k), &[1, 0]),
        (s, true) => tile_to_shape_swizzled(&atom, (mn, k, s)),
        (s, false) => tile_to_shape_swizzled_ordered(&atom, (mn, k, s), &[1, 0, 2]),
    }
}

/// The warpgroup MMA for these operands.
pub fn wgmma_op<T: Numeric, C: Numeric>(n: usize, a: SmemLayoutAtomKind, b: SmemLayoutAtomKind) -> TiledMma {
    let op =
        warpgroup::MmaF16BF16Op::new(T::DTYPE, C::DTYPE, (64, n), OperandSource::SMEM, a.major_mode(), b.major_mode());
    make_tiled_mma(op, Layout::col_major((1, 1, 1)))
}

/// The global layout of an `(mn, k)` operand: row-major (K-major) or column-major
/// (MN-major), as its shared-memory atom (TMA copies keep the majorness).
fn global_layout(kind: SmemLayoutAtomKind, mn: usize, k: usize) -> Layout {
    if kind.major_mode() == OperandMajorMode::K { Layout::new((mn, k), (k, 1)) } else { Layout::new((mn, k), (1, mn)) }
}

/// One `(M, N)` tile of `C = A B^T` (A `(M, K)`, B `(N, K)`, in global memory with the
/// majorness of their shared-memory atoms; C row-major) by `warpgroups` warpgroups stacked along M, each `64 x N`, over
/// `WG_K / 16` k-blocks: ACCUMULATE false for the first, true after.
#[allow(clippy::too_many_arguments)]
fn tile_kernel<T: Numeric, C: Numeric>(
    name: &'static str,
    n: usize,
    a_kind: SmemLayoutAtomKind,
    b_kind: SmemLayoutAtomKind,
    fill: Fill,
    warpgroups: usize,
) -> Kernel {
    let m = 64 * warpgroups;
    let threads = 128 * warpgroups;
    Kernel::builder(name).block_dim([threads as u32, 1, 1]).build(move || {
        let a = param_tensor_aligned::<T>("a", global_layout(a_kind, m, WG_K), 16);
        let b = param_tensor_aligned::<T>("b", global_layout(b_kind, n, WG_K), 16);
        let c = param_tensor::<C>("c", Layout::new((m, n), (n, 1)));
        let mma = make_tiled_mma(
            warpgroup::MmaF16BF16Op::new(
                T::DTYPE,
                C::DTYPE,
                (64, n),
                OperandSource::SMEM,
                a_kind.major_mode(),
                b_kind.major_mode(),
            ),
            Layout::col_major((warpgroups, 1, 1)),
        );
        let (la, lb) = (operand_layout::<T>(a_kind, m, WG_K, 1), operand_layout::<T>(b_kind, n, WG_K, 1));
        let t = thread_idx().x;
        let (sa, sb) = match fill {
            Fill::Stores => {
                let sa = make_smem_tensor_aligned::<T>(la, 1024);
                let sb = make_smem_tensor_aligned::<T>(lb, 1024);
                for_range(0, (m * WG_K / threads) as i32, 1, |j| {
                    let idx = j * threads as i32 + t;
                    let (r, k) = (idx % m as i32, idx / m as i32);
                    sa.store((r, k), a.load((r, k)));
                });
                for_range(0, ((n * WG_K).div_ceil(threads)) as i32, 1, |j| {
                    let idx = j * threads as i32 + t;
                    if_(idx.lt((n * WG_K) as i32), || {
                        let (r, k) = (idx % n as i32, idx / n as i32);
                        sb.store((r, k), b.load((r, k)));
                    });
                });
                fence_view_async_shared();
                sync_threads();
                (sa, sb)
            }
            Fill::Tma => {
                let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
                let (atom_a, ta) = cpasync::make_tiled_tma_atom(g2s, &a, la.clone(), (m, WG_K), 1).into();
                let (atom_b, tb) = cpasync::make_tiled_tma_atom(g2s, &b, lb.clone(), (n, WG_K), 1).into();
                let sa = make_smem_tensor_aligned::<T>(la, 1024);
                let sb = make_smem_tensor_aligned::<T>(lb, 1024);
                let mbar = alloc_smem::<u64>(1, 8);
                if_(t.eq(0), || mbarrier_init(&mbar, 1));
                mbarrier_init_fence();
                sync_threads();
                let one = make_layout(1, 0);
                let ga = local_tile(&ta, (m, WG_K), (0, 0));
                let gb = local_tile(&tb, (n, WG_K), (0, 0));
                let (tsa, tga) = cpasync::tma_partition(&atom_a, 0, &one, &sa.group_modes(0, 2), &ga.group_modes(0, 2));
                let (tsb, tgb) = cpasync::tma_partition(&atom_b, 0, &one, &sb.group_modes(0, 2), &gb.group_modes(0, 2));
                if_(warp_idx().eq(0), || {
                    let bytes = ((m + n) * WG_K * T::BYTES) as i32;
                    if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, bytes));
                    copy_with(&atom_a, &tga, &tsa, CopyArgs::new().tma_bar_ptr(&mbar));
                    copy_with(&atom_b, &tgb, &tsb, CopyArgs::new().tma_bar_ptr(&mbar));
                });
                mbarrier_wait(&mbar, 0);
                (sa, sb)
            }
        };
        let thr = mma.get_slice(t);
        let (tcsa, tcsb) = (thr.partition_a(&sa), thr.partition_b(&sb));
        let (tcra, tcrb) = (mma.make_fragment_a(&tcsa), mma.make_fragment_b(&tcsb));
        let tcgc = thr.partition_c(&c);
        let acc = mma.make_fragment_c::<C>(&tcgc);
        let mut mma = mma;
        mma.set(MmaField::ACCUMULATE, false);
        warpgroup::fence();
        for kb in 0..WG_K / 16 {
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

/// The reference: `C = A B^T` over `k` columns, each 16-column k-block added to C in
/// Float32 in order of K and rounded to `C` (the simulator's order). `at_a(r, k)` and
/// `at_b(col, k)` index the operands.
fn reference<T: Numeric + HostFloat, C: Numeric + HostFloat>(
    at_a: impl Fn(usize, usize) -> T,
    at_b: impl Fn(usize, usize) -> T,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for kb in 0..k / 16 {
                let mut s = acc;
                for kk in kb * 16..kb * 16 + 16 {
                    s += at_a(r, kk).host_f32() * at_b(col, kk).host_f32();
                }
                acc = C::round_f32(s);
            }
            out[r * n + col] = acc;
        }
    }
    out
}

/// Host conversions of the operand and accumulator types.
pub trait HostFloat: Copy {
    fn host_f32(self) -> f32;
    fn round_f32(x: f32) -> f32;
}

impl HostFloat for F16 {
    fn host_f32(self) -> f32 {
        self.to_f32()
    }
    fn round_f32(x: f32) -> f32 {
        F16::from_f32(x).to_f32()
    }
}

impl HostFloat for BF16 {
    fn host_f32(self) -> f32 {
        self.to_f32()
    }
    fn round_f32(x: f32) -> f32 {
        BF16::from_f32(x).to_f32()
    }
}

impl HostFloat for f32 {
    fn host_f32(self) -> f32 {
        self
    }
    fn round_f32(x: f32) -> f32 {
        x
    }
}

/// Check `c` against [`reference`] (A, B and C are the first three arguments; M and N
/// follow from their lengths; A and B are column-major when `A_MN` and `B_MN`).
fn check<T: Numeric + HostFloat, C: Numeric + HostFloat, const A_MN: bool, const B_MN: bool>(args: &[KernelArg]) {
    let (a, b, c) = (args[0].to_vec::<T>(), args[1].to_vec::<T>(), args[2].to_vec::<C>());
    let (m, n) = (a.len() / WG_K, b.len() / WG_K);
    let at_a = |r: usize, k: usize| if A_MN { a[k * m + r] } else { a[r * WG_K + k] };
    let at_b = |r: usize, k: usize| if B_MN { b[k * n + r] } else { b[r * WG_K + k] };
    let want = reference::<T, C>(at_a, at_b, m, n, WG_K);
    for (i, (got, want)) in c.iter().zip(&want).enumerate() {
        assert_eq!(got.host_f32(), *want, "C({}, {}) of a {m}x{n} tile", i / n, i % n);
    }
}

fn case<T: Numeric + HostFloat, C: Numeric + HostFloat>(
    name: &'static str,
    n: usize,
    a: SmemLayoutAtomKind,
    b: SmemLayoutAtomKind,
    fill: Fill,
    warpgroups: usize,
) -> Case {
    let kernel = tile_kernel::<T, C>(name, n, a, b, fill, warpgroups);
    let cfg = LaunchConfig::new([1, 1, 1], [128 * warpgroups as u32, 1, 1]);
    let check = match (a.major_mode() == OperandMajorMode::MN, b.major_mode() == OperandMajorMode::MN) {
        (false, false) => check::<T, C, false, false>,
        (false, true) => check::<T, C, false, true>,
        (true, false) => check::<T, C, true, false>,
        (true, true) => check::<T, C, true, true>,
    };
    Case { name, kernel, cfg, small_only: true, check }
}

/// Every shape, type and layout combination of the single-tile kernels.
pub fn wgmma_cases() -> Vec<Case> {
    use Fill::*;
    use SmemLayoutAtomKind::*;
    vec![
        case::<F16, f32>("wgmma_n8_kinter_kinter", 8, K_INTER, K_INTER, Stores, 1),
        case::<F16, f32>("wgmma_n8_ksw128_mninter", 8, K_SW128, MN_INTER, Stores, 1),
        case::<F16, f32>("wgmma_n64_ksw128_ksw128", 64, K_SW128, K_SW128, Stores, 1),
        case::<F16, f32>("wgmma_n64_ksw64_ksw32", 64, K_SW64, K_SW32, Stores, 1),
        case::<F16, f32>("wgmma_n64_mnsw128_mnsw128", 64, MN_SW128, MN_SW128, Stores, 1),
        case::<F16, f32>("wgmma_n64_mnsw64_mnsw32", 64, MN_SW64, MN_SW32, Stores, 1),
        case::<F16, f32>("wgmma_n64_mninter_kinter", 64, MN_INTER, K_INTER, Stores, 1),
        case::<F16, F16>("wgmma_n128_f16acc_ksw128_mnsw128", 128, K_SW128, MN_SW128, Stores, 1),
        case::<BF16, f32>("wgmma_n128_bf16_mnsw64_ksw64", 128, MN_SW64, K_SW64, Stores, 1),
        case::<F16, f32>("wgmma_n256_ksw128_ksw128", 256, K_SW128, K_SW128, Stores, 1),
        case::<BF16, f32>("wgmma_n256_bf16_ksw32_mnsw128", 256, K_SW32, MN_SW128, Stores, 1),
        case::<F16, f32>("wgmma_tma_n128_ksw128_ksw128", 128, K_SW128, K_SW128, Tma, 1),
        case::<F16, F16>("wgmma_tma_n64_f16acc_mnsw128_ksw64", 64, MN_SW128, K_SW64, Tma, 1),
        case::<F16, f32>("wgmma_two_warpgroups_n64", 64, K_SW128, K_SW128, Stores, 2),
        case::<BF16, f32>("wgmma_tma_two_warpgroups_n128", 128, K_SW128, MN_SW128, Tma, 2),
        wgmma_pipelined(2, 1),
        wgmma_pipelined(4, 2),
    ]
}

/// A sample of [`wgmma_cases`] for the sm_90a feature list (whose seeded-schedule runs
/// would otherwise take minutes): N of 8 and 256, both major modes and every swizzle,
/// Float16 accumulators, TMA, two warpgroups and the pipelined main loop.
pub fn wgmma_cases_sampled() -> Vec<Case> {
    const SAMPLE: [&str; 6] = [
        "wgmma_n8_ksw128_mninter",
        "wgmma_n64_mnsw64_mnsw32",
        "wgmma_n128_f16acc_ksw128_mnsw128",
        "wgmma_tma_two_warpgroups_n128",
        "wgmma_n256_bf16_ksw32_mnsw128",
        "wgmma_pipelined_s2_k1",
    ];
    wgmma_cases().into_iter().filter(|c| SAMPLE.contains(&c.name)).collect()
}

/// k-tiles of [`wgmma_pipelined`]: K = 8 x 64.
const PIPE_TILES: usize = 8;

/// A dense-GEMM main loop for one `64 x 128` tile over `K = 512` (8 k-tiles of 64) with
/// `stages` shared-memory stages filled by TMA (warp 0's elected lane), as CuTe DSL's
/// Hopper `dense_gemm` runs it: `k_pipe_mmas` k-tiles of MMAs issued up front, then each
/// iteration waits for its stage, issues its four k-blocks of MMAs, commits them and
/// waits until `k_pipe_mmas` groups are left (`wait_group(k_pipe_mmas)`), releases the
/// stage whose MMAs that completed, and refills it. The accumulators and the
/// `ACCUMULATE` field are carried through the loop; `wait_group(0)` ends it.
pub fn wgmma_pipelined(stages: usize, k_pipe_mmas: usize) -> Case {
    const N: usize = 128;
    const K: usize = PIPE_TILES * WG_K;
    let name: &'static str = Box::leak(format!("wgmma_pipelined_s{stages}_k{k_pipe_mmas}").into_boxed_str());
    let kernel = Kernel::builder(name).block_dim([128, 1, 1]).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, K), (K, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((N, K), (K, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((64, N), (N, 1)));
        let mut mma = wgmma_op::<F16, f32>(N, SmemLayoutAtomKind::K_SW128, SmemLayoutAtomKind::K_SW128);
        let la = operand_layout::<F16>(SmemLayoutAtomKind::K_SW128, 64, WG_K, stages);
        let lb = operand_layout::<F16>(SmemLayoutAtomKind::K_SW128, N, WG_K, stages);
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom_a, ta) = cpasync::make_tiled_tma_atom(g2s, &a, la.clone(), (64, WG_K), 1).into();
        let (atom_b, tb) = cpasync::make_tiled_tma_atom(g2s, &b, lb.clone(), (N, WG_K), 1).into();
        let sa = make_smem_tensor_aligned::<F16>(la, 1024);
        let sb = make_smem_tensor_aligned::<F16>(lb, 1024);
        let (full, empty) = (alloc_smem::<u64>(stages, 8), alloc_smem::<u64>(stages, 8));
        let t = thread_idx().x;
        if_(t.eq(0), || {
            for st in 0..stages {
                mbarrier_init(&full.offset(st), 1);
                // One arrival per consumer warp releases a stage.
                mbarrier_init(&empty.offset(st), 4);
            }
        });
        mbarrier_init_fence();
        sync_threads();
        let one = make_layout(1, 0);
        let ga = local_tile(&ta, (64, WG_K), (0, ..));
        let gb = local_tile(&tb, (N, WG_K), (0, ..));
        let (tsa, tga) = cpasync::tma_partition(&atom_a, 0, &one, &sa.group_modes(0, 2), &ga.group_modes(0, 2));
        let (tsb, tgb) = cpasync::tma_partition(&atom_b, 0, &one, &sb.group_modes(0, 2), &gb.group_modes(0, 2));
        let bytes = ((64 + N) * WG_K * 2) as i32;
        // Load k-tile `kt` into its stage once the stage is free (its previous tile's
        // MMAs released it; the first round finds every stage free).
        let produce = |kt: Int32| {
            if_(warp_idx().eq(0), || {
                let (st, round) = (kt % stages as i32, kt / stages as i32);
                if_(round.gt(0), || mbarrier_wait(&empty.offset(st), (round - 1) & 1));
                let bar = full.offset(st);
                if_(elect_one(), || {
                    mbarrier_arrive_and_expect_tx(&bar, bytes);
                    copy_with(&atom_a, &tga.slice((.., kt)), &tsa.slice((.., st)), CopyArgs::new().tma_bar_ptr(&bar));
                    copy_with(&atom_b, &tgb.slice((.., kt)), &tsb.slice((.., st)), CopyArgs::new().tma_bar_ptr(&bar));
                });
            });
        };
        for kt in 0..stages {
            produce(Int32::from(kt as i32));
        }
        let thr = mma.get_slice(t);
        let (tcsa, tcsb) = (thr.partition_a(&sa), thr.partition_b(&sb));
        let (tcra, tcrb) = (mma.make_fragment_a(&tcsa), mma.make_fragment_b(&tcsb));
        let tcgc = thr.partition_c(&c);
        let acc = thr.partition_fragment_c::<f32, _>(&c);
        // The MMAs of k-tile `kt`, from its stage.
        let mmas = |mma: &mut TiledMma, acc: &Tensor<f32>, kt: Int32| {
            let st = kt % stages as i32;
            mbarrier_wait(&full.offset(st), (kt / stages as i32) & 1);
            warpgroup::fence();
            for kb in 0..WG_K / 16 {
                gemm(mma, acc, &tcra.slice((.., .., kb, st)), &tcrb.slice((.., .., kb, st)), acc);
                mma.set(MmaField::ACCUMULATE, true);
            }
            warpgroup::commit_group();
        };
        mma.set(MmaField::ACCUMULATE, false);
        for kt in 0..k_pipe_mmas {
            mmas(&mut mma, &acc, Int32::from(kt as i32));
        }
        let (acc, _) = for_range_carry(k_pipe_mmas as i32, PIPE_TILES as i32, 1, (acc, mma), |kt, (acc, mut mma)| {
            mmas(&mut mma, &acc, kt);
            warpgroup::wait_group(k_pipe_mmas as u32);
            // The MMAs of k-tile `kt - k_pipe_mmas` are complete: release its stage,
            // and load the tile that goes there next.
            let done = kt - k_pipe_mmas as i32;
            sync_warp();
            if_(elect_one(), || mbarrier_arrive(&empty.offset(done % stages as i32)));
            let next = done + stages as i32;
            if_(next.lt(PIPE_TILES as i32), || produce(next));
            (acc, mma)
        });
        warpgroup::wait_group(0);
        for i in 0..acc.size() {
            tcgc.store(i, acc.load(i));
        }
    });
    fn check_pipe(args: &[KernelArg]) {
        const N: usize = 128;
        const K: usize = PIPE_TILES * WG_K;
        let (a, b, c) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
        for r in 0..64 {
            for col in 0..N {
                let want: f32 = (0..K).map(|k| a[r * K + k].to_f32() * b[col * K + k].to_f32()).sum();
                assert_eq!(c[r * N + col], want, "C({r}, {col})");
            }
        }
    }
    Case { name, kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: true, check: check_pipe }
}

/// One case of [`wgmma_descriptors`]: an `(mn, k)` operand tile in `kind`'s atom, A or
/// B, and the descriptor fields the PTX ISA's canonical layout gives it, worked out by
/// hand: the leading and stride byte offsets and the layout type (in the descriptor's
/// units), and the start offset of each k-block in 16-byte units.
pub struct DescCase {
    pub kind: SmemLayoutAtomKind,
    pub a: bool,
    pub mn: usize,
    pub k: usize,
    pub lbo: u64,
    pub sbo: u64,
    pub layout_type: u64,
    pub k_block_units: u64,
}

/// The [`wgmma_descriptors`] cases (Float16 operands).
pub fn desc_cases() -> Vec<DescCase> {
    use SmemLayoutAtomKind::*;
    let c = |kind, a, mn, k, lbo, sbo, layout_type, k_block_units| DescCase {
        kind,
        a,
        mn,
        k,
        lbo,
        sbo,
        layout_type,
        k_block_units,
    };
    vec![
        // 8-row groups of 128 bytes; k-blocks 32 bytes apart within a row.
        c(K_SW128, true, 128, 64, 1, 1024 / 16, 1, 2),
        // 8-row groups of 64 bytes.
        c(K_SW64, false, 64, 32, 1, 512 / 16, 2, 2),
        // 256-byte atoms (16 N x 8 K) along K first: K groups 256 bytes apart (SBO), N
        // groups after the 4 atoms of K (LBO); a k-block is two atoms.
        c(MN_SW32, false, 64, 32, 1024 / 16, 256 / 16, 3, 512 / 16),
        // 128-byte atoms (8 M x 8 K) along K first: N groups 512 bytes apart (SBO), K
        // groups 128 (LBO).
        c(MN_INTER, true, 64, 32, 128 / 16, 512 / 16, 0, 256 / 16),
        // 128-byte atoms (8 rows x 8 K) along M first: 8-row groups 128 bytes apart
        // (SBO), the second 8 columns of K 1024 bytes on (LBO); a k-block is two atoms
        // of K.
        c(K_INTER, true, 64, 32, 1024 / 16, 128 / 16, 0, 2048 / 16),
        // 256-byte atoms (8 rows x 16 K) along N first; a k-block is the next atom of K.
        c(K_SW32, false, 64, 32, 1, 256 / 16, 3, 2048 / 16),
    ]
}

/// Words of output per case and warpgroup: the tile's start address, then the
/// descriptor of each k-block.
const DESC_WORDS: usize = 8;

/// The descriptors `make_fragment_a/b` builds (no MMA is issued, so this runs on any
/// GPU): for each [`desc_cases`] tile, warpgroup 0 (thread 0) and 1 (thread 128) store
/// the tile's shared-memory address and the descriptor of each k-block of their
/// partition. The first case is tiled over two warpgroups along M (the second's rows
/// start 64 rows on); the check holds them to the hand-worked fields.
pub fn wgmma_descriptors() -> Case {
    let cases = desc_cases();
    let n_out = cases.len() * 2 * DESC_WORDS;
    let kernel = Kernel::builder("wgmma_descriptors").block_dim([256, 1, 1]).build(move || {
        let out = param_tensor::<u64>("out", Layout::col_major(n_out));
        let t = thread_idx().x;
        for (j, c) in desc_cases().into_iter().enumerate() {
            let s = make_smem_tensor_aligned::<F16>(operand_layout::<F16>(c.kind, c.mn, c.k, 1), 1024);
            let (a_kind, b_kind) =
                if c.a { (c.kind, SmemLayoutAtomKind::K_SW128) } else { (SmemLayoutAtomKind::K_SW128, c.kind) };
            let n = if c.a { 64 } else { c.mn };
            let op = warpgroup::MmaF16BF16Op::new(
                DType::F16,
                DType::F32,
                (64, n),
                OperandSource::SMEM,
                a_kind.major_mode(),
                b_kind.major_mode(),
            );
            let warpgroups = if c.a { c.mn / 64 } else { 1 };
            let mma = make_tiled_mma(op, Layout::col_major((warpgroups, 1, 1)));
            let thr = mma.get_slice(t % (128 * warpgroups) as i32);
            let frag =
                if c.a { mma.make_fragment_a(&thr.partition_a(&s)) } else { mma.make_fragment_b(&thr.partition_b(&s)) };
            let start = s.ptr().toint_i32().to::<u64>();
            for wg in 0..2 {
                if_(t.eq(128 * wg), || {
                    let base = (j * 2 + wg as usize) * DESC_WORDS;
                    out.store(base as i32, start);
                    for kb in 0..c.k / 16 {
                        out.store((base + 1 + kb) as i32, frag.smem_desc((0, 0, kb)));
                    }
                });
            }
        }
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<u64>();
        let encode = |start: u64, c: &DescCase| (start >> 4 & 0x3fff) | c.lbo << 16 | c.sbo << 32 | c.layout_type << 62;
        for (j, c) in desc_cases().iter().enumerate() {
            for wg in 0..2usize {
                let base = (j * 2 + wg) * DESC_WORDS;
                let start = out[base];
                assert_eq!(start % 1024, 0, "case {j}: the tile is 1024-byte aligned");
                // A tiled over two warpgroups (K_SW128, 128-byte rows): the second's rows
                // start 64 rows on.
                let rows = if c.a && c.mn > 64 { 64 * wg as u64 } else { 0 };
                let first = encode(start + rows * 128, c);
                for kb in 0..c.k / 16 {
                    let want = first + kb as u64 * c.k_block_units;
                    assert_eq!(
                        out[base + 1 + kb],
                        want,
                        "case {j} ({:?}), warpgroup {wg}, k-block {kb}: {:#x} (want {want:#x})",
                        c.kind,
                        out[base + 1 + kb]
                    );
                }
            }
        }
    }
    Case { name: "wgmma_descriptors", kernel, cfg: LaunchConfig::new([1, 1, 1], [256, 1, 1]), small_only: false, check }
}

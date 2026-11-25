// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// This kernel is tensorop_gemm.rs's, so it follows the design of a CuTe DSL
// example of NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and takes its tile size, its MMA, copy and shared-memory layouts and its
// swizzle from it. Those parts are under the BSD-3-Clause licence, and the rest
// of the file is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md
// has the licence text and describes what is taken.

//! One compiled kernel for every problem size: a tensor-core GEMM whose `M`,
//! `N` and `K` are only known at launch.
//!
//! It is `tensorop_gemm` with the global tensors declared differently. Tile
//! sizes stay static, because tiling and partitioning need them. The strides
//! of A, B and C depend on the runtime `k` and `n`, which only enter the
//! offsets the kernel finally computes, and the K loop runs to a runtime bound.
//!
//! The global tensors are declared two ways, with one kernel body. [`build`]
//! gives each mode a static upper bound as its extent and runtime strides
//! (`Layout::with_strides((BOUND, BOUND), (k, 1))`) and counts the K tiles
//! as `k / BK`. [`build_dynamic`] gives the tensors their runtime extents,
//! as CuTe DSL's framework tensors have them (`Layout::row_major((m, k))`,
//! `(?,?):(?,1)`), and the number of K tiles is the extent of the tiles'
//! rest mode, `size(&g_a, &[2])`. The two compute the same bits.
//!
//! Run with `cargo run --release --example runtime_gemm`.

use cute::prelude::*;

const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 32;
const STAGES: usize = 3;

/// A static upper bound on every mode. It only has to be a multiple of the
/// tile sizes; the buffers themselves are sized at launch.
const BOUND: usize = 1 << 20;

/// The kernel with static upper bounds as the extents of A, B and C.
pub fn build() -> Kernel {
    Kernel::builder("runtime_gemm").block_dim([128, 1, 1]).build(|| {
        let a = param_ptr::<F16>("a");
        let b = param_ptr::<F16>("b");
        let c = param_ptr::<f32>("c");
        // The kernel's alignment proofs rely on these being tile multiples.
        let n = param_extent("n", BN);
        let k = param_extent("k", BK);
        let a = make_tensor(&a, Layout::with_strides((BOUND, BOUND), (k, 1)));
        let b = make_tensor(&b, Layout::with_strides((BOUND, BOUND), (k, 1)));
        let c = make_tensor(&c, Layout::with_strides((BOUND, BOUND), (n, 1)));
        let k_tiles = k / BK as i32;
        gemm_body(&a, &b, &c, Some(k_tiles));
    })
}

/// The kernel with A, B and C of runtime extents: `m x k`, `n x k` and
/// `m x n`, row-major.
pub fn build_dynamic() -> Kernel {
    Kernel::builder("runtime_gemm_dynamic").block_dim([128, 1, 1]).build(|| {
        let a = param_ptr::<F16>("a");
        let b = param_ptr::<F16>("b");
        let c = param_ptr::<f32>("c");
        // The kernel's alignment proofs rely on these being tile multiples.
        let m = param_extent("m", BM);
        let n = param_extent("n", BN);
        let k = param_extent("k", BK);
        let a = make_tensor(&a, Layout::row_major((m, k)));
        let b = make_tensor(&b, Layout::row_major((n, k)));
        let c = make_tensor(&c, Layout::row_major((m, n)));
        gemm_body(&a, &b, &c, None);
    })
}

/// The kernel itself, on the global tensors `a`, `b` and `c`. `k_tiles` is
/// the number of K tiles, or `None` to take it from the tiles of `a`: the
/// extent of their rest mode, a runtime value for a tensor of runtime
/// extents.
fn gemm_body(a: &Tensor<F16>, b: &Tensor<F16>, c: &Tensor<f32>, k_tiles: Option<Int32>) {
    let tid = thread_idx().x;
    let bid = block_idx();

    let cta = (bid.x, bid.y, ..);
    let g_a = local_tile_proj(a, (BM, BN, BK), cta, &[true, false, true]);
    let g_b = local_tile_proj(b, (BM, BN, BK), cta, &[false, true, true]);
    let g_c = local_tile_proj(c, (BM, BN, BK), cta, &[true, true, false]);
    let k_tiles = k_tiles.unwrap_or_else(|| Int32::from(size(&g_a, &[2])));

    let atom = Layout::row_major((8, 32));
    let sw = make_swizzle(2, 3, 3);
    let s_a = make_smem_tensor::<F16>(make_composed_layout(sw, 0, tile_to_shape(&atom, (BM, BK, STAGES))));
    let s_b = make_smem_tensor::<F16>(make_composed_layout(sw, 0, tile_to_shape(&atom, (BN, BK, STAGES))));

    let g2s_atom = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
    let g2s = make_tiled_copy_tv(&g2s_atom, &Layout::row_major((32, 4)), &Layout::row_major((1, 8)));
    let thr_g2s = g2s.get_slice(tid);
    let t_a_g = thr_g2s.partition_s(&g_a);
    let t_a_s = thr_g2s.partition_d(&s_a);
    let t_b_g = thr_g2s.partition_s(&g_b);
    let t_b_s = thr_g2s.partition_d(&s_b);

    let mma = make_tiled_mma(make_mma_atom(MmaOp::Sm80(MmaKind::M16N8K16F32F16)), Layout::col_major((2, 2, 1)));
    let thr_mma = mma.get_slice(tid);
    let r_a = thr_mma.partition_fragment_a::<F16, _>(&s_a.slice((.., .., 0)));
    let r_b = thr_mma.partition_fragment_b::<F16, _>(&s_b.slice((.., .., 0)));
    let t_c_g = thr_mma.partition_c(&g_c);
    let acc = thr_mma.partition_fragment_c::<f32, _>(&g_c);
    acc.fill(0.0);

    let s2r_a = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
    let s2r_b = make_tiled_copy_b(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 2, trans: false }), &mma);
    let thr_s2r_a = s2r_a.get_slice(tid);
    let thr_s2r_b = s2r_b.get_slice(tid);
    let t_a_s_view = thr_s2r_a.partition_s(&s_a);
    let t_b_s_view = thr_s2r_b.partition_s(&s_b);
    let r_a_view = thr_s2r_a.retile_d(&r_a);
    let r_b_view = thr_s2r_b.retile_d(&r_b);
    let k_blocks = r_a.shape().mode(2).size();

    // Prologue: the number of K tiles is a runtime value here.
    for s in 0..STAGES - 1 {
        if_(Int32::from(s).lt(k_tiles), || {
            copy(&g2s, &t_a_g.slice((.., .., .., s)), &t_a_s.slice((.., .., .., s)));
            copy(&g2s, &t_b_g.slice((.., .., .., s)), &t_b_s.slice((.., .., .., s)));
        });
        cp_async_commit_group();
    }

    let acc = for_range_carry(0, k_tiles, 1, acc, |k_tile, acc| {
        cp_async_wait_group((STAGES - 2) as u32);
        sync_threads();

        let next = k_tile + (STAGES - 1) as i32;
        if_(next.lt(k_tiles), || {
            let stage = next % STAGES as i32;
            copy(&g2s, &t_a_g.slice((.., .., .., next)), &t_a_s.slice((.., .., .., stage)));
            copy(&g2s, &t_b_g.slice((.., .., .., next)), &t_b_s.slice((.., .., .., stage)));
        });
        cp_async_commit_group();

        let stage = k_tile % STAGES as i32;
        for kb in 0..k_blocks {
            copy(&s2r_a, &t_a_s_view.slice((.., .., kb, stage)), &r_a_view.slice((.., .., kb)));
            copy(&s2r_b, &t_b_s_view.slice((.., .., kb, stage)), &r_b_view.slice((.., .., kb)));
            gemm(&mma, &acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), &acc);
        }
        acc
    });

    autovec_copy(&acc, &t_c_g);
}

/// The launch and arguments for an `m x n x k` problem. Inputs are small
/// integers, so the result is exact.
pub fn problem(m: usize, n: usize, k: usize) -> (LaunchConfig, Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    let (a, b) = (small_integers(m * k, 1), small_integers(n * k, 2));
    let (launch, args) = problem_on(m, n, k, &a, &b);
    (launch, args, a, b)
}

/// The launch and arguments for an `m x n x k` problem on `a` (`m x k`) and
/// `b` (`n x k`), both row-major, which are rounded to f16.
pub fn problem_on(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> (LaunchConfig, Vec<KernelArg>) {
    assert!(m.is_multiple_of(BM) && n.is_multiple_of(BN) && k.is_multiple_of(BK), "size must be a tile multiple");
    assert_eq!((a.len(), b.len()), (m * k, n * k), "A is m x k and B is n x k");
    let args = vec![
        KernelArg::f16_from_f32(a),
        KernelArg::f16_from_f32(b),
        KernelArg::f32(vec![0.0; m * n]),
        KernelArg::ScalarI32(n as i32),
        KernelArg::ScalarI32(k as i32),
    ];
    (LaunchConfig::new([(m / BM) as u32, (n / BN) as u32, 1], [128, 1, 1]), args)
}

/// `len` integers from -8 to 8, from a generator seeded with `seed`. They
/// are exact in f16, and their sums of products are exact in f32 (below
/// 2^24 for any `k` up to 2^18), so the result is exact; they are random, so
/// an element read from or written to the wrong place shows.
pub fn small_integers(len: usize, seed: u64) -> Vec<f32> {
    random_bits(len, seed).map(|r| (r % 17) as f32 - 8.0).collect()
}

/// `len` values in [-1, 1) from a generator seeded with `seed`, which keep
/// all 11 significant bits of f16: their sums of products round in f32, so
/// kernels that add them in different orders would get different bits.
pub fn random_values(len: usize, seed: u64) -> Vec<f32> {
    random_bits(len, seed).map(|r| (r >> 7) as f32 / (1 << 23) as f32 - 1.0).collect()
}

/// `len` 31-bit numbers from a linear congruential generator.
fn random_bits(len: usize, seed: u64) -> impl Iterator<Item = u64> {
    let mut state = seed;
    (0..len).map(move |_| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    })
}

/// Check `c` against a CPU GEMM.
pub fn check(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &[f32]) {
    let mut zeros = 0;
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
            assert_eq!(c[i * n + j], expected, "C[{i},{j}] of {m}x{n}x{k}");
            zeros += usize::from(expected == 0.0);
        }
    }
    // C starts out 0: inputs that leave most of it 0 would let a kernel that
    // writes little of it pass.
    assert!(zeros * 4 < m * n, "{zeros} of the {} elements of C are 0: the inputs test little", m * n);
}

/// [`problem`] for [`build_dynamic`], which also takes `m`.
pub fn problem_dynamic(m: usize, n: usize, k: usize) -> (LaunchConfig, Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    let (launch, args, a, b) = problem(m, n, k);
    (launch, with_m(m, args), a, b)
}

/// The arguments of [`build`] (from [`problem`] or [`problem_on`]) as
/// [`build_dynamic`] takes them, with `m` before `n`.
pub fn with_m(m: usize, mut args: Vec<KernelArg>) -> Vec<KernelArg> {
    args.insert(3, KernelArg::ScalarI32(m as i32));
    args
}

/// Simulate one problem size with an already-built kernel.
pub fn run(kernel: &Kernel, m: usize, n: usize, k: usize) -> SimReport {
    let (launch, mut args, a, b) = problem(m, n, k);
    let report = kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
    check(m, n, k, &a, &b, args[2].as_f32());
    report
}

/// [`run`] for [`build_dynamic`]; returns the report and `C`.
pub fn run_dynamic(kernel: &Kernel, m: usize, n: usize, k: usize) -> (SimReport, Vec<f32>) {
    let (launch, mut args, a, b) = problem_dynamic(m, n, k);
    let report = kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
    check(m, n, k, &a, &b, args[2].as_f32());
    (report, args[2].as_f32().to_vec())
}

#[allow(dead_code)]
fn main() {
    let (kernel, dynamic) = (build(), build_dynamic());
    for (m, n, k) in [(128, 128, 32), (256, 128, 96), (128, 256, 160), (256, 256, 256)] {
        let report = run(&kernel, m, n, k);
        println!("{m}x{n}x{k}: correct, {} mma.sync", report.mma_instructions);
        run_dynamic(&dynamic, m, n, k);
        // Inputs whose sums round: the two kernels add in the same order.
        let (a, b) = (random_values(m * k, 3), random_values(n * k, 4));
        let (launch, mut args) = problem_on(m, n, k, &a, &b);
        let mut dynamic_args = with_m(m, args.clone());
        kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
        dynamic.simulate(launch, &mut dynamic_args).unwrap_or_else(|e| panic!("{e}"));
        assert!(args[2].as_f32().iter().zip(dynamic_args[2].as_f32()).all(|(x, y)| x.to_bits() == y.to_bits()));
        println!("{m}x{n}x{k}: correct with runtime extents, and the same bits on random inputs");
    }
    if let Some(tc) = Toolchain::discover() {
        for (name, kernel) in [("static bounds", &kernel), ("runtime extents", &dynamic)] {
            match kernel.compile_ptx(&tc, Arch::Sm80) {
                Ok(ptx) => println!("PTX ({name}): {} lines, the same for every size", ptx.lines().count()),
                Err(e) => eprintln!("PTX compilation failed: {e}"),
            }
        }
    }
}

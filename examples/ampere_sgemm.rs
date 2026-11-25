// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/sgemm.py
//   (SGemm and run),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The Ampere SIMT GEMM of the CuTe DSL examples (`sgemm.py`): `C = A *
//! B^T` in f32 on CUDA cores, one compiled kernel for any `M`, `N` and `K`.
//!
//! A is `(M, K)`, B is `(N, K)` and C is `(M, N)`, each either major mode
//! (`a_major` "m" or "k", `b_major` "n" or "k", `c_major` "m" or "n"). The
//! host tensors are marked as `sgemm.py` marks them:
//! `mark_layout_dynamic(leading_dim)` then
//! `mark_compact_shape_dynamic(leading mode, divisibility)`, so every extent
//! and every stride but the unit one is a launch-time value, and `M`, `N`
//! and `K` need not be tile multiples. (`sgemm.py` leaves A static unless
//! `--static_shape` is given, and takes each leading extent itself as its
//! divisibility, which compiles a kernel per size; here A is marked too, and
//! the divisibility is a parameter, so one kernel serves every size it
//! divides.)
//!
//! * Each block computes a 128 x 128 tile of C with 256 threads; the grid
//!   is `ceil_div(mC.shape, (bM, bN))`.
//! * A and B tiles (128 x 8) go to shared memory through a three-stage
//!   `cp.async` pipeline. For an M-major A (N-major B), `max_alignment` of
//!   the tensor's mode 0 picks 4-wide copies when it is a multiple of 16,
//!   and scalar copies otherwise; a K-major operand is copied one element at
//!   a time into a padded tile.
//! * The K residue goes first: the tiles are shifted by `domain_offset((0,
//!   residue_k, 0))`, a negative amount, and the first tile's copy is
//!   predicated with `elem_less((m, -1), (M, k))`; the others only check M
//!   (or N).
//! * A 16 x 16 `TiledMma` of scalar FMAs, permuted so each thread owns 4
//!   contiguous rows and columns, computes the tile; the k-block loop is
//!   `cutlass.range(k_block_max, unroll_full=True)`, a runtime loop that
//!   LLVM unrolls.
//! * The epilogue stores C predicated with the residues `M - bM * bidx`
//!   and `N - bN * bidy`.
//!
//! Run with `cargo run --release --example ampere_sgemm`.

use cute::pipeline::NamedBarrier;
use cute::prelude::*;

pub const BM: usize = 128;
pub const BN: usize = 128;
pub const BK: usize = 8;
pub const STAGES: usize = 3;
pub const THREADS: usize = 256;

/// The parameters of the kernel: A, B and C.
pub type Params = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// The host tensors a launch takes: A, B and C.
pub type Host = (HostTensor, HostTensor, HostTensor);

/// Whether mode 0 of a tensor is its leading dimension (CuTe DSL's
/// `LayoutEnum.from_tensor` gives `COL_MAJOR`; otherwise `ROW_MAJOR`).
fn mode0_major<T>(t: &Tensor<T>) -> bool {
    t.leading_dim() == Some(vec![0])
}

/// Build the kernel for the specs of `host`'s tensors (CuTe DSL's
/// `cute.compile(sgemm, a, b, c)`).
pub fn build(host: &Host) -> TypedKernel<Params> {
    Kernel::builder("ampere_sgemm")
        .block_dim([THREADS as u32, 1, 1])
        .jit_for(host, |m_a: Tensor<f32>, m_b: Tensor<f32>, m_c: Tensor<f32>| kernel(&m_a, &m_b, &m_c))
}

/// The tiled `cp.async` copy of A (or B): 4-wide along mode 0 when mode 0
/// is the leading mode and its `max_alignment` is a multiple of 16, one
/// element at a time otherwise.
fn g2s_copy(t: &Tensor<f32>, tile_mn: usize) -> TiledCopy<f32> {
    let (thr, val, bytes) = if mode0_major(t) {
        let num_vectorized = if t.layout().mode(0).max_alignment().is_multiple_of(16) { 4 } else { 1 };
        let major_mode_size = tile_mn / num_vectorized;
        (
            Layout::new((major_mode_size, THREADS / major_mode_size), (1, major_mode_size)),
            Layout::col_major((num_vectorized, 1)),
            4 * num_vectorized,
        )
    } else {
        (Layout::new((THREADS / BK, BK), (BK, 1)), Layout::col_major((1, 1)), 4)
    };
    let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes, bypass_l1: false });
    make_tiled_copy_tv(&atom, &thr, &val)
}

/// The kernel (`SGemm.kernel`), with the configuration `SGemm.__call__`
/// derives from the tensors.
fn kernel(m_a: &Tensor<f32>, m_b: &Tensor<f32>, m_c: &Tensor<f32>) {
    // Shared-memory tiles, padded by 4 for a K-major operand.
    let padding_a = if mode0_major(m_a) { 0 } else { 4 };
    let padding_b = if mode0_major(m_b) { 0 } else { 4 };
    let s_a_layout = Layout::new((BM, BK, STAGES), (1, BM + padding_a, BK * (BM + padding_a)));
    let s_b_layout = Layout::new((BN, BK, STAGES), (1, BN + padding_b, BK * (BN + padding_b)));

    let tiled_copy_a = g2s_copy(m_a, BM);
    let tiled_copy_b = g2s_copy(m_b, BN);

    // 16 x 16 threads of scalar FMAs, each owning 4 contiguous rows and
    // columns of a 64 x 64 block, repeated twice in M and N.
    let atoms_layout = if mode0_major(m_c) {
        Layout::new((16, THREADS / 16, 1), (1, 16, 0))
    } else {
        Layout::new((THREADS / 16, 16, 1), (16, 1, 0))
    };
    let perm_m = Layout::new((atoms_layout.mode(0).size(), 4), (4, 1));
    let perm_n = Layout::new((atoms_layout.mode(1).size(), 4), (4, 1));
    let tiled_mma = make_tiled_mma(make_mma_atom(MmaOp::UniversalFma(DType::F32)), atoms_layout)
        .with_permutation_mnk([Some(perm_m), Some(perm_n), None]);
    let barrier = NamedBarrier::new(1, THREADS as i32);

    let tidx = thread_idx().x;
    let bid = block_idx();
    let tiler_coord = (bid.x, bid.y, ..);
    let thr_mma = tiled_mma.get_slice(tidx);

    let g_a = local_tile_proj(m_a, (BM, BN, BK), tiler_coord, &[true, false, true]);
    let g_b = local_tile_proj(m_b, (BM, BN, BK), tiler_coord, &[false, true, true]);
    let g_c = local_tile_proj(m_c, (BM, BN, BK), tiler_coord, &[true, true, false]);

    // The K residue (0 or negative): the first K tile starts before the
    // tensor, so that the others are whole.
    let residue_k = IndexExpr::signed(Int32::from(size(m_a, &[1])) - BK as i32 * Int32::from(size(&g_a, &[2])));
    let g_a = domain_offset((0, residue_k.clone(), 0), &g_a);
    let g_b = domain_offset((0, residue_k.clone(), 0), &g_b);

    let s_a = make_smem_tensor::<f32>(s_a_layout);
    let s_b = make_smem_tensor::<f32>(s_b_layout);
    let thr_copy_a = tiled_copy_a.get_slice(tidx);
    let thr_copy_b = tiled_copy_b.get_slice(tidx);
    let t_a_g_a = thr_copy_a.partition_s(&g_a); // (CPY, CPY_M, CPY_K, k)
    let t_a_s_a = thr_copy_a.partition_d(&s_a); // (CPY, CPY_M, CPY_K, PIPE)
    let t_b_g_b = thr_copy_b.partition_s(&g_b);
    let t_b_s_b = thr_copy_b.partition_d(&s_b);

    // The coordinates of the tiles, shifted as the tiles are.
    let c_a = local_tile_proj(&make_identity_tensor(m_a.shape()), (BM, BN, BK), tiler_coord, &[true, false, true]);
    let c_b = local_tile_proj(&make_identity_tensor(m_b.shape()), (BM, BN, BK), tiler_coord, &[false, true, true]);
    let c_a = domain_offset((0, residue_k.clone(), 0), &c_a);
    let c_b = domain_offset((0, residue_k, 0), &c_b);
    let t_a_c_a = thr_copy_a.partition_s(&c_a).slice((.., .., .., 0));
    let t_b_c_b = thr_copy_b.partition_s(&c_b).slice((.., .., .., 0));

    // Predicates: M (or N) only, for every K tile but the first, and with
    // the K residue for the first, where `k < 0` is outside the tensor.
    let (m_extent, n_extent) = (size(m_a, &[0]), size(m_b, &[0]));
    let t_a_p_a = make_predicate(&t_a_c_a, |c| elem_less(c.mode(0).clone(), m_extent.clone()));
    let t_b_p_b = make_predicate(&t_b_c_b, |c| elem_less(c.mode(0).clone(), n_extent.clone()));
    let t_a_p_a_residue_k =
        make_predicate(&t_a_c_a, |c| elem_less((c.mode(0).clone(), -1), (m_extent.clone(), c.mode(1).clone())));
    let t_b_p_b_residue_k =
        make_predicate(&t_b_c_b, |c| elem_less((c.mode(0).clone(), -1), (n_extent.clone(), c.mode(1).clone())));

    // Prologue: the residue tile, then the next STAGES - 2 tiles.
    let k_pipe_max = STAGES as i32;
    let k_tile_count = Int32::from(size(&t_a_g_a, &[3]));
    let next_tile = |read: Int32, wrap_to: i32| select((read + 1).lt(k_tile_count), read + 1, wrap_to);
    let gmem_pipe_read = Var::new(Int32::from(0));
    copy_if(&tiled_copy_a, &t_a_p_a_residue_k, &t_a_g_a.slice((.., .., .., 0)), &t_a_s_a.slice((.., .., .., 0)));
    copy_if(&tiled_copy_b, &t_b_p_b_residue_k, &t_b_g_b.slice((.., .., .., 0)), &t_b_s_b.slice((.., .., .., 0)));
    cp_async_commit_group();
    gmem_pipe_read.set(next_tile(gmem_pipe_read.get(), 0));
    for k_tile in 1..STAGES - 1 {
        if_(Int32::from(k_tile).lt(k_tile_count), || {
            let read = gmem_pipe_read.get();
            copy_if(&tiled_copy_a, &t_a_p_a, &t_a_g_a.slice((.., .., .., read)), &t_a_s_a.slice((.., .., .., k_tile)));
            copy_if(&tiled_copy_b, &t_b_p_b, &t_b_g_b.slice((.., .., .., read)), &t_b_s_b.slice((.., .., .., k_tile)));
        });
        gmem_pipe_read.set(next_tile(gmem_pipe_read.get(), 0));
        cp_async_commit_group();
    }
    // With fewer K tiles than stages, the main loop's copies are switched
    // off (sgemm.py clears the predicates under `if k_tile_count <
    // k_pipe_max`).
    let enough_tiles = k_tile_count.ge(k_pipe_max);
    for p in [&t_a_p_a, &t_b_p_b] {
        p.map_in_place(|v| v & enough_tiles);
    }

    // The MMA partitions, and per-thread fragments that the k-block loop
    // indexes at run time (CuTe DSL's register tensors are memory, which
    // LLVM promotes to registers once the loop is unrolled).
    let t_c_s_a = thr_mma.partition_a(&s_a); // (MMA, MMA_M, MMA_K, PIPE)
    let t_c_s_b = thr_mma.partition_b(&s_b);
    let t_c_g_c = thr_mma.partition_c(&g_c); // (MMA, MMA_M, MMA_N)
    let t_c_r_a = make_rmem_tensor::<f32>(Layout::col_major(t_c_s_a.slice((.., .., .., 0)).shape().clone()));
    let t_c_r_b = make_rmem_tensor::<f32>(Layout::col_major(t_c_s_b.slice((.., .., .., 0)).shape().clone()));
    let t_c_r_c = make_rmem_tensor::<f32>(Layout::col_major(t_c_g_c.shape().clone()));
    t_c_r_c.fill(0.0);

    let smem_pipe_read = Var::new(Int32::from(0));
    let smem_pipe_write = Var::new(Int32::from(k_pipe_max - 1));
    // The stage the register fragments are loaded from (sgemm.py's tCsA_p).
    let smem_pipe_p = Var::new(smem_pipe_read.get());
    let k_block_max = t_c_r_a.layout().mode(2).size() as i32;

    if k_block_max > 1 {
        cp_async_wait_group((STAGES - 2) as u32);
        barrier.arrive_and_wait();
        let p = smem_pipe_p.get();
        autovec_copy(&t_c_s_a.slice((.., .., 0, p)), &t_c_r_a.slice((.., .., 0)));
        autovec_copy(&t_c_s_b.slice((.., .., 0, p)), &t_c_r_b.slice((.., .., 0)));
    }

    for_range(0, k_tile_count, 1, |_| {
        for_range_with(0, k_block_max, 1, LoopOptions::unroll_full(), |k_block| {
            if_(k_block.eq(k_block_max - 1), || {
                smem_pipe_p.set(smem_pipe_read.get());
                cp_async_wait_group((STAGES - 2) as u32);
                barrier.arrive_and_wait();
            });

            let k_block_next = (k_block + 1) % k_block_max;
            let p = smem_pipe_p.get();
            autovec_copy(&t_c_s_a.slice((.., .., k_block_next, p)), &t_c_r_a.slice((.., .., k_block_next)));
            autovec_copy(&t_c_s_b.slice((.., .., k_block_next, p)), &t_c_r_b.slice((.., .., k_block_next)));

            if_(k_block.eq(0), || {
                let (read, write) = (gmem_pipe_read.get(), smem_pipe_write.get());
                copy_if(
                    &tiled_copy_a,
                    &t_a_p_a,
                    &t_a_g_a.slice((.., .., .., read)),
                    &t_a_s_a.slice((.., .., .., write)),
                );
            });

            gemm(&tiled_mma, &t_c_r_c, &t_c_r_a.slice((.., .., k_block)), &t_c_r_b.slice((.., .., k_block)), &t_c_r_c);

            if_(k_block.eq(0), || {
                let (read, write) = (gmem_pipe_read.get(), smem_pipe_write.get());
                copy_if(
                    &tiled_copy_b,
                    &t_b_p_b,
                    &t_b_g_b.slice((.., .., .., read)),
                    &t_b_s_b.slice((.., .., .., write)),
                );
                cp_async_commit_group();
                let next_read = smem_pipe_read.get() + 1;
                smem_pipe_write.set(smem_pipe_read.get());
                smem_pipe_read.set(select(next_read.eq(k_pipe_max), 0, next_read));
                gmem_pipe_read.set(next_tile(read, 1));
            });
        });
    });

    cp_async_wait_group(0);
    barrier.arrive_and_wait();

    // Epilogue: the identity epilogue_op, and stores predicated with the
    // residues of this block's tile.
    let c_c = make_identity_tensor(g_c.shape());
    let t_c_p_c = thr_mma.partition_c(&c_c);
    let residue_m = Int32::from(size(m_c, &[0])) - BM as i32 * bid.x;
    let residue_n = Int32::from(size(m_c, &[1])) - BN as i32 * bid.y;
    let pred_c = make_predicate(&t_c_p_c, |c| elem_less(c.clone(), (residue_m, residue_n)));
    let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
    copy_if(&atom, &pred_c, &t_c_r_c, &t_c_g_c);
}

/// A host tensor of `rows x cols` f32 values (row-major in `values`, as a
/// framework tensor of that shape), seen with mode 0 as the unit-stride
/// mode when `mode0_major`, and marked as `sgemm.py` marks it:
/// `mark_layout_dynamic(leading_dim)`, then the leading mode's extent dynamic
/// with `divisibility`.
pub fn host_tensor(values: &[f32], rows: usize, cols: usize, mode0_major: bool, divisibility: usize) -> HostTensor {
    assert_eq!(values.len(), rows * cols);
    // The stride order, outermost mode first, is given (as PyTorch's
    // `dim_order()` would): with an extent of 1 it cannot be deduced.
    let (data, layout, leading, order) = if mode0_major {
        // Stored transposed, as a permuted framework tensor: element (i, j)
        // at i + rows * j.
        let data = (0..rows * cols).map(|e| values[(e % rows) * cols + e / rows]).collect();
        (data, Layout::new((rows, cols), (1, rows)), 0, [1, 0])
    } else {
        (values.to_vec(), Layout::new((rows, cols), (cols, 1)), 1, [0, 1])
    };
    HostTensor::from_slice(&data, layout)
        .with_assumed_align(16)
        .mark_layout_dynamic(Some(leading as i32))
        .and_then(|t| t.mark_compact_shape_dynamic(leading, Some(&order), divisibility))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The layout of one problem: the major mode of each tensor (`a_major` "m"
/// or "k", `b_major` "n" or "k", `c_major` "m" or "n") and the
/// divisibility of each leading extent that is M or N (a K leading extent
/// takes 1, so K is any size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Majors {
    pub a: char,
    pub b: char,
    pub c: char,
    pub divisibility: usize,
}

impl Majors {
    pub const fn new(a: char, b: char, c: char, divisibility: usize) -> Majors {
        Majors { a, b, c, divisibility }
    }
}

/// An `m x n x k` problem: the host tensors (C zeroed) and the inputs, A
/// (`m x k`) and B (`n x k`) row-major, from `seed`.
pub fn problem(majors: Majors, m: usize, n: usize, k: usize, seed: u64) -> (Host, Vec<f32>, Vec<f32>) {
    let a = random_values(m * k, seed);
    let b = random_values(n * k, seed + 1);
    // M and N take the divisibility; a K leading extent takes 1.
    let d = majors.divisibility;
    let (a_m, b_n) = (majors.a == 'm', majors.b == 'n');
    let host = (
        host_tensor(&a, m, k, a_m, if a_m { d } else { 1 }),
        host_tensor(&b, n, k, b_n, if b_n { d } else { 1 }),
        host_tensor(&vec![0.0; m * n], m, n, majors.c == 'm', d),
    );
    (host, a, b)
}

/// The launch of an `m x n` C: `ceil_div(mC.shape, (bM, bN))` blocks of
/// `size(atoms_layout)` threads.
pub fn launch(m: usize, n: usize) -> LaunchConfig {
    let grid = ceil_div(Shape::from((m, n)), (BM, BN));
    let dim = |i: usize| grid.mode(i).as_leaf().and_then(|e| e.as_static()).unwrap() as u32;
    LaunchConfig::new([dim(0), dim(1), 1], [THREADS as u32, 1, 1])
}

/// C as an `m x n` row-major matrix, from the host tensor.
pub fn c_row_major(c: &HostTensor, m: usize, n: usize) -> Vec<f32> {
    let data = c.to_vec::<f32>();
    let (s0, s1) = (c.stride().leaves()[0], c.stride().leaves()[1]);
    (0..m * n).map(|e| data[(e / n) * *s0 + (e % n) * *s1]).collect()
}

/// The CPU reference: each element of C an f32 FMA chain over k in order,
/// as each thread of the kernel accumulates it, so the bits must match.
pub fn reference(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).fold(0.0f32, |acc, l| a[i * k + l].mul_add(b[j * k + l], acc));
        }
    }
    c
}

/// `len` values in [-1, 1) from a generator seeded with `seed`, with 24
/// significant bits, so the sums round and their order shows.
pub fn random_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1 << 23) as f32 - 1.0
        })
        .collect()
}

/// The simulator's options for a problem with `k` columns of A. With a
/// single K tile, the last k-block of `sgemm.py`'s main loop loads the next
/// k-block from a pipeline stage no copy has written (the value is never
/// used, as the loop ends), which the simulator would report.
pub fn sim_options(k: usize) -> SimOptions {
    SimOptions { detect_uninitialized: k > BK, ..SimOptions::default() }
}

/// Simulate one problem with an already-built kernel and check C bit for
/// bit; returns the report and C (row-major).
pub fn run(kernel: &TypedKernel<Params>, majors: Majors, m: usize, n: usize, k: usize) -> (SimReport, Vec<f32>) {
    let (mut host, a, b) = problem(majors, m, n, k, 7);
    let report =
        kernel.simulate_with(launch(m, n), &mut host, &sim_options(k)).unwrap_or_else(|e| panic!("{m}x{n}x{k}: {e}"));
    let c = c_row_major(&host.2, m, n);
    let want = reference(m, n, k, &a, &b);
    for (e, (got, want)) in c.iter().zip(&want).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "{majors:?} {m}x{n}x{k}: C[{}, {}]", e / n, e % n);
    }
    (report, c)
}

/// The kernel for `majors`, built from the specs of a problem's tensors
/// (any size the divisibility divides gives the same specs).
pub fn build_for(majors: Majors) -> TypedKernel<Params> {
    let d = majors.divisibility;
    build(&problem(majors, 16 * d, 16 * d, 16 * d, 1).0)
}

#[allow(dead_code)]
fn main() {
    for majors in [Majors::new('m', 'n', 'n', 16), Majors::new('k', 'k', 'm', 1)] {
        let kernel = build_for(majors);
        println!("A {}-major, B {}-major, C {}-major: {}", majors.a, majors.b, majors.c, kernel.spec().0);
        for (m, n, k) in [(128usize, 128usize, 8), (200, 144, 37), (64, 272, 5)] {
            let (m, n) = (m.next_multiple_of(majors.divisibility), n.next_multiple_of(majors.divisibility));
            let (report, _) = run(&kernel, majors, m, n, k);
            println!("  {m}x{n}x{k}: correct, bit for bit ({} warp instructions)", report.warp_instructions);
        }
        if let Some(tc) = Toolchain::discover() {
            match kernel.compile_ptx(&tc, Arch::Sm80) {
                Ok(ptx) => println!("  PTX: {} lines, the same for every size", ptx.lines().count()),
                Err(e) => eprintln!("  PTX compilation failed: {e}"),
            }
        }
    }
}

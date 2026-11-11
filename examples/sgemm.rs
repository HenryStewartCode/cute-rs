//! Single-precision GEMM on CUDA cores, following CuTe's `sgemm_2` tutorial.
//!
//! `C = alpha * A * B^T + beta * C` with A `(M, K)` and B `(N, K)` both
//! M/N-major (CuTe's "NT" GEMM) and C `(M, N)` column-major.
//!
//! * Each block computes a 128 x 128 tile of C with 256 threads.
//! * A and B tiles (128 x 8) are copied global -> shared with 128-bit
//!   vector copies: 32 x 8 threads, 4 contiguous values each.
//! * A 16 x 16 `TiledMma` of scalar FMAs gives every thread an 8 x 8 block of
//!   C, strided by 16 in M and N.
//! * The K loop runs on the GPU (`for_range_carry`), carrying the
//!   accumulator fragment from one iteration to the next.
//!
//! Run with `cargo run --release --example sgemm`.

use cute::prelude::*;

const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 8;

/// Build the kernel for an `m x n x k` problem (multiples of the tile).
pub fn build(m: usize, n: usize, k: usize) -> Kernel {
    assert!(
        m.is_multiple_of(BM) && n.is_multiple_of(BN) && k.is_multiple_of(BK),
        "problem size must be a multiple of the tile size"
    );
    Kernel::builder("sgemm_nt").block_dim([256, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major((m, k)));
        let b = param_tensor::<f32>("b", Layout::col_major((n, k)));
        let c = param_tensor::<f32>("c", Layout::col_major((m, n)));
        let alpha = param_scalar::<f32>("alpha");
        let beta = param_scalar::<f32>("beta");
        let tid = thread_idx().x;
        let bid = block_idx();

        // This block's tiles: gA is (BM, BK, k), gB is (BN, BK, k).
        let cta_coord = (bid.x, bid.y, ..);
        let g_a = local_tile_proj(&a, (BM, BN, BK), cta_coord, &[true, false, true]);
        let g_b = local_tile_proj(&b, (BM, BN, BK), cta_coord, &[false, true, true]);
        let g_c = local_tile_proj(&c, (BM, BN, BK), cta_coord, &[true, true, false]);

        let s_a = make_smem_tensor::<f32>(Layout::col_major((BM, BK)));
        let s_b = make_smem_tensor::<f32>(Layout::col_major((BN, BK)));

        // Global -> shared: 32 x 8 threads, 4 contiguous M (or N) values each.
        let copy_atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        let copy_a = make_tiled_copy_tv(&copy_atom, &Layout::col_major((32, 8)), &Layout::col_major((4, 1)));
        let copy_b = copy_a.clone();
        let thr_copy_a = copy_a.get_slice(tid);
        let thr_copy_b = copy_b.get_slice(tid);
        let t_a_g = thr_copy_a.partition_s(&g_a); // (CPY, CPY_M, CPY_K, k)
        let t_a_s = thr_copy_a.partition_d(&s_a); // (CPY, CPY_M, CPY_K)
        let t_b_g = thr_copy_b.partition_s(&g_b);
        let t_b_s = thr_copy_b.partition_d(&s_b);

        // 16 x 16 threads of scalar FMAs.
        let mma = make_tiled_mma(make_mma_atom(MmaOp::UniversalFma(DType::F32)), Layout::col_major((16, 16, 1)));
        let thr_mma = mma.get_slice(tid);
        let t_c_sa = thr_mma.partition_a(&s_a); // (MMA, MMA_M, MMA_K)
        let t_c_sb = thr_mma.partition_b(&s_b); // (MMA, MMA_N, MMA_K)
        let t_c_gc = thr_mma.partition_c(&g_c); // (MMA, MMA_M, MMA_N)
        let acc = thr_mma.partition_fragment_c::<f32, _>(&g_c);
        acc.fill(0.0);

        let acc = for_range_carry(0, k / BK, 1, acc, |k_tile, acc| {
            copy(&copy_a, &t_a_g.slice((.., .., .., k_tile)), &t_a_s);
            copy(&copy_b, &t_b_g.slice((.., .., .., k_tile)), &t_b_s);
            sync_threads();
            gemm(&mma, &acc, &t_c_sa, &t_c_sb, &acc);
            sync_threads();
            acc
        });

        // Epilogue: C = alpha * acc + beta * C.
        for i in 0..acc.size() {
            let out = alpha * acc.load(i) + beta * t_c_gc.load(i);
            t_c_gc.store(i, out);
        }
    })
}

/// Simulate an `m x n x k` problem and check it against the CPU.
pub fn run(m: usize, n: usize, k: usize) -> SimReport {
    let kernel = build(m, n, k);

    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i * 3) % 13) as f32 - 6.0).collect();
    let c0: Vec<f32> = (0..m * n).map(|i| (i % 5) as f32).collect();
    let (alpha, beta) = (1.5f32, -0.5f32);

    let mut args = vec![
        KernelArg::f32(a.clone()),
        KernelArg::f32(b.clone()),
        KernelArg::f32(c0.clone()),
        KernelArg::ScalarF32(alpha),
        KernelArg::ScalarF32(beta),
    ];
    let cfg = LaunchConfig::new([(m / BM) as u32, (n / BN) as u32, 1], [256, 1, 1]);
    let report = kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));

    let c = args[2].as_f32();
    for i in 0..m {
        for j in 0..n {
            let dot: f32 = (0..k).map(|l| a[i + l * m] * b[j + l * n]).sum();
            let expected = alpha * dot + beta * c0[i + j * m];
            let got = c[i + j * m];
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "C[{i},{j}] = {got}, expected {expected}"
            );
        }
    }
    report
}

#[allow(dead_code)]
fn main() {
    let (m, n, k) = (256, 256, 64);
    let report = run(m, n, k);
    println!("sgemm_nt: correct ({m}x{n}x{k})\n{report}");

    if let Some(tc) = Toolchain::discover() {
        match build(m, n, k).compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let ffma = ptx.lines().filter(|l| l.trim_start().starts_with("fma.rn.f32")).count();
                println!("\nPTX: {} lines, {ffma} fma.rn.f32 in the unrolled K-block loop body", ptx.lines().count());
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        }
    }
}

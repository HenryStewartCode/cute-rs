//! A tensor-core GEMM for SM80 (Ampere), in the style of CuTe DSL's
//! `ampere/tensorop_gemm` example.
//!
//! `C = A * B^T` with f16 (or bf16) inputs and f32 accumulation, A `(M, K)`
//! and B `(N, K)` both K-major (CuTe's "TN" GEMM), C `(M, N)` row-major.
//!
//! * 128 threads (4 warps, 2 x 2) compute a `BM x BN` tile of C with
//!   `mma.sync.m16n8k16`.
//! * A `stages`-deep `cp.async` pipeline streams `BM x BK` and `BN x BK`
//!   tiles of A and B into shared memory while earlier tiles are consumed.
//! * Shared memory is swizzled (`Swizzle<2,3,3>` over 8 x 32 atoms) so that
//!   `ldmatrix` reads are free of bank conflicts; the simulator's report
//!   shows the difference against an unswizzled layout.
//! * Fragments move from shared memory to registers with `ldmatrix`, using
//!   tiled copies derived from the MMA's own operand layouts.
//! * The K loop runs on the GPU and carries the 128 accumulators per thread.
//!
//! Run with `cargo run --release --example tensorop_gemm`.

use cute::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub stages: usize,
    pub swizzle: bool,
    /// The type of A and B: `DType::F16` or `DType::BF16`.
    pub ab: DType,
}

impl GemmConfig {
    pub fn new(m: usize, n: usize, k: usize) -> GemmConfig {
        GemmConfig { m, n, k, bm: 128, bn: 128, bk: 32, stages: 3, swizzle: true, ab: DType::F16 }
    }
}

pub fn build(cfg: GemmConfig) -> Kernel {
    match cfg.ab {
        DType::F16 => build_typed::<F16>(cfg),
        DType::BF16 => build_typed::<BF16>(cfg),
        other => panic!("tensorop_gemm takes f16 or bf16 inputs, not {other}"),
    }
}

/// The kernel for inputs of type `AB` (`F16` or `BF16`).
fn build_typed<AB: Numeric>(cfg: GemmConfig) -> Kernel {
    let GemmConfig { m, n, k, bm, bn, bk, stages, swizzle, .. } = cfg;
    assert!(
        m.is_multiple_of(bm) && n.is_multiple_of(bn) && k.is_multiple_of(bk),
        "problem size must be a multiple of the tile size"
    );
    assert!(bk == 32, "the shared-memory swizzle atom below assumes BK = 32");
    let k_tiles = k / bk;
    let name = match (swizzle, AB::DTYPE) {
        (true, DType::F16) => "tensorop_gemm",
        (false, DType::F16) => "tensorop_gemm_unswizzled",
        (true, _) => "tensorop_gemm_bf16",
        (false, _) => "tensorop_gemm_bf16_unswizzled",
    };

    Kernel::builder(name).block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<AB>("a", Layout::row_major((m, k)));
        let b = param_tensor::<AB>("b", Layout::row_major((n, k)));
        let c = param_tensor::<f32>("c", Layout::row_major((m, n)));
        let tid = thread_idx().x;
        let bid = block_idx();

        let cta = (bid.x, bid.y, ..);
        let g_a = local_tile_proj(&a, (bm, bn, bk), cta, &[true, false, true]); // (BM, BK, k)
        let g_b = local_tile_proj(&b, (bm, bn, bk), cta, &[false, true, true]); // (BN, BK, k)
        let g_c = local_tile_proj(&c, (bm, bn, bk), cta, &[true, true, false]); // (BM, BN)

        // Shared memory: `stages` K-major tiles of A and B. Each 8 x 32 atom
        // is a row-major block of eight 64-byte rows; the swizzle XORs the
        // row (bits 6..8 of the offset) into the 16-byte chunk index (bits
        // 3..5) so the eight rows `ldmatrix` reads hit eight different bank
        // groups.
        let atom = Layout::row_major((8, 32));
        let s_a_layout = tile_to_shape(&atom, (bm, bk, stages));
        let s_b_layout = tile_to_shape(&atom, (bn, bk, stages));
        let (s_a, s_b) = if swizzle {
            let sw = make_swizzle(2, 3, 3);
            (
                make_smem_tensor::<AB>(make_composed_layout(sw, s_a_layout)),
                make_smem_tensor::<AB>(make_composed_layout(sw, s_b_layout)),
            )
        } else {
            (make_smem_tensor::<AB>(s_a_layout), make_smem_tensor::<AB>(s_b_layout))
        };

        // Global -> shared with 16-byte cp.async: 32 x 4 threads, each moving
        // 8 contiguous K values.
        let g2s_atom = make_copy_atom::<AB>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let g2s = make_tiled_copy_tv(&g2s_atom, &Layout::row_major((32, 4)), &Layout::row_major((1, 8)));
        let thr_g2s = g2s.get_slice(tid);
        let t_a_g = thr_g2s.partition_s(&g_a); // (CPY, CPY_M, CPY_K, k)
        let t_a_s = thr_g2s.partition_d(&s_a); // (CPY, CPY_M, CPY_K, stage)
        let t_b_g = thr_g2s.partition_s(&g_b);
        let t_b_s = thr_g2s.partition_d(&s_b);

        // 2 x 2 warps of m16n8k16 tensor-core MMAs with f32 accumulators.
        let mma = make_tiled_mma(MmaF16BF16Op::new(AB::DTYPE, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)));
        let thr_mma = mma.get_slice(tid);
        let r_a = thr_mma.partition_fragment_a::<AB, _>(&s_a.slice((.., .., 0))); // (MMA, MMA_M, MMA_K)
        let r_b = thr_mma.partition_fragment_b::<AB, _>(&s_b.slice((.., .., 0))); // (MMA, MMA_N, MMA_K)
        let t_c_g = thr_mma.partition_c(&g_c); // (MMA, MMA_M, MMA_N)
        let acc = thr_mma.partition_fragment_c::<f32, _>(&g_c);
        acc.fill(0.0);

        // Shared -> registers with ldmatrix, laid out exactly like the MMA's
        // A and B fragments.
        let s2r_a = make_tiled_copy_a(&make_copy_atom::<AB>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
        let s2r_b = make_tiled_copy_b(&make_copy_atom::<AB>(CopyOp::LdMatrix { num: 2, trans: false }), &mma);
        let thr_s2r_a = s2r_a.get_slice(tid);
        let thr_s2r_b = s2r_b.get_slice(tid);
        let t_a_s_view = thr_s2r_a.partition_s(&s_a); // (CPY, CPY_M, CPY_K, stage)
        let t_b_s_view = thr_s2r_b.partition_s(&s_b);
        let r_a_view = thr_s2r_a.retile_d(&r_a);
        let r_b_view = thr_s2r_b.retile_d(&r_b);
        let k_blocks = r_a.shape().mode(2).size();

        // Prologue: start loading the first `stages - 1` K tiles.
        for s in 0..stages - 1 {
            if s < k_tiles {
                copy(&g2s, &t_a_g.slice((.., .., .., s)), &t_a_s.slice((.., .., .., s)));
                copy(&g2s, &t_b_g.slice((.., .., .., s)), &t_b_s.slice((.., .., .., s)));
            }
            cp_async_commit_group();
        }

        let acc = for_range_carry(0, k_tiles, 1, acc, |k_tile, acc| {
            // Tile `k_tile` has landed once at most `stages - 2` newer
            // groups are still in flight; the barrier publishes every
            // thread's copies and retires the previous iteration's reads of
            // the stage about to be refilled.
            cp_async_wait_group((stages - 2) as u32);
            sync_threads();

            let next = k_tile + (stages - 1) as i32;
            if_(next.lt(k_tiles), || {
                let stage = next % stages as i32;
                copy(&g2s, &t_a_g.slice((.., .., .., next)), &t_a_s.slice((.., .., .., stage)));
                copy(&g2s, &t_b_g.slice((.., .., .., next)), &t_b_s.slice((.., .., .., stage)));
            });
            cp_async_commit_group();

            let stage = k_tile % stages as i32;
            for kb in 0..k_blocks {
                copy(&s2r_a, &t_a_s_view.slice((.., .., kb, stage)), &r_a_view.slice((.., .., kb)));
                copy(&s2r_b, &t_b_s_view.slice((.., .., kb, stage)), &r_b_view.slice((.., .., kb)));
                gemm(&mma, &acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), &acc);
            }
            acc
        });

        // Epilogue: registers -> global, two contiguous f32 per store.
        autovec_copy(&acc, &t_c_g);
    })
}

/// Simulate `cfg` on small-integer inputs and check against a CPU GEMM.
pub fn run(cfg: GemmConfig) -> SimReport {
    let kernel = build(cfg);
    let GemmConfig { m, n, k, .. } = cfg;
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i * 5 + 1) % 5) as f32 - 2.0).collect();
    let input = |v: &[f32]| match cfg.ab {
        DType::BF16 => KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x)).collect::<Vec<_>>()),
        _ => KernelArg::f16_from_f32(v),
    };
    let mut args = vec![input(&a), input(&b), KernelArg::f32(vec![0.0; m * n])];
    let launch = LaunchConfig::new([(m / cfg.bm) as u32, (n / cfg.bn) as u32, 1], [128, 1, 1]);
    let report = kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let c = args[2].as_f32();
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
            assert_eq!(c[i * n + j], expected, "C[{i},{j}]");
        }
    }
    report
}

#[allow(dead_code)]
fn main() {
    let cfg = GemmConfig::new(256, 256, 256);
    let report = run(cfg);
    println!("tensorop_gemm {}x{}x{}: correct\n{report}\n", cfg.m, cfg.n, cfg.k);

    let unswizzled = run(GemmConfig { swizzle: false, ..cfg });
    println!(
        "without the swizzle, ldmatrix and shared-memory traffic is {:.2}x serialised by bank conflicts (vs {:.2}x)\n",
        unswizzled.bank_conflict_factor(),
        report.bank_conflict_factor()
    );

    match Toolchain::discover() {
        Some(tc) => match build(cfg).compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let count = |pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
                println!(
                    "PTX: {} lines; {} mma.sync, {} ldmatrix, {} cp.async per K-tile iteration",
                    ptx.lines().count(),
                    count("mma.sync.aligned.m16n8k16"),
                    count("ldmatrix.sync"),
                    count("cp.async.cg"),
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

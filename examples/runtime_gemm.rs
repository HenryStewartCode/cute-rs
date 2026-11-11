//! One compiled kernel for every problem size: a tensor-core GEMM whose `M`,
//! `N` and `K` are only known at launch.
//!
//! It is `tensorop_gemm` with the global tensors declared differently. Tile
//! sizes stay static, because tiling and partitioning need them. The strides
//! of A, B and C depend on the runtime `k` and `n`, which only enter the
//! offsets the kernel finally computes, and the K loop runs to a runtime bound.
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
        let tid = thread_idx().x;
        let bid = block_idx();

        let cta = (bid.x, bid.y, ..);
        let g_a = local_tile_proj(&a, (BM, BN, BK), cta, &[true, false, true]);
        let g_b = local_tile_proj(&b, (BM, BN, BK), cta, &[false, true, true]);
        let g_c = local_tile_proj(&c, (BM, BN, BK), cta, &[true, true, false]);

        let atom = Layout::row_major((8, 32));
        let sw = make_swizzle(2, 3, 3);
        let s_a = make_smem_tensor::<F16>(make_composed_layout(sw, tile_to_shape(&atom, (BM, BK, STAGES))));
        let s_b = make_smem_tensor::<F16>(make_composed_layout(sw, tile_to_shape(&atom, (BN, BK, STAGES))));

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
    })
}

/// The launch and arguments for an `m x n x k` problem. Inputs are small
/// integers, so the result is exact.
pub fn problem(m: usize, n: usize, k: usize) -> (LaunchConfig, Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    assert!(m.is_multiple_of(BM) && n.is_multiple_of(BN) && k.is_multiple_of(BK), "size must be a tile multiple");
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i * 5 + 1) % 5) as f32 - 2.0).collect();
    let args = vec![
        KernelArg::f16_from_f32(&a),
        KernelArg::f16_from_f32(&b),
        KernelArg::f32(vec![0.0; m * n]),
        KernelArg::ScalarI32(n as i32),
        KernelArg::ScalarI32(k as i32),
    ];
    (LaunchConfig::new([(m / BM) as u32, (n / BN) as u32, 1], [128, 1, 1]), args, a, b)
}

/// Check `c` against a CPU GEMM.
pub fn check(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
            assert_eq!(c[i * n + j], expected, "C[{i},{j}] of {m}x{n}x{k}");
        }
    }
}

/// Simulate one problem size with an already-built kernel.
pub fn run(kernel: &Kernel, m: usize, n: usize, k: usize) -> SimReport {
    let (launch, mut args, a, b) = problem(m, n, k);
    let report = kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
    check(m, n, k, &a, &b, args[2].as_f32());
    report
}

#[allow(dead_code)]
fn main() {
    let kernel = build();
    for (m, n, k) in [(128, 128, 32), (256, 128, 96), (128, 256, 160), (256, 256, 256)] {
        let report = run(&kernel, m, n, k);
        println!("{m}x{n}x{k}: correct, {} mma.sync", report.mma_instructions);
    }
    if let Some(tc) = Toolchain::discover() {
        match kernel.compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => println!("PTX: {} lines, the same for every size", ptx.lines().count()),
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        }
    }
}

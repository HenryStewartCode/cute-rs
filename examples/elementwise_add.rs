//! Element-wise addition, following the CuTe DSL `elementwise_add` tutorial.
//!
//! Two kernels compute `C = A + B` on an `M x N` row-major matrix:
//!
//! 1. `naive`: one element per thread, plain indexing.
//! 2. `tiled`: each block owns a 16x128 tile built from a thread layout and a
//!    value layout (`make_layout_tv`). Every thread copies a 4x4 block with
//!    128-bit vector loads, adds in registers and stores the result.
//!    `M` is deliberately not a multiple of the tile height, so the last row
//!    of tiles is predicated with an identity tensor, as in the tutorial.
//!
//! Both kernels are run in the simulator and checked against the CPU; the
//! simulator's sector counts show the effect of vectorisation. If an
//! `mlir-opt` is installed, the tiled kernel is also compiled to PTX.
//!
//! Run with `cargo run --example elementwise_add`.

use cute::prelude::*;

const M: usize = 300;
const N: usize = 512;

pub fn naive() -> Kernel {
    Kernel::builder("elementwise_add_naive").block_dim([256, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::row_major((M, N)));
        let b = param_tensor::<f32>("b", Layout::row_major((M, N)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));

        let idx = block_idx().x * 256 + thread_idx().x;
        let (m, n) = (idx / N as i32, idx % N as i32);
        if_(idx.lt((M * N) as i32), || {
            c.store((m, n), a.load((m, n)) + b.load((m, n)));
        });
    })
}

pub fn tiled() -> Kernel {
    // 4 x 32 threads (row-major), each owning 4 x 4 values (row-major):
    // a 16 x 128 tile per block.
    let thr_layout = Layout::row_major((4, 32));
    let val_layout = Layout::row_major((4, 4));
    let (tiler_mn, _) = make_layout_tv(&thr_layout, &val_layout);

    Kernel::builder("elementwise_add_tiled").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::row_major((M, N)));
        let b = param_tensor::<f32>("b", Layout::row_major((M, N)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));
        let tidx = thread_idx().x;
        let bid = block_idx();

        // This block's tile of each tensor, plus the matching coordinates.
        let tile = |t: &Tensor<f32>| local_tile(t, &tiler_mn, (bid.x, bid.y));
        let (blk_a, blk_b, blk_c) = (tile(&a), tile(&b), tile(&c));
        let blk_coord = local_tile(&make_identity_tensor((M, N)), &tiler_mn, (bid.x, bid.y));

        // 128-bit loads and stores: four f32 per instruction.
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        let tiled_copy = make_tiled_copy_tv(&atom, &thr_layout, &val_layout);
        let thr = tiled_copy.get_slice(tidx);

        let thr_a = thr.partition_s(&blk_a);
        let thr_b = thr.partition_s(&blk_b);
        let thr_c = thr.partition_d(&blk_c);
        let thr_coord = thr.partition_s(&blk_coord);

        // Rows past M belong to the partial last tile and are skipped.
        let pred = make_predicate(&thr_coord, |crd| elem_less(crd, &(M, N).into()));

        let frg_a = make_fragment_like::<f32, _>(&thr_a);
        let frg_b = make_fragment_like::<f32, _>(&thr_b);
        copy_if(&tiled_copy, &pred, &thr_a, &frg_a);
        copy_if(&tiled_copy, &pred, &thr_b, &frg_b);

        let frg_c = make_fragment_like::<f32, _>(&thr_c);
        for i in 0..frg_c.size() {
            frg_c.store(i, frg_a.load(i) + frg_b.load(i));
        }
        copy_if(&tiled_copy, &pred, &frg_c, &thr_c);
    })
}

pub fn run(kernel: &Kernel, grid: [u32; 3], block: [u32; 3]) -> SimReport {
    let a: Vec<f32> = (0..M * N).map(|i| (i % 97) as f32 * 0.5).collect();
    let b: Vec<f32> = (0..M * N).map(|i| (i % 13) as f32 - 6.0).collect();
    let mut args = vec![KernelArg::f32(a.clone()), KernelArg::f32(b.clone()), KernelArg::f32(vec![0.0; M * N])];
    let report = kernel.simulate(LaunchConfig::new(grid, block), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let c = args[2].as_f32();
    for i in 0..M * N {
        assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
    }
    report
}

pub const NAIVE_GRID: [u32; 3] = [(M * N).div_ceil(256) as u32, 1, 1];
pub const TILED_GRID: [u32; 3] = [M.div_ceil(16) as u32, N.div_ceil(128) as u32, 1];

#[allow(dead_code)]
fn main() {
    let naive = naive();
    let report = run(&naive, NAIVE_GRID, [256, 1, 1]);
    println!("{}: correct\n{report}\n", naive.name());

    let tiled = tiled();
    let report = run(&tiled, TILED_GRID, [128, 1, 1]);
    println!("{}: correct\n{report}\n", tiled.name());

    match Toolchain::discover() {
        Some(tc) => match tiled.compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let vector_loads = ptx.lines().filter(|l| l.contains("ld.global.v")).count();
                println!("PTX for {}: {} lines, {vector_loads} vector loads", tiled.name(), ptx.lines().count());
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

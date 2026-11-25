//! The example kernels, simulated and checked against CPU references.
//!
//! The examples are compiled into this test as modules, so the kernels
//! tested here are exactly the ones in `examples/`.

#[allow(dead_code)]
#[path = "../examples/ampere_sgemm.rs"]
mod ampere_sgemm;
#[allow(dead_code)]
#[path = "../examples/async_pipeline.rs"]
mod async_pipeline;
// The first Blackwell tutorial GEMM (m10-port-fp16-gemm).
// The canonical Blackwell GEMM (m10-port-dense-gemm).
#[allow(dead_code)]
#[path = "../examples/blackwell_dense_gemm.rs"]
mod blackwell_dense_gemm;
#[allow(dead_code)]
#[path = "../examples/blackwell_fp16_gemm.rs"]
mod blackwell_fp16_gemm;
// The first Blackwell tutorial NVFP4 GEMM (m10-port-nvfp4-gemm).
#[allow(dead_code)]
#[path = "../examples/blackwell_nvfp4_gemm.rs"]
mod blackwell_nvfp4_gemm;
#[allow(dead_code)]
#[path = "../examples/block_reduce.rs"]
mod block_reduce;
#[allow(dead_code)]
#[path = "../examples/cooperative_launch.rs"]
mod cooperative_launch;
#[allow(dead_code)]
#[path = "../examples/cp_async_mbarrier_arrive.rs"]
mod cp_async_mbarrier_arrive;
#[allow(dead_code)]
#[path = "../examples/cta_norm.rs"]
mod cta_norm;
// The cute_ext foundation ports (m11-cute-ext-core).
#[allow(dead_code)]
#[path = "../examples/cute_ext_memcpy.rs"]
mod cute_ext_memcpy;
#[allow(dead_code)]
#[path = "../examples/cute_ext_sgemm.rs"]
mod cute_ext_sgemm;
#[path = "support/dlpack_alloc.rs"]
mod dlpack_alloc;
#[allow(dead_code)]
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
#[allow(dead_code)]
#[path = "../examples/elementwise_add_autotune.rs"]
mod elementwise_add_autotune;
#[allow(dead_code)]
#[path = "../examples/elementwise_apply.rs"]
mod elementwise_apply;
#[allow(dead_code)]
#[path = "../examples/fp4_x2_tensor.rs"]
mod fp4_x2_tensor;
#[allow(dead_code)]
#[path = "../examples/grouped_gemm.rs"]
mod grouped_gemm;
#[allow(dead_code)]
#[path = "../examples/pipeline_software_fill.rs"]
mod pipeline_software_fill;
#[allow(dead_code)]
#[path = "../examples/red.rs"]
mod red;
#[allow(dead_code)]
#[path = "../examples/redux_sync.rs"]
mod redux_sync;
#[allow(dead_code)]
#[path = "../examples/sgemm.rs"]
mod sgemm;
#[allow(dead_code)]
#[path = "../examples/single_warp_pipeline.rs"]
mod single_warp_pipeline;
#[allow(dead_code)]
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;
#[allow(dead_code)]
#[path = "../examples/tvm_ffi_add.rs"]
mod tvm_ffi_add;
#[allow(dead_code)]
#[path = "../examples/warp_named_barrier.rs"]
mod warp_named_barrier;
// The TMA tutorials (m7-port-tma-tutorials).
#[allow(dead_code)]
#[path = "../examples/tma_copy.rs"]
mod tma_copy;
#[allow(dead_code)]
#[path = "../examples/tma_transpose.rs"]
mod tma_transpose;

use cute::prelude::*;
use tensorop_gemm::GemmConfig;

#[test]
fn elementwise_add_naive() {
    let report = elementwise_add::run(&elementwise_add::naive(), elementwise_add::NAIVE_GRID, [256, 1, 1]);
    assert_eq!(report.global_efficiency(), 1.0);
}

#[test]
fn elementwise_add_tiled_with_partial_tiles() {
    let kernel = elementwise_add::tiled();
    let report = elementwise_add::run(&kernel, elementwise_add::TILED_GRID, [128, 1, 1]);
    // Each thread moves its 4x4 block with four 128-bit loads per input.
    let vector_loads =
        kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::Load { results, .. } if results.len() == 4));
    assert_eq!(vector_loads, 8);
    assert_eq!(report.global_efficiency(), 1.0);
}

/// `elementwise_add.py`'s kernel on `mark_layout_dynamic()` tensors: one
/// build for every size, partial tiles in both modes and a single partial
/// tile included, with one-element copies (nothing is known of the runtime
/// row stride's alignment).
#[test]
fn elementwise_add_dynamic_one_kernel_for_every_size() {
    let kernel = elementwise_add::dynamic();
    assert_eq!(
        kernel.spec().0.layout_string(),
        "(?,?):(?{i64},1)",
        "the host tensors' layouts are dynamic but for the unit stride"
    );
    for (m, n) in [(300, 512), (3, 12), (17, 130), (64, 128), (1, 5), (33, 257)] {
        let (report, _) = elementwise_add::run_dynamic(&kernel, m, n);
        assert_eq!(report.blocks as usize, m.div_ceil(16) * n.div_ceil(128), "{m}x{n}");
    }
    let vector_loads =
        kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::Load { results, .. } if results.len() > 1));
    assert_eq!(vector_loads, 0);
}

/// The layouts `ampere_sgemm` is tested with: every major mode of A, B and
/// C, with M-major A and N-major B both at a divisibility of 16 (4-wide
/// `cp.async`) and of 4 (scalar copies, as `max_alignment % 16` is not 0).
const SGEMM_MAJORS: [ampere_sgemm::Majors; 4] = [
    ampere_sgemm::Majors::new('m', 'n', 'n', 16),
    ampere_sgemm::Majors::new('m', 'n', 'm', 4),
    ampere_sgemm::Majors::new('k', 'k', 'n', 1),
    ampere_sgemm::Majors::new('k', 'n', 'm', 4),
];

/// `sgemm.py` with runtime M, N and K: each layout's kernel is built once
/// and run at sizes that are tile multiples and not, with K below, at and
/// above bK (a single K tile, fewer tiles than stages, and more), bit for
/// bit against an FMA chain over k in order.
#[test]
fn ampere_sgemm_runtime_shapes() {
    for majors in SGEMM_MAJORS {
        let kernel = ampere_sgemm::build_for(majors);
        let d = majors.divisibility;
        for (m, n, k) in [(128, 128, 8), (128, 256, 24), (144, 112, 5), (48, 272, 37), (16, 16, 1)] {
            let (m, n) = (usize::next_multiple_of(m, d), usize::next_multiple_of(n, d));
            ampere_sgemm::run(&kernel, majors, m, n, k);
        }
    }
}

/// `max_alignment` of A's and B's mode 0 picks the copy width: 16-byte
/// `cp.async` at a divisibility of 16, 4-byte otherwise and for K-major
/// operands.
#[test]
fn ampere_sgemm_copy_width_follows_max_alignment() {
    for majors in SGEMM_MAJORS {
        let kernel = ampere_sgemm::build_for(majors);
        let cp_async =
            |n: u32| kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::CpAsync { elems, .. } if *elems == n));
        let (wide, narrow) = (cp_async(4), cp_async(1));
        let vectorized = majors.divisibility == 16;
        assert_eq!((wide > 0, narrow > 0), (vectorized, !vectorized), "{majors:?}: {wide} wide, {narrow} narrow");
    }
}

/// Every operation of the example's test, checked bit for bit against a
/// Float32 reference: 128-bit loads and stores (16 per input and 16 for
/// the output per thread), so every global sector is used.
#[test]
fn elementwise_apply_ops() {
    use elementwise_apply::ApplyOp;
    let vectors = |k: &Kernel, lanes: usize| {
        let load = k.ir().count_ops(|op| matches!(op, cute::ir::Op::Load { results, .. } if results.len() == lanes));
        let store = k.ir().count_ops(|op| matches!(op, cute::ir::Op::Store { values, .. } if values.len() == lanes));
        (load, store)
    };
    for op in ApplyOp::F32 {
        let report = elementwise_apply::run_f32(op);
        assert_eq!(report.global_efficiency(), 1.0, "{op:?}");
        assert_eq!(vectors(&elementwise_apply::f32_kernel(op).kernel, 4), (16 * op.inputs(), 16), "{op:?}");
    }
    for op in ApplyOp::HALF {
        assert_eq!(elementwise_apply::run_f16(op).global_efficiency(), 1.0, "{op:?}");
        assert_eq!(elementwise_apply::run_bf16(op).global_efficiency(), 1.0, "{op:?}");
        assert_eq!(vectors(&elementwise_apply::f16_kernel(op).kernel, 8), (32, 16), "{op:?}");
    }
}

/// LayerNorm and RMSNorm on each type, a row a whole number of chunks and
/// one not (the predicates matter), within the f64 reference's tolerance.
#[test]
fn cta_norm_configs() {
    for cfg in cta_norm::CONFIGS {
        cta_norm::run_f16(cfg);
        cta_norm::run_bf16(cfg);
        cta_norm::run_f32(cfg);
    }
}

#[test]
fn sgemm() {
    let report = sgemm::run(128, 128, 32);
    assert_eq!(report.blocks, 1);
    assert_eq!(report.bank_conflict_factor(), 1.0);
}

#[test]
fn sgemm_multiple_blocks() {
    sgemm::run(256, 128, 16);
}

#[test]
fn single_warp_pipeline() {
    let report = single_warp_pipeline::run();
    assert_eq!(report.mma_instructions, 128 / 16);
}

#[test]
fn tensorop_gemm_single_tile() {
    let report = tensorop_gemm::run(GemmConfig::new(128, 128, 64));
    // 4 warps x (4 x 8 MMAs per 16-deep k-block) x 4 k-blocks.
    assert_eq!(report.mma_instructions, 4 * 32 * 4);
    assert_eq!(report.bank_conflict_factor(), 1.0);
}

#[test]
fn tensorop_gemm_grid_and_short_pipeline() {
    // Two blocks along M, and a K loop of exactly `stages` tiles.
    tensorop_gemm::run(GemmConfig::new(256, 128, 96));
}

#[test]
fn tensorop_gemm_bf16() {
    let report = tensorop_gemm::run(GemmConfig { ab: cute::DType::BF16, ..GemmConfig::new(256, 128, 96) });
    assert_eq!(report.mma_instructions, 2 * 4 * 32 * 6);
    assert_eq!(report.bank_conflict_factor(), 1.0);
}

/// The swizzle on the shared-memory pointers, on bytes (`Swizzle<2,4,3>`,
/// CuTe DSL's values) gives what the composed layout's `Swizzle<2,3,3>` on
/// f16 elements gives: the same results (`run` checks them exactly), the
/// same traffic and the same bank conflicts.
#[test]
fn tensorop_gemm_pointer_swizzle() {
    let cfg = GemmConfig::new(128, 128, 64);
    let (composed, pointer) = (tensorop_gemm::run(cfg), tensorop_gemm::run(GemmConfig { ptr_swizzle: true, ..cfg }));
    assert_eq!(pointer.bank_conflict_factor(), 1.0);
    assert_eq!(
        (pointer.shared_wavefronts, pointer.mma_instructions, pointer.warp_instructions),
        (composed.shared_wavefronts, composed.mma_instructions, composed.warp_instructions)
    );
}

/// The A and B stages in one shared-memory struct (`cute_struct!`, its
/// members 1024-byte aligned): the same exact results, shared-memory traffic,
/// MMAs and bank conflicts as the separate buffers, f16 and FP8, swizzled or not.
#[test]
fn tensorop_gemm_struct_storage() {
    for cfg in [GemmConfig::new(128, 128, 64), GemmConfig::fp8(128, 128, 128, cute::DType::F8E4M3FN)] {
        for swizzle in [true, false] {
            let cfg = GemmConfig { swizzle, ..cfg };
            let (buffers, storage) =
                (tensorop_gemm::run(cfg), tensorop_gemm::run(GemmConfig { smem_struct: true, ..cfg }));
            // (B's tensor starts past A's in the one buffer, which costs the
            // simulator's count some address additions.)
            assert_eq!(
                (storage.shared_wavefronts, storage.mma_instructions),
                (buffers.shared_wavefronts, buffers.mma_instructions)
            );
            assert_eq!(storage.bank_conflict_factor(), buffers.bank_conflict_factor());
        }
    }
    let kernel = tensorop_gemm::build(GemmConfig { smem_struct: true, ..GemmConfig::new(128, 128, 64) });
    let dump = kernel.dump();
    assert!(dump.contains("shared smem0: 49152 x u8 (align 1024)"), "{dump}");
    assert!(dump.contains("struct GemmStorage { s_a:24576:0, s_b:24576:24576 }"), "{dump}");
}

#[test]
fn tensorop_gemm_two_stages_unswizzled() {
    let report = tensorop_gemm::run(GemmConfig { stages: 2, swizzle: false, ..GemmConfig::new(128, 128, 64) });
    assert!(report.bank_conflict_factor() > 2.0, "unswizzled K-major tiles should conflict: {report}");
}

/// `block_smem_reduce.py`'s kernel for every dtype, operation and lane width
/// it takes, on its data.
#[test]
fn block_reduce_every_configuration() {
    use block_reduce::ReduceOp;
    for op in ReduceOp::ALL {
        for items in [2, 4, 8] {
            block_reduce::run::<i32>(2, items, op);
            block_reduce::run::<f32>(2, items, op);
            block_reduce::run::<F16>(2, items, op);
            block_reduce::run::<BF16>(2, items, op);
        }
    }
    let report =
        block_reduce::run::<i32>(block_reduce::DEFAULT_ROW_TILES, block_reduce::DEFAULT_ITEMS_PER_LANE, ReduceOp::Add);
    assert_eq!(report.blocks, block_reduce::DEFAULT_ROW_TILES as u64);
}

/// The block reduction on random Float32 data, whose sums round: the
/// result is one of the orders the warps' reductions can land in.
#[test]
fn block_reduce_random_floats() {
    use block_reduce::ReduceOp;
    let mut state = 7u64;
    let rows = 2 * block_reduce::ROWS_PER_CTA;
    let src: Vec<f32> = (0..rows * block_reduce::THREADS * 4)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 20) as f32 - 8.0
        })
        .collect();
    for op in ReduceOp::ALL {
        let mut args = vec![KernelArg::f32(src.clone()), KernelArg::f32(vec![0.0; rows])];
        block_reduce::build::<f32>(2, 4, op).simulate(block_reduce::launch(2), &mut args).unwrap();
        block_reduce::check::<f32>(&args, op);
    }
}

/// `redux_sync.py`'s kernels 1 to 3 on its data and on random data.
#[test]
fn redux_sync_kernels() {
    redux_sync::run_integer(&redux_sync::int_inputs());
    redux_sync::run_float(&redux_sync::float_inputs());
    let mut state = 3u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };
    for _ in 0..4 {
        let ints: Vec<i32> = (0..32).map(|_| next() as i32).collect();
        redux_sync::run_integer(&ints);
        let floats: Vec<f32> = (0..32).map(|_| (next() >> 8) as f32 / (1 << 20) as f32 - 8.0).collect();
        redux_sync::run_float(&floats);
    }
}

/// `red.py` at its default size and with a partial last block.
#[test]
fn red_example() {
    red::run(red::DEFAULT_N, red::DEFAULT_THREADS);
    red::run(1000, 256);
}

/// Seeds for the seeded-schedule runs: `CUTE_SIM_SEEDS` of them (20 by
/// default).
fn seeds() -> Vec<u64> {
    let n = std::env::var("CUTE_SIM_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    (0..n).collect()
}

/// The examples at small sizes under seeded random schedules and random
/// completion of asynchronous copies: every run gives the round-robin
/// schedule's results (which the tests above hold to Rust references).
#[test]
fn examples_under_seeded_schedules() {
    let seeds = seeds();
    let run = |name: &str, kernel: &Kernel, cfg: LaunchConfig, mut args: Vec<KernelArg>| {
        kernel
            .simulate_schedules(cfg, &mut args, &SimOptions::default(), &seeds)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    };
    let f = |n: usize, m: usize| KernelArg::f32((0..n).map(|i| (i % m) as f32 - 3.0).collect());
    let h = |n: usize| KernelArg::f16_from_f32(&(0..n).map(|i| (i % 7) as f32 - 3.0).collect::<Vec<_>>());
    run(
        "elementwise_add (naive)",
        &elementwise_add::naive(),
        LaunchConfig::new(elementwise_add::NAIVE_GRID, [256, 1, 1]),
        vec![f(300 * 512, 97), f(300 * 512, 13), f(300 * 512, 1)],
    );
    run(
        "sgemm",
        &sgemm::build(128, 128, 16),
        LaunchConfig::new([1, 1, 1], [256, 1, 1]),
        vec![f(128 * 16, 11), f(128 * 16, 13), f(128 * 128, 5), KernelArg::ScalarF32(1.5), KernelArg::ScalarF32(-0.5)],
    );
    let majors = ampere_sgemm::Majors::new('m', 'n', 'n', 16);
    let (host, _, _) = ampere_sgemm::problem(majors, 128, 32, 20, 3);
    let sgemm = ampere_sgemm::build_for(majors);
    run("ampere_sgemm", &sgemm, ampere_sgemm::launch(128, 32), sgemm.pack(&host).unwrap());
    run(
        "elementwise_add (dynamic)",
        &elementwise_add::dynamic(),
        LaunchConfig::new(elementwise_add::dynamic_grid(20, 130), [128, 1, 1]),
        elementwise_add::dynamic()
            .pack(&elementwise_add::dynamic_host(&vec![1.0; 20 * 130], &vec![2.0; 20 * 130], 20, 130))
            .unwrap(),
    );
    run(
        "single_warp_pipeline",
        &single_warp_pipeline::build(),
        LaunchConfig::new([1, 1, 1], [32, 1, 1]),
        vec![h(16 * 128), h(128 * 8), f(16 * 8, 1)],
    );
    let gemm = GemmConfig::new(128, 128, 64);
    run(
        "tensorop_gemm",
        &tensorop_gemm::build(gemm),
        LaunchConfig::new([1, 1, 1], [128, 1, 1]),
        vec![h(128 * 64), h(128 * 64), f(128 * 128, 1)],
    );
    run(
        "block_reduce",
        &block_reduce::build::<i32>(2, 4, block_reduce::ReduceOp::Add),
        block_reduce::launch(2),
        block_reduce::example_args::<i32>(2, 4),
    );
    run("red", &red::build::<i32>(1000, 256), red::launch(1000, 256), red::s32_args(1000));
    run(
        "redux_sync",
        &redux_sync::integer_reductions(),
        redux_sync::launch(),
        vec![
            KernelArg::buffer(&redux_sync::int_inputs()),
            KernelArg::buffer(&[0i32; 8]),
            KernelArg::buffer(&[0i32; 8]),
        ],
    );
}

/// `warp_named_barrier.py` at both pipeline depths and two tile counts, for
/// SM89 (`mbarrier_test_wait` spins) and SM90 (`mbarrier_try_wait` and the
/// init fence), under every schedule and seeds.
#[test]
fn warp_named_barrier_example() {
    for stages in [2, 4] {
        for tiles in [warp_named_barrier::DEFAULT_NUM_TILES, 2 * warp_named_barrier::DEFAULT_NUM_TILES] {
            for arch in [Arch::Sm89, Arch::Sm90] {
                warp_named_barrier::run(stages, tiles, arch);
            }
        }
    }
}

/// `pipeline_software_fill.py`, likewise.
#[test]
fn pipeline_software_fill_example() {
    for stages in [2, 4] {
        for tiles in [pipeline_software_fill::DEFAULT_NUM_TILES, 2 * pipeline_software_fill::DEFAULT_NUM_TILES] {
            for arch in [Arch::Sm89, Arch::Sm90] {
                pipeline_software_fill::run(stages, tiles, arch);
            }
        }
    }
}

/// `cp_async_mbarrier_arrive_cta.py` (SM89 and SM90) and
/// `cp_async_mbarrier_arrive_cluster.py` (SM90, a cluster of 2).
#[test]
fn cp_async_mbarrier_arrive_example() {
    cp_async_mbarrier_arrive::run();
}

/// `cooperative_launch.py`'s grid barrier as a cooperative launch (the whole grid resident,
/// with no `resident_clusters` option), and the hang report, with the hint to launch
/// cooperatively, without the attribute.
#[test]
fn cooperative_launch_example() {
    let report = cooperative_launch::run();
    assert_eq!(report.blocks, cooperative_launch::BLOCKS as u64);
    let opts = SimOptions { max_idle_steps: 1 << 14, ..SimOptions::default() };
    let e = cooperative_launch::build()
        .simulate_with(cooperative_launch::launch_not_cooperative(), &mut cooperative_launch::args(), &opts)
        .unwrap_err();
    assert!(e.message.contains("LaunchConfig::with_cooperative(true)"), "{e}");
    assert!(e.message.contains("(a hang)") && e.message.contains("polling barrier[0] with an ordered load"), "{e}");
    assert!(
        e.message.contains("7 cluster(s) of the grid have not started") && e.message.contains("ResidentClusters::Grid"),
        "{e}"
    );
}

/// `tensorop_gemm.py`'s FP8 path: Float8E4M3FN inputs (built from f32 with `fp8_from_f32`),
/// `MmaFP8Op` 16x8x32, the 128x128x64 tile, `LdMatrix8x16x8bOp(False, 4)` for A and B and the
/// example's `Swizzle<2,4,4>`, exact on small integers; and Float8E5M2 inputs with FP8 C (the
/// exact sums rounded to E5M2 and E4M3).
#[test]
fn tensorop_gemm_fp8() {
    let report = tensorop_gemm::run(GemmConfig::fp8(128, 128, 128, cute::DType::F8E4M3FN));
    // 4 warps x (4 x 8 m16n8k32 MMAs per 32-deep k-block) x 4 k-blocks.
    assert_eq!(report.mma_instructions, 4 * 32 * 4);
    eprintln!("FP8 tensorop_gemm: {:.2}x serialised by bank conflicts", report.bank_conflict_factor());
    let e5m2 = GemmConfig { c: cute::DType::F8E5M2, ..GemmConfig::fp8(256, 128, 192, cute::DType::F8E5M2) };
    tensorop_gemm::run(e5m2);
    tensorop_gemm::run(GemmConfig { c: cute::DType::F8E4M3FN, ..GemmConfig::fp8(128, 128, 64, cute::DType::F8E4M3FN) });
}

/// `redux_sync.py`'s kernel 4, per-row abs-max and FP8 quantisation, with `redux.sync.max.abs.f32`
/// (sm_100a) and with the shuffle tree (SM89), on the example's shape and on more rows, bit for
/// bit against plain Rust and within the example's tolerances.
#[test]
fn redux_sync_absmax_quantize() {
    for (rows, seed) in [(redux_sync::DEFAULT_NUM_ROWS, 0), (24, 7)] {
        let inp = redux_sync::quantize_inputs(rows, seed);
        for redux in [true, false] {
            redux_sync::run_quantize(&inp, redux);
        }
    }
}

/// `fp4_x2_tensor.py`'s recast: the same `float4_e2m1fn_x2` bytes as a Float4E2M1FN tensor and
/// as a Uint8 tensor recast to it decode to the same values, with integer operations and with
/// `cvt.rn.f16x2.e2m1x2`, on the example's random 16 x 16 bytes and on every byte value.
#[test]
fn fp4_x2_tensor_recast() {
    let every: Vec<u8> = (0..=255).collect();
    for decode in [fp4_x2_tensor::Decode::Integer, fp4_x2_tensor::Decode::Cvt] {
        fp4_x2_tensor::run(decode, &fp4_x2_tensor::random_bytes(1));
        fp4_x2_tensor::run(decode, &every);
    }
}

/// The kernels of the `tvm_ffi_add` shared library (CuTe DSL's `jit_and_use_in_torch.py` and
/// `error_reporting.py`) on host DLPack tensors: `b = a + 1`, with a dynamic extent and the
/// static `(10,)`, which refuses 5 elements.
#[test]
fn tvm_ffi_add_one() {
    use cute::external::dlpack::{DLDevice, device_type};
    use dlpack_alloc::{Owned, f32_type, managed};
    let cpu = DLDevice { device_type: device_type::KDL_CPU, device_id: 0 };
    let tensor = |v: &[f32]| managed(Owned::f32(v), f32_type(), &[v.len() as i64], None, 0, 0, cpu, true);
    let cfg = LaunchConfig::new([1, 1, 1], [1, 1, 1]);
    for (len, n) in [(None, 10), (None, 37), (Some(10), 10)] {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let ((ta, _), (tb, cb)) = (tensor(&a), tensor(&vec![0.0; n]));
        let mut host = (ta, tb);
        tvm_ffi_add::add_one_kernel(len).simulate(cfg, &mut host).unwrap_or_else(|e| panic!("{e}"));
        drop(host);
        assert_eq!(cb.data_f32(), a.iter().map(|x| x + 1.0).collect::<Vec<_>>(), "{len:?} {n}");
    }
    let ((ta, _), (tb, _)) = (tensor(&[1.0; 5]), tensor(&[0.0; 10]));
    let mut host = (ta, tb);
    let e = tvm_ffi_add::add_one_kernel(Some(10)).simulate(cfg, &mut host).unwrap_err();
    assert!(e.to_string().contains("arg0"), "{e}");
}

/// `elementwise_add_autotune`: every candidate tile on the simulator (correct at the edges;
/// `copy_bits = 1` cannot be built, and 128-bit copies cannot serve rows of 6), then an
/// autotuner over the simulator's warp instructions as the measure, which picks the widest
/// copies where they fit and tunes again when `N` changes.
#[test]
fn elementwise_add_autotune_candidates() {
    use elementwise_add_autotune::{CANDIDATES, TileConfig, simulate, spec};
    for (m, n) in [(3, 12), (37, 64), (5, 6)] {
        for cfg in CANDIDATES {
            match (simulate(cfg, m, n), cfg.copy_bits, n % cfg.vector().max(1)) {
                (Err(e), 1, _) => assert!(e.contains("cannot be built"), "{e}"),
                (Err(e), _, r) if r != 0 => assert!(e.contains("not a multiple"), "{e}"),
                (Ok(r), _, _) => assert_eq!(r.blocks, cfg.grid(m, n)[0] as u64),
                (Err(e), _, _) => panic!("{cfg:?} {m}x{n}: {e}"),
            }
        }
    }
    let mut tuner = cute::testing::Autotuner::new(CANDIDATES).update_on_change(&["M", "N"]);
    let mut pick = |m: usize, n: usize| -> TileConfig {
        let key = cute::testing::TuneKey::new(&spec(1).cache_key()).part("M", m).part("N", n);
        tuner
            .get_or_tune_by(&key, |&cfg| Ok(simulate(cfg, m, n)?.warp_instructions as f64))
            .unwrap_or_else(|e| panic!("{e}"))
            .config
    };
    assert_eq!(pick(37, 64).copy_bits, 128);
    assert!(matches!(pick(5, 6).copy_bits, 32 | 64));
    assert_eq!(pick(37, 64).copy_bits, 128);
    assert_eq!(tuner.tunings(), 2);
}

/// `async_pipeline.ipynb`'s kernels (the synced hand-off, the one-stage and
/// five-stage `PipelineAsync`s, and the try-token variant), built for SM89
/// and SM90, under every schedule and eight seeds: the notebook's outputs,
/// `res = [0, ..., 7]` and `staging = [5, 6, 7, 3, 4]`.
#[test]
fn async_pipeline_example() {
    async_pipeline::run();
}

/// The persistent grouped GEMM (`mma.sync`, direct and cached problem
/// shapes) on one, six and 37 groups, and the MoE layer driven by the
/// contiguous group search, against CPU references.
#[test]
fn grouped_gemm_example() {
    grouped_gemm::run();
}

/// CuTe DSL's TMA tutorial `tma_v0.py` (`examples/tma_copy.rs`): the tutorial's default
/// problem, a square one and one with partial edge tiles in both modes (the loads
/// zero-fill past the edge and the stores clip there), each copied bit for bit under every
/// built-in schedule; one TMA load of one 128x128 box per CTA.
#[test]
fn tma_copy_example() {
    for (m, n) in [(tma_copy::DEFAULT_M, tma_copy::DEFAULT_N), (256, 256), (200, 144), (72, 48)] {
        let report = tma_copy::run(m, n);
        let ctas = m.div_ceil(128) * n.div_ceil(128);
        assert_eq!(report.tma_instructions as usize, ctas, "{m}x{n}: {report}");
        assert_eq!(report.tma_bytes as usize, ctas * 128 * 128 * 2, "{m}x{n}: {report}");
    }
}

/// CuTe DSL's TMA tutorial `tma_v1.py` (`examples/tma_transpose.rs`): warp-specialised
/// transposes of the tutorial's default problem, a rectangular one and one with partial
/// edge tiles, bit for bit under every built-in schedule and four seeded ones; each CTA's
/// load is two boxes (the 128-byte swizzle spans 64 Float16 columns).
#[test]
fn tma_transpose_example() {
    for (m, n) in [(tma_transpose::DEFAULT_M, tma_transpose::DEFAULT_N), (256, 128), (144, 208)] {
        let report = tma_transpose::run(m, n, &[0, 1, 2, 3]);
        let ctas = m.div_ceil(128) * n.div_ceil(128);
        assert_eq!(report.tma_instructions as usize, 2 * ctas, "{m}x{n}: {report}");
    }
}

/// CuTe DSL's Blackwell tutorial `fp16_gemm_0.py` (`examples/blackwell_fp16_gemm.rs`) on
/// a 256x512x256 problem (2x2 CTAs, 4 k-tiles each, so the 4-stage ring wraps once per
/// CTA): C bit for bit against the CPU; 4 MMAs of 128x256x16 per k-tile; the global
/// reads are exactly the TMA loads of A and B (each CTA reads its 128 rows of A and 256
/// of B once).
#[test]
fn blackwell_fp16_gemm_0_example() {
    let cfg = blackwell_fp16_gemm::FP16_GEMM_0;
    let (m, n, k) = (256, 512, 256);
    let report = blackwell_fp16_gemm::run(&cfg, m, n, k);
    let ctas = (m / 128) * (n / 256);
    let k_tiles = k / 64;
    assert_eq!(report.blocks as usize, ctas, "{report}");
    assert_eq!(report.tcgen05_mma_instructions as usize, ctas * k_tiles * 4, "{report}");
    assert_eq!(report.tma_instructions as usize, ctas * k_tiles * 2, "{report}");
    let ab_bytes = ctas * k * (128 + 256) * 2;
    assert_eq!(report.tma_bytes as usize, ab_bytes, "{report}");
    assert_eq!(report.global_bytes_read as usize, ab_bytes, "{report}");
    assert_eq!(report.global_bytes_written as usize, m * n * 2, "{report}");
}

/// CuTe DSL's Blackwell tutorial `nvfp4_gemm_0.py` (`examples/blackwell_nvfp4_gemm.rs`):
/// UE4M3 scale factors on a 256x512x512 problem (2x2 CTAs, 2 k-tiles each), UE8M0 ones on
/// one CTA with 5 k-tiles (the 4-stage ring wraps). C bit for bit against a CPU reference
/// that decodes the E2M1 and scale-factor codes itself, with random factors per row and
/// vector, so it holds only if the TMA, the SMEM and TMEM scale-factor layouts and the S2T
/// copies put every factor where the simulator's block-scaled MMA (PTX-derived) reads it.
/// Per k-tile: 4 TMA loads (A, B, SFA, SFB) and 4 MMAs of 128x256x64.
#[test]
fn blackwell_nvfp4_gemm_0_example() {
    for (cfg, (m, n, k)) in [
        (blackwell_nvfp4_gemm::NVFP4_GEMM_0, (256, 512, 512)),
        (blackwell_nvfp4_gemm::NVFP4_GEMM_0_UE8M0, (128, 256, 1280)),
    ] {
        let report = blackwell_nvfp4_gemm::run(&cfg, m, n, k);
        let ctas = (m / 128) * (n / 256);
        let k_tiles = k / 256;
        assert_eq!(report.blocks as usize, ctas, "{report}");
        assert_eq!(report.tcgen05_mma_instructions as usize, ctas * k_tiles * 4, "{report}");
        assert_eq!(report.tma_instructions as usize, ctas * k_tiles * 4, "{report}");
        // A and B packed FP4, one scale factor per 16 elements of each.
        let tile_bytes = (128 + 256) * 256 / 2 + (128 + 256) * 256 / 16;
        assert_eq!(report.tma_bytes as usize, ctas * k_tiles * tile_bytes, "{report}");
        assert_eq!(report.global_bytes_written as usize, m * n * 2, "{report}");
    }
}

/// The sm_100a PTX of `nvfp4_gemm_0` (both scale-factor types): TMA loads, the S2T copies,
/// the block-scaled MMAs; assembled by `ptxas -arch=sm_100a` when it is on PATH.
#[test]
fn blackwell_nvfp4_gemm_0_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for cfg in [blackwell_nvfp4_gemm::NVFP4_GEMM_0, blackwell_nvfp4_gemm::NVFP4_GEMM_0_UE8M0] {
        let kernel = cfg.build();
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
        let ptx = compiled.ptx();
        for w in [
            "cp.async.bulk.tensor",
            "tcgen05.cp.cta_group::1.32x128b.warpx4",
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
            "tcgen05.commit.cta_group::1",
            "tcgen05.ld.sync.aligned.32x32b.x128",
        ] {
            assert!(ptx.contains(w), "{}: no `{w}` in the PTX", cfg.name);
        }
        let ptxas = std::env::var_os("PATH")
            .and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
        let Some(ptxas) = ptxas else { return };
        let base = std::env::temp_dir().join(format!("cute-{}-{}", cfg.name, std::process::id()));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{}: ptxas: {}{}",
            cfg.name,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{}: ptxas -arch=sm_100a ok", cfg.name);
    }
}

/// The sm_100a PTX of `fp16_gemm_0`: TMA loads, tcgen05 MMAs, commits and TMEM loads;
/// assembled by `ptxas -arch=sm_100a` when it is on PATH.
#[test]
fn blackwell_fp16_gemm_0_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let kernel = blackwell_fp16_gemm::FP16_GEMM_0.build();
    let compiled =
        kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
    let ptx = compiled.ptx();
    for w in [
        "cp.async.bulk.tensor",
        "tcgen05.mma.cta_group::1.kind::f16",
        "tcgen05.commit.cta_group::1",
        "tcgen05.ld.sync.aligned.32x32b.x64",
        "tcgen05.alloc.cta_group::1",
        "tcgen05.dealloc.cta_group::1",
    ] {
        assert!(ptx.contains(w), "no `{w}` in the PTX");
    }
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let Some(ptxas) = ptxas else { return };
    let base = std::env::temp_dir().join(format!("cute-blackwell-fp16-gemm-{}", std::process::id()));
    let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
    std::fs::write(&src, ptx).unwrap();
    let out = std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&cubin);
    assert!(
        out.status.success(),
        "ptxas: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------------------
// The canonical Blackwell GEMM, `dense_gemm.py` (m10-port-dense-gemm).
// ---------------------------------------------------------------------------------------

use blackwell_dense_gemm::{
    DENSE_GEMM_1CTA, DENSE_GEMM_2CTA, DENSE_GEMM_2CTA_MULTICAST, DenseGemm, run as run_dense,
    run_with as run_dense_with,
};

/// The three configurations (1 CTA, a 2x1 cluster of a CTA pair, a 2x2 cluster of pairs with
/// multicast) with Float16 and Float32 C on problems whose M, N and K are not tile multiples
/// (TMA zero-fills the loads and clips the stores), compared bit for bit with the CPU, and
/// the MMA count: one `m x n` tile per CTA (pair), four MMAs per k-tile on each issuing CTA.
#[test]
fn blackwell_dense_gemm_ragged_problems() {
    for (cfg, (m, n, k)) in [
        (DENSE_GEMM_1CTA, (200usize, 136usize, 200usize)),
        (DENSE_GEMM_2CTA, (264, 136, 72)),
        (DENSE_GEMM_2CTA_MULTICAST, (264, 264, 136)),
    ] {
        // The grid is rounded up to whole clusters; its extra tiles run (and store nothing).
        let (cx, cy) = cfg.cluster_shape_mn;
        let v = if cfg.use_2cta_instrs { 2 } else { 1 };
        let tiles = (m.div_ceil(cfg.mma_tiler_mn.0 / v).div_ceil(cx) * cx / v)
            * (n.div_ceil(cfg.mma_tiler_mn.1).div_ceil(cy) * cy);
        let k_tiles = k.div_ceil(64);
        let report = run_dense::<F16>(&cfg, (m, n, k));
        assert_eq!(report.tcgen05_mma_instructions as usize, tiles * k_tiles * 4, "{}: {report}", cfg.name);
        assert!(report.tma_stores > 0, "{}: {report}", cfg.name);
        run_dense::<f32>(&cfg, (m, n, k));
    }
}

/// An M-major Float16 C goes through `stmatrix.trans` and a transposed TMA store.
#[test]
fn blackwell_dense_gemm_m_major_c() {
    let cfg =
        DenseGemm { c_layout: LayoutEnum::COL_MAJOR, name: "blackwell_dense_gemm_1cta_m_major", ..DENSE_GEMM_1CTA };
    run_dense::<F16>(&cfg, (136, 72, 72));
    let cfg =
        DenseGemm { c_layout: LayoutEnum::COL_MAJOR, name: "blackwell_dense_gemm_2cta_m_major", ..DENSE_GEMM_2CTA };
    run_dense::<F16>(&cfg, (264, 72, 64));
}

/// The race detector (on by default) stays clean and the results do not depend on the
/// schedule: every built-in schedule, early and late async completion, and random seeds.
#[test]
fn blackwell_dense_gemm_is_race_free_under_several_schedules() {
    for (cfg, mnk) in [
        (DENSE_GEMM_1CTA, (136, 72, 72)),
        (DENSE_GEMM_2CTA, (136, 72, 72)),
        (DENSE_GEMM_2CTA_MULTICAST, (264, 136, 72)),
    ] {
        let opts = SimOptions { detect_races: true, ..Default::default() };
        run_dense_with::<F16>(&cfg, mnk, &opts, &[3, 11]);
    }
}

/// The MLIR and the sm_100a PTX of each configuration (TMA loads with and without
/// multicast, tcgen05 MMAs and commits, TMEM loads, the TMA store), assembled by
/// `ptxas -arch=sm_100a` when it is on PATH.
#[test]
fn blackwell_dense_gemm_in_mlir_and_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for (cfg, group, multicast) in
        [(DENSE_GEMM_1CTA, 1, false), (DENSE_GEMM_2CTA, 2, true), (DENSE_GEMM_2CTA_MULTICAST, 2, true)]
    {
        let kernel = cfg.build::<F16>();
        let mlir = kernel.to_mlir();
        assert!(mlir.contains("tcgen05"), "{}: no tcgen05 in the MLIR", cfg.name);
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
        let ptx = compiled.ptx();
        let mut wanted = vec![
            "cp.async.bulk.tensor".to_string(),
            format!("tcgen05.mma.cta_group::{group}.kind::f16"),
            format!("tcgen05.commit.cta_group::{group}"),
            format!("tcgen05.alloc.cta_group::{group}"),
            format!("tcgen05.dealloc.cta_group::{group}"),
            "tcgen05.ld.sync.aligned".to_string(),
            "cp.async.bulk.tensor.2d.global.shared::cta.tile".to_string(),
            "cp.async.bulk.commit_group".to_string(),
        ];
        if multicast {
            wanted.push(".multicast::cluster".to_string());
        }
        for w in &wanted {
            assert!(ptx.contains(w.as_str()), "{}: no `{w}` in the PTX", cfg.name);
        }
        let ptxas = std::env::var_os("PATH")
            .and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
        let Some(ptxas) = ptxas else { continue };
        let base = std::env::temp_dir().join(format!("cute-blackwell-dense-gemm-{}-{}", std::process::id(), cfg.name));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{}: ptxas: {}{}",
            cfg.name,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// ---------------------------------------------------------------------------------------
// The Blackwell tutorial GEMMs 2, 3, 3.1 and 4 (m10-port-clc-clusters).
// ---------------------------------------------------------------------------------------

use blackwell_fp16_gemm::{
    FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4, FP16_GEMM_4_STATIC, Fp16GemmWsConfig, run_ws, run_ws_with,
};

/// The MMAs of `cfg` on an `m x n x k` problem when every tile of the problem rounded up to
/// whole clusters is computed once: four `256x256x16` MMAs per k-tile and CTA pair.
fn ws_mmas(cfg: &Fp16GemmWsConfig, m: usize, n: usize, k: usize) -> u64 {
    let (tm, tn) = cfg.num_ctas_mn(m, n);
    let (cx, cy, _) = cfg.cluster_shape_mnk;
    let pairs = (tm.div_ceil(cx) * cx * tn.div_ceil(cy) * cy / cfg.atom_ctas()) as u64;
    pairs * k.div_ceil(64) as u64 * 4
}

/// `fp16_gemm_2`: warp-specialized, one tile per CTA pair, a TMA-store epilogue; tile
/// multiples and ragged problems (TMA zero-fills the loads and clips the stores).
#[test]
fn blackwell_fp16_gemm_2_example() {
    for (m, n, k) in [(256, 256, 128), (264, 136, 72)] {
        let report = run_ws(&FP16_GEMM_2, (m, n, k), 0);
        assert_eq!(report.tcgen05_mma_instructions, ws_mmas(&FP16_GEMM_2, m, n, k), "{m}x{n}x{k}: {report}");
        assert!(report.tma_stores > 0, "{report}");
    }
}

/// `fp16_gemm_3`: the static persistent scheduler on grids smaller than the tile count (one
/// and three persistent clusters for four), equal to it, and larger (idle clusters), with
/// two accumulator stages; every tile computed once (the MMA count).
#[test]
fn blackwell_fp16_gemm_3_static_persistent() {
    let cfg = FP16_GEMM_3;
    let (m, n, k) = (512, 512, 128);
    for max_active_clusters in [1, 3, u32::MAX / 2] {
        let report = run_ws(&cfg, (m, n, k), max_active_clusters);
        assert_eq!(report.tcgen05_mma_instructions, ws_mmas(&cfg, m, n, k), "{max_active_clusters}: {report}");
    }
    // Six persistent clusters for four clusters of tiles.
    let report = run_ws_with(&cfg, (m, n, k), cfg.launch_grid([2, 1, 6]), &SimOptions::default());
    assert_eq!(report.tcgen05_mma_instructions, ws_mmas(&cfg, m, n, k), "{report}");
    // A ragged problem: three CTA tiles along M, padded to two pairs.
    let report = run_ws(&cfg, (264, 136, 72), 1);
    assert_eq!(report.tcgen05_mma_instructions, ws_mmas(&cfg, 264, 136, 72), "{report}");
}

/// `fp16_gemm_3_1`: the CLC scheduler with one, two and three resident clusters, several
/// launch-order seeds and schedules, and the whole grid resident: C exact, every tile
/// computed exactly once (the MMA count) and the cancelled clusters never started.
#[test]
fn blackwell_fp16_gemm_3_1_clc() {
    let cfg = FP16_GEMM_3_1;
    let (m, n, k) = (512, 512, 128);
    let grid = cfg.grid(m, n, 0);
    let grid_blocks = grid.iter().map(|&g| g as u64).product::<u64>();
    for (resident, seed, schedule) in [
        (ResidentClusters::Count(1), None, Schedule::RoundRobin),
        (ResidentClusters::Count(2), Some(7), Schedule::Random { seed: 3 }),
        (ResidentClusters::Count(3), Some(11), Schedule::RunToBlock),
        (ResidentClusters::Grid, None, Schedule::RoundRobin),
    ] {
        let opts = SimOptions { resident_clusters: resident, clc_launch_seed: seed, schedule, ..Default::default() };
        let report = run_ws_with(&cfg, (m, n, k), cfg.launch(m, n, 0), &opts);
        assert_eq!(report.tcgen05_mma_instructions, ws_mmas(&cfg, m, n, k), "{resident:?} {seed:?}: {report}");
        assert_eq!(report.blocks + report.clc_cancelled * 2, grid_blocks, "{resident:?} {seed:?}: {report}");
        if resident == ResidentClusters::Count(1) {
            assert_eq!(report.blocks, 2, "{report}");
        }
    }
}

/// `fp16_gemm_4`: preferred `(2, 4, 1)` and fallback `(2, 1, 1)` clusters with CLC under
/// every cluster mode of the simulator (all preferred, all fallback, and a mix of one
/// preferred and one split cluster), and with the static scheduler under the mix; the
/// kernel picks its arm from the cluster registers.
#[test]
fn blackwell_fp16_gemm_4_preferred_clusters() {
    // Two preferred clusters (the N tiles padded from 2 to 4), one k-tile. Mixed seeds 0 and
    // 6 split one of the two preferred clusters into fallback clusters.
    let (m, n, k) = (512, 512, 64);
    for (cfg, cluster_shapes, resident) in [
        (FP16_GEMM_4, ClusterShapes::Preferred, ResidentClusters::Count(1)),
        (FP16_GEMM_4, ClusterShapes::Fallback, ResidentClusters::Count(3)),
        (FP16_GEMM_4, ClusterShapes::Mixed { seed: 0 }, ResidentClusters::Grid),
        (FP16_GEMM_4_STATIC, ClusterShapes::Mixed { seed: 6 }, ResidentClusters::Count(2)),
    ] {
        let opts =
            SimOptions { cluster_shapes, resident_clusters: resident, clc_launch_seed: Some(5), ..Default::default() };
        let report = run_ws_with(&cfg, (m, n, k), cfg.launch(m, n, 1), &opts);
        let mmas = ws_mmas(&cfg, m, n, k);
        assert_eq!(report.tcgen05_mma_instructions, mmas, "{} {cluster_shapes:?}: {report}", cfg.name);
    }
}

/// The MLIR and the sm_100a PTX of each warp-specialized configuration, assembled by
/// `ptxas -arch=sm_100a` when it is on PATH: CLC queries in `fp16_gemm_3_1` and
/// `fp16_gemm_4`, and the cluster-shape register read of the mixed-cluster kernels.
#[test]
fn blackwell_fp16_gemm_2_to_4_in_mlir_and_ptx_for_sm100a() {
    let configs = [FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4, FP16_GEMM_4_STATIC];
    for cfg in configs {
        let mlir = cfg.build().to_mlir();
        let clc = mlir.contains("clusterlaunchcontrol.try_cancel");
        assert_eq!(clc, cfg.scheduling == blackwell_fp16_gemm::Scheduling::ClcDynamic, "{}: CLC in the MLIR", cfg.name);
    }
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    for cfg in configs {
        let kernel = cfg.build();
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
        let ptx = compiled.ptx();
        let mut wanted = vec![
            "cp.async.bulk.tensor",
            "tcgen05.mma.cta_group::2.kind::f16",
            "tcgen05.commit.cta_group::2",
            "tcgen05.alloc.cta_group::2",
            "tcgen05.ld.sync.aligned",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile",
        ];
        if cfg.scheduling == blackwell_fp16_gemm::Scheduling::ClcDynamic {
            wanted.push("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128");
            wanted.push("clusterlaunchcontrol.query_cancel.is_canceled.pred.b128");
        }
        if cfg.fallback_cluster_shape_mnk.is_some() {
            wanted.push("%cluster_nctaid");
        }
        for w in wanted {
            assert!(ptx.contains(w), "{}: no `{w}` in the PTX", cfg.name);
        }
        let Some(ptxas) = &ptxas else { continue };
        let base = std::env::temp_dir().join(format!("cute-{}-{}", cfg.name, std::process::id()));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{}: ptxas: {}{}",
            cfg.name,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// The launch constraints of preferred clusters: a fallback that does not divide the
/// preferred shape, a preferred shape that does not divide the grid, and unequal depths
/// are reported on the host, and the simulator refuses such a launch.
#[test]
fn blackwell_fp16_gemm_4_cluster_shape_diagnostics() {
    use cute::utils::check_mixed_cluster_shapes;
    let e = check_mixed_cluster_shapes([2, 4, 1], [2, 3, 1], None).unwrap_err();
    assert!(e.contains("does not divide the preferred"), "{e}");
    let e = check_mixed_cluster_shapes([2, 4, 1], [2, 1, 1], Some([2, 6, 1])).unwrap_err();
    assert!(e.contains("does not divide the grid"), "{e}");
    let e = check_mixed_cluster_shapes([2, 4, 2], [2, 1, 1], None).unwrap_err();
    assert!(e.contains("depths"), "{e}");
    check_mixed_cluster_shapes([2, 4, 1], [2, 1, 1], Some([4, 8, 1])).unwrap();
    let e = std::panic::catch_unwind(|| FP16_GEMM_4.launch_grid([2, 6, 1])).unwrap_err();
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(e.contains("does not divide the grid"), "{e}");
    let bad = Fp16GemmWsConfig { fallback_cluster_shape_mnk: Some((2, 3, 1)), ..FP16_GEMM_4 };
    let e = std::panic::catch_unwind(|| bad.build()).unwrap_err();
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(e.contains("does not divide the preferred"), "{e}");
    // The simulator checks the launch itself.
    let kernel = FP16_GEMM_4.build();
    let mut args = vec![KernelArg::scalar(512), KernelArg::scalar(512), KernelArg::scalar(64)];
    for len in [512 * 64, 512 * 64, 512 * 512] {
        args.push(KernelArg::buffer(&vec![F16::from_f32(0.0); len]));
    }
    let launch = LaunchConfig::new([4, 4, 1], [224, 1, 1]).with_fallback_cluster([2, 4, 1], [2, 3, 1]);
    let e = kernel.simulate(launch, &mut args).unwrap_err().to_string();
    assert!(e.contains("fallback"), "{e}");
}

/// `cute_ext/ampere/memcpy_simt_universal_copy.py`: one compiled kernel, predicated
/// automatically against runtime M, N and L, bit-exact at the example's ragged default
/// (136, 7, 9), at whole tiles and at sizes smaller than a tile.
#[test]
fn cute_ext_memcpy_predicates_automatically() {
    let kernel = cute_ext_memcpy::kernel();
    for mnl in [cute_ext_memcpy::DEFAULT_MNL, (256, 128, 1), (16, 3, 1), (40, 65, 3)] {
        cute_ext_memcpy::run(&kernel, mnl);
    }
}

/// `cute_ext/ampere/sgemm.py`: exact on small integers, within the Float32 summation bound
/// on random data, for one tile, the default problem and a batch of rectangular tiles.
#[test]
fn cute_ext_sgemm_matches_the_reference() {
    for mnkl in [(16, 8, 8, 1), cute_ext_sgemm::DEFAULT_MNKL, (48, 24, 40, 3)] {
        let kernel = cute_ext_sgemm::build(mnkl);
        cute_ext_sgemm::run(&kernel, mnkl, true);
        cute_ext_sgemm::run(&kernel, mnkl, false);
    }
}

/// The ports keep their vector widths in PTX: 128-bit global accesses for the memcpy,
/// `cp.async` and `ldmatrix` for the GEMM (when `mlir-opt` is installed).
#[test]
fn cute_ext_ports_keep_vector_widths_in_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = cute_ext_memcpy::kernel().kernel().compile_ptx(&tc, Arch::Sm80).unwrap();
    assert!(ptx.contains("ld.global.v4") && ptx.contains("st.global.v4"), "{ptx}");
    let ptx = cute_ext_sgemm::build((32, 16, 16, 1)).compile_ptx(&tc, Arch::Sm80).unwrap();
    assert!(ptx.contains("cp.async.ca.shared.global") && ptx.contains("ldmatrix") && ptx.contains("mma.sync"), "{ptx}");
}

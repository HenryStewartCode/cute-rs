//! Every type, operation and intrinsic: the simulator is held to a reference
//! written in Rust, and every kernel must compile to PTX (when `mlir-opt` is
//! available) so that each path of the MLIR emitter is exercised.

mod feature_kernels;

use cute::prelude::*;
use feature_kernels::{Case, all, inputs_with_len, sm86, sm89, sm90, sm90a, sm100, sm100a, sm120a};

/// Simulate a case under every built-in schedule, which must agree.
fn simulate(case: &Case, small: bool) -> Vec<KernelArg> {
    let mut args = inputs_with_len(&case.kernel, small, 256);
    case.kernel
        .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
    args
}

/// Every feature kernel, whatever its target.
fn every_case() -> impl Iterator<Item = Case> {
    all()
        .into_iter()
        .chain(sm90())
        .chain(sm86())
        .chain(sm89())
        .chain(sm100a())
        .chain(sm120a())
        .chain(sm90a())
        .chain(sm100())
}

#[test]
fn simulator_matches_rust_on_small_integers() {
    for case in every_case() {
        (case.check)(&simulate(&case, true));
    }
}

/// Every feature kernel on small inputs under `CUTE_SIM_SEEDS` (20 by
/// default) seeded random schedules with random completion of
/// asynchronous operations: each gives the round-robin results.
#[test]
fn simulator_results_do_not_depend_on_seeded_schedules() {
    let n = std::env::var("CUTE_SIM_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let seeds: Vec<u64> = (0..n).collect();
    for case in every_case() {
        let mut args = inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
    }
}

#[test]
fn simulator_matches_rust_on_random_data() {
    for case in every_case().filter(|c| !c.small_only) {
        (case.check)(&simulate(&case, false));
    }
}

#[test]
fn printf_output_is_collected() {
    let case = feature_kernels::printing();
    let mut args = feature_kernels::inputs(&case.kernel, true);
    let report = case.kernel.simulate(case.cfg, &mut args).unwrap();
    let x = args[0].to_vec::<f32>()[5];
    assert_eq!(report.printed, vec![format!("thread 5 has {x:.6}\n")]);
}

#[test]
fn every_feature_compiles_to_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let failures: Vec<String> = all()
        .into_iter()
        .map(|c| (c, Arch::Sm80))
        .chain(sm90().into_iter().map(|c| (c, Arch::Sm90)))
        .chain(sm86().into_iter().map(|c| (c, Arch::Sm86)))
        .chain(sm89().into_iter().map(|c| (c, Arch::Sm89)))
        .chain(sm100a().into_iter().map(|c| (c, Arch::Sm100a)))
        .chain(sm100().into_iter().map(|c| (c, Arch::Sm100a)))
        .chain(sm90a().into_iter().map(|c| (c, Arch::Sm90a)))
        .chain(sm120a().into_iter().map(|c| (c, Arch::Sm120a)))
        .filter_map(|(c, arch)| {
            let options = feature_kernels::options_for(&c.kernel, &tc).gpu_arch(arch);
            c.kernel.compile_with(&tc, &options).err().map(|e| format!("{}: {e}", c.name))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A loop that never ends is reported instead of hanging.
#[test]
fn runaway_while_loop_is_reported() {
    let kernel = Kernel::builder("spin").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let n = while_carry(Int32::from(0), |n| n.ge(0), |n| n & 1);
        out.store(thread_idx().x, n);
    });
    let mut args = vec![KernelArg::i32(vec![0; 32])];
    let opts = SimOptions { max_while_iterations: 1000, ..SimOptions::default() };
    let e = kernel.simulate_with(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args, &opts).unwrap_err();
    assert!(e.message.contains("did not terminate"), "{e}");
}

/// Shuffling with lanes of the member mask that never reach the shuffle
/// is a deadlock that names them.
#[test]
fn shuffle_from_inactive_lane_is_reported() {
    let kernel = Kernel::builder("bad_shuffle").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        if_(t.lt(4), || out.store(t, shuffle_sync(t, 20)));
    });
    let mut args = vec![KernelArg::i32(vec![0; 32])];
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("lanes 4..=31 took another path"), "{e}");
}

/// Reading a stack array before writing it is an error.
#[test]
fn uninitialised_stack_array_is_reported() {
    let kernel = Kernel::builder("bad_stack").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        let table = make_rmem_tensor::<i32>(Layout::col_major(4));
        out.store(t, table.load(t % 4));
    });
    let mut args = vec![KernelArg::i32(vec![0; 32])];
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("never written"), "{e}");
}

/// Inline PTX without a description cannot run in the simulator.
#[test]
fn inline_ptx_needs_a_simulation() {
    let kernel = Kernel::builder("opaque").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        let bits: Uint32 = inline_ptx("popc.b32 $0, $1;", &[t.into()]);
        out.store(t, bits);
    });
    let mut args = vec![KernelArg::buffer(&[0u32; 32])];
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("cannot be simulated"), "{e}");
}

/// Two blocks in one cluster run together and each has its own shared memory.
#[test]
fn clusters_run_blocks_together() {
    let kernel = Kernel::builder("cluster").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(4 * 32));
        let smem = make_smem_tensor::<i32>(Layout::col_major(32));
        let t = thread_idx().x;
        let id = block_idx().x;
        smem.store(t, id * 100 + block_in_cluster_idx().x * 10 + cluster_idx().x);
        sync_threads();
        out.store(id * 32 + t, smem.load(t) + block_in_cluster_dim().x * 1000);
    });
    let mut args = vec![KernelArg::i32(vec![0; 4 * 32])];
    let cfg = LaunchConfig::new([4, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    let report = kernel.simulate(cfg, &mut args).unwrap();
    assert_eq!(report.blocks, 4);
    assert_eq!(report.barriers, 4);
    let c = args[0].as_i32();
    assert_eq!([c[0], c[32], c[64], c[96]], [2000, 2110, 2201, 2311]);
}

fn bulk_copy_kernel() -> Kernel {
    Kernel::builder("bulk_copy").block_dim([64, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(16));
        let out = param_tensor::<u32>("out", Layout::col_major(16));
        let t = thread_idx().x;
        let barrier = alloc_smem::<u64>(1, 8);
        let data = alloc_smem::<u32>(16, 64);
        if_(t.eq(0), || mbarrier_init(&barrier, 1));
        sync_threads();
        if_(t.eq(0), || {
            mbarrier_arrive_and_expect_tx(&barrier, 64);
            cp_async_bulk(&src.ptr(), &data, 64, &barrier);
        });
        mbarrier_wait(&barrier, 0);
        let landed = make_tensor(&data, Layout::col_major(16));
        // The other warp reads too: waiting on the mbarrier orders it.
        if_(t.ge(48), || out.store(t - 48, landed.load(t - 48)));
    })
}

/// A bulk copy is visible once its mbarrier's phase is over.
#[test]
fn mbarrier_orders_bulk_copies() {
    let kernel = bulk_copy_kernel();
    let src: Vec<u32> = (100..116).collect();
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 16])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<u32>(), src);
}

/// A phase that never completes is reported, not hung on.
#[test]
fn waiting_without_an_arrival_is_reported() {
    let kernel = Kernel::builder("stuck").block_dim([32, 1, 1]).build(|| {
        let barrier = alloc_smem::<u64>(1, 8);
        mbarrier_init(&barrier, 1);
        mbarrier_wait(&barrier, 0);
    });
    let opts = SimOptions { max_while_iterations: 100, ..SimOptions::default() };
    let e = kernel.simulate_with(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut [], &opts).unwrap_err();
    assert!(e.message.contains("did not terminate"), "{e}");
}

#[test]
fn hopper_instructions_are_rejected_for_older_targets() {
    let Some(tc) = Toolchain::discover() else { return };
    let kernel = bulk_copy_kernel();
    let e = kernel.compile_ptx(&tc, Arch::Sm80).unwrap_err().to_string();
    assert!(e.contains("sm_90 or newer"), "{e}");
    let ptx = kernel.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains("mbarrier.try_wait.parity.shared::cta.b64"), "{ptx}");
    assert!(ptx.contains("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"), "{ptx}");
}

/// One compiled kernel, several problem sizes.
#[test]
fn runtime_shapes() {
    #[allow(dead_code)]
    #[path = "../examples/runtime_gemm.rs"]
    mod runtime_gemm;
    let kernel = runtime_gemm::build();
    for (m, n, k) in [(128, 128, 32), (256, 128, 96), (128, 256, 160)] {
        runtime_gemm::run(&kernel, m, n, k);
    }
    // A size the kernel's alignment proof excludes is refused at launch.
    let (launch, mut args, ..) = runtime_gemm::problem(128, 128, 32);
    args[4] = KernelArg::ScalarI32(40);
    let e = kernel.simulate(launch, &mut args).unwrap_err();
    assert!(e.message.contains("must be a multiple of 32"), "{e}");
}

/// A xorshift generator for the property tests.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Shuffles of every kind with random per-lane operands `b` and `c`
/// (`mask_and_clamp`), over many seeds: the simulator follows the PTX
/// ISA's pseudo-code (`feature_kernels::shfl_ref`). Every other `c` is a
/// segment configuration (a segment mask of whole low bits and a clamp
/// below the segment size), which random bits rarely form.
#[test]
fn shuffles_follow_the_ptx_pseudo_code_on_random_operands() {
    let case = feature_kernels::shuffle_operands();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..200 {
        let xs: Vec<u32> = (0..64).map(|_| xorshift(&mut state) as u32).collect();
        let bs: Vec<i32> = (0..64).map(|_| xorshift(&mut state) as i32).collect();
        let cs: Vec<i32> = (0..64)
            .map(|t| {
                let r = xorshift(&mut state);
                if t % 2 == 0 {
                    r as i32
                } else {
                    let width: i32 = 1 << (r % 6);
                    ((32 - width) << 8) | ((r >> 8) as i32 & (width - 1))
                }
            })
            .collect();
        let mut args = vec![
            KernelArg::buffer(&xs),
            KernelArg::buffer(&bs),
            KernelArg::buffer(&cs),
            KernelArg::buffer(&[0u32; 256]),
        ];
        case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
        (case.check)(&args);
    }
}

/// Warp collectives over random partitions of each warp into groups (each
/// lane's member mask the lanes of its group), over many seeds: integer
/// `redux.sync` of every kind, `match.all` and a broadcast from each
/// group's lowest lane.
#[test]
fn warp_collectives_over_random_partitions() {
    let case = feature_kernels::warp_partitions();
    let outputs = case.kernel.ir().buffers[2].len.expect("a sized output");
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..200 {
        let groups: Vec<u32> = (0..64).map(|_| (xorshift(&mut state) as u32) << 3).collect();
        let xs: Vec<u32> = (0..64).map(|_| xorshift(&mut state) as u32).collect();
        let mut args =
            vec![KernelArg::buffer(&groups), KernelArg::buffer(&xs), KernelArg::buffer(&vec![0u32; outputs])];
        case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
        (case.check)(&args);
    }
}

/// The kernels only the GPU can run compile, and the simulator refuses them
/// (their inline PTX follows pointers the simulation closure cannot, or
/// their hand-written MLIR reads hardware state): it reports the statement
/// it cannot simulate.
#[test]
fn gpu_only_kernels_compile_but_do_not_simulate() {
    for case in feature_kernels::gpu_only() {
        let mut args = inputs_with_len(&case.kernel, true, 256);
        let e = case.kernel.simulate(case.cfg, &mut args).expect_err(case.name);
        assert!(e.message.contains("cannot be simulated"), "{}: {e}", case.name);
    }
    let Some(tc) = Toolchain::discover() else { return };
    for case in feature_kernels::gpu_only() {
        case.kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    }
}

/// One layout places shared memory for the emitter, the launcher and the simulator: build-time
/// buffers first, then the dynamic region at a multiple of the largest alignment asked for (and
/// of 16). Every `get_dyn_smem` returns the region's start, whatever its element type, and a
/// kernel with a region keeps all its shared memory in the dynamic window.
#[test]
fn dynamic_region_layout() {
    let kernel = Kernel::builder("regions").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        let words = alloc_smem::<u32>(25, 4);
        let wide = get_dyn_smem::<u64>(None);
        let bytes = get_dyn_smem::<u8>(Some(1024));
        if_(t.eq(0), || {
            words.store(5u32);
            wide.store(0x0102_0304_0506_0708u64);
        });
        sync_threads();
        out.store(t, bytes.offset(t % 8).load().to::<u32>() + words.load());
    });
    let ir = kernel.ir();
    let layout = ir.shared_memory_layout();
    let region = ir.dynamic_smem_region().expect("a region");
    assert_eq!(ir.buffers.iter().filter(|b| b.len.is_none()).count(), 1);
    assert_eq!((ir.buffer(region).align, ir.buffer(region).elem), (1024, DType::U8));
    assert_eq!((layout.static_bytes, layout.in_window, layout.dynamic_start), (100, true, 1024));
    assert_eq!(layout.starts[region.0 as usize], Some(1024));
    assert_eq!((layout.launch_bytes(64), layout.block_bytes(64)), (1088, 1088));
    assert!(kernel.dump().contains("shared dynamic_smem: ? x u8 (align 1024)"), "{}", kernel.dump());
    // The window is an external array as aligned as its most aligned buffer, and nothing else
    // is shared memory.
    assert_eq!(layout.window_align, 1024);
    let mlir = kernel.to_mlir();
    let window = "memref.global @__cute_dynamic_smem : memref<0xi8, #gpu.address_space<workgroup>> \
                  {alignment = 1024 : i64}";
    assert_eq!(mlir.matches("memref.global").count(), 1, "{mlir}");
    assert!(mlir.contains(window), "{mlir}");
    let mut args = vec![KernelArg::buffer(&[0u32; 32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]).with_dynamic_smem(8), &mut args).unwrap();
    let want: Vec<u32> = (0..32).map(|t| 5 + (8 - t % 8)).collect();
    assert_eq!(args[0].to_vec::<u32>(), want);
    // Without a region, small build-time buffers stay static, large ones move to the window,
    // and a launch's dynamic bytes come on top.
    let small = Kernel::build("small", || alloc_smem::<u8>(100, 1).store(1u8)).ir().shared_memory_layout();
    assert_eq!((small.static_bytes, small.in_window, small.dynamic_start), (100, false, 0));
    assert_eq!((small.launch_bytes(28), small.block_bytes(28)), (28, 128));
    let big = Kernel::build("big", || alloc_smem::<u8>(50_000, 16).store(1u8)).ir().shared_memory_layout();
    assert_eq!((big.static_bytes, big.in_window, big.dynamic_start), (50_000, true, 50_000));
    assert_eq!((big.launch_bytes(100), big.block_bytes(100)), (50_100, 50_100));
    assert_eq!((small.window_align, big.window_align), (16, 16));
    // A region's default alignment is 16.
    let plain = Kernel::build("plain", || {
        alloc_smem::<u8>(3, 1).store(1u8);
        get_dyn_smem::<f32>(None).store(1.0f32);
    });
    assert_eq!(plain.ir().shared_memory_layout().dynamic_start, 16);
}

/// `exp2_packed_f16x2` on every Float16 value: the simulator gives the
/// correctly rounded `2^x` (the GPU approximates it, within the bound
/// `tests/gpu.rs` checks), and the input where the predicate is false.
#[test]
fn exp2_packed_simulates_every_value() {
    let case = feature_kernels::exp2_f16x2();
    (case.check)(&simulate(&case, true));
}

/// The int8 to BFloat16 conversions of sm_100a-class targets, which no
/// kernel list holds (PTX 9.1), give the int8 values exactly.
#[test]
fn s2f6x2_conversions_are_exact() {
    let case = feature_kernels::cvt_i8_bf16_s2f6x2();
    (case.check)(&simulate(&case, true));
}

/// The `with_shuffle` int4 to BFloat16 conversions (the scaled `.s2f6x2` cvt, PTX 9.1), which
/// no kernel list holds, give the int4 values exactly in natural order from shuffled input.
#[test]
fn int4_shuffle_conversions_are_exact() {
    let case = feature_kernels::int4_bf16_shuffle();
    (case.check)(&simulate(&case, true));
}

/// Fill the buffers of `case` with random bits from `state`.
fn random_bits(case: &feature_kernels::Case, state: &mut u64) -> Vec<KernelArg> {
    let ir = case.kernel.ir();
    ir.args
        .iter()
        .map(|p| match p {
            cute::ir::Param::Buffer(b) => {
                let d = ir.buffer(*b);
                let n = d.len.expect("sized buffers");
                let data = (0..n * d.elem.bytes()).map(|_| xorshift(state) as u8).collect();
                KernelArg::Bytes { elem: d.elem, data }
            }
            cute::ir::Param::Scalar { .. } | cute::ir::Param::ByValue { .. } => panic!("no scalar parameters here"),
        })
        .collect()
}

/// The packed and scalar half-precision operations, the conversions and
/// Pack/Unpack on random bit patterns (every kind of value, NaN payloads
/// included) over many seeds: the simulator matches the plain-Rust
/// references.
#[test]
fn half_precision_and_conversions_on_random_bits() {
    let mut state = 0x853c_49e6_748f_ea9bu64;
    for case in [
        feature_kernels::half_f16(),
        feature_kernels::half_bf16(),
        feature_kernels::half_sm90(),
        feature_kernels::cvt_floats(),
        feature_kernels::cvt_int8(),
        feature_kernels::pack_unpack(),
    ] {
        for _ in 0..24 {
            let mut args = random_bits(&case, &mut state);
            case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

/// `fma` on Float16 and BFloat16 values in the float kernels rounds once,
/// and so does their reference: for these operands an f32 fma rounded to
/// the type gives a tie and then the even neighbour, one ulp off. The
/// operands go in among the input data, `x` with both signs, and the
/// kernels' checks must pass on them.
#[test]
fn float_kernels_round_half_fma_once() {
    /// The fma column of threads 0 and 37, which get `x` (negated in
    /// thread 37) and `y`.
    fn fma_of<T: Numeric>(case: &Case, x: u16, y: u16) -> [u16; 2] {
        let mut args = inputs_with_len(&case.kernel, false, 256);
        let (mut xs, mut ys) = (args[0].to_vec::<T>(), args[1].to_vec::<T>());
        for (i, sign) in [(0, 0), (37, 0x8000)] {
            xs[i] = T::from_bits((x ^ sign) as u64);
            ys[i] = T::from_bits(y as u64);
        }
        (args[0], args[1]) = (KernelArg::buffer(&xs), KernelArg::buffer(&ys));
        case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
        let out = args[2].to_vec::<T>();
        let ops = out.len() / xs.len();
        [0, 37].map(|i| out[i * ops + ops - 1].raw_bits() as u16)
    }
    // 0.19507 * 26400 + 0.19507 = 5149.99976 is 5148 in Float16; through f32
    // it is 5150, a tie, which goes to 5152.
    let (x, y) = (F16(0x323e).to_f32(), F16(0x7672).to_f32());
    assert_eq!(F16::from_f32(x.mul_add(y, x)).0, 0x6d08, "Float16 rounded twice");
    assert_eq!(fma_of::<F16>(&feature_kernels::float_f16(), 0x323e, 0x7672), [0x6d07, 0xed07]);
    // A subnormal x: x * y is a BFloat16 midpoint and x tips it up, but f32
    // loses x.
    let (x, y) = (BF16(0x0026).to_f32(), BF16(0x7e98).to_f32());
    assert_eq!(BF16::from_f32(x.mul_add(y, x)).0, 0x3eb4, "BFloat16 rounded twice");
    assert_eq!(fma_of::<BF16>(&feature_kernels::float_bf16(), 0x0026, 0x7e98), [0x3eb5, 0xbeb5]);
}

/// One built kernel at many launch-time extents: `dynamic_tiles`, whose
/// global tensors have runtime extents, with partial tiles at the bottom
/// and right edges, against plain Rust. The buffers are exactly the
/// tensors' size (so that the simulator reports any access past them) or
/// padded (so that the padding shows any write past the tensors).
#[test]
fn dynamic_tiles_at_several_extents() {
    let case = feature_kernels::dynamic_tiles();
    for (m, n) in [(1, 4), (8, 8), (13, 20), (17, 44), (32, 64), (5, 100), (64, 4), (9, 12)] {
        for pad in [0, 37] {
            let (cfg, mut args) = feature_kernels::dynamic_tiles_problem(m, n, pad, (m * 1000 + n + pad) as u64);
            case.kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{m} x {n}: {e}"));
            let want = feature_kernels::dynamic_tiles_expected(m, n, &args);
            assert_eq!(args[3].to_vec::<i32>(), want, "{m} x {n}, padded by {pad}");
            (case.check)(&args);
        }
    }
    // The launch checks the promise that n is a multiple of 4, which the
    // 128-bit copies rely on.
    let (cfg, mut args) = feature_kernels::dynamic_tiles_problem(8, 8, 0, 1);
    args[1] = KernelArg::ScalarI32(6);
    let e = case.kernel.simulate(cfg, &mut args).unwrap_err();
    assert!(e.message.contains("must be a multiple of 4"), "{e}");
}

/// `autovec_copy` picks vectors from `max_common_layout`: whole-tile runs
/// of row-major tiles and their fragments, one row of a tile with a runtime
/// row stride, the swizzle's runs, and 32-byte vectors from 32-byte-aligned
/// buffers. Counted as the IR's loads and stores of each width.
#[test]
fn autovec_copy_vector_widths() {
    use cute::ir::Op;
    let widths = |k: &Kernel| {
        let loads = |n: usize| k.ir().count_ops(|op| matches!(op, Op::Load { results, .. } if results.len() == n));
        let stores = |n: usize| k.ir().count_ops(|op| matches!(op, Op::Store { values, .. } if values.len() == n));
        ((loads(4), loads(8), loads(1)), (stores(4), stores(8), stores(1)))
    };
    // 32 elements each way: eight 16-byte vectors.
    for case in [feature_kernels::autovec_row_major(), feature_kernels::autovec_runtime_stride()] {
        assert_eq!(widths(&case.kernel), ((8, 0, 0), (8, 0, 0)), "{}", case.name);
    }
    // The swizzled kernel fills its tile with scalar stores first.
    let swizzled = feature_kernels::autovec_swizzled();
    assert_eq!(widths(&swizzled.kernel), ((8, 0, 32), (8, 0, 32)));
    assert_eq!(widths(&feature_kernels::autovec_256().kernel), ((0, 1, 0), (0, 1, 0)));
}

/// CuTe DSL's tensor properties: `memspace`, `stride`, `element_type` and
/// `dtype`, `iterator`, `leading_dim`, and the pointers' `memspace` and
/// `max_alignment`; `AddressSpace` has NVVM's numbers.
#[test]
fn tensor_metadata() {
    use cute::ir::MemSpace;
    let numbers = [
        AddressSpace::Generic,
        AddressSpace::Gmem,
        AddressSpace::Smem,
        AddressSpace::Cmem,
        AddressSpace::Rmem,
        AddressSpace::Tmem,
        AddressSpace::Dsmem,
    ]
    .map(|a| a as u32);
    assert_eq!(numbers, [0, 1, 3, 4, 5, 6, 7]);
    assert_eq!(AddressSpace::from(MemSpace::Local), AddressSpace::Rmem);
    assert_eq!(MemSpace::try_from(AddressSpace::Smem), Ok(MemSpace::Shared));
    assert_eq!(MemSpace::try_from(AddressSpace::Tmem), Err(AddressSpace::Tmem));
    assert_eq!(AddressSpace::Dsmem.to_string(), "dsmem");
    Kernel::build("tensor_metadata", || {
        let k = param_extent("k", 4);
        let g = param_tensor::<f32>("g", Layout::with_strides((8, 16), (k, 1)));
        let s = make_smem_tensor::<F16>(make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64))));
        let r = make_fragment::<i32>((4, 2));
        let stack = make_rmem_tensor::<f32>(Layout::col_major(8));
        assert_eq!(
            [g.memspace(), s.memspace(), r.memspace(), stack.memspace()],
            [AddressSpace::Gmem, AddressSpace::Smem, AddressSpace::Rmem, AddressSpace::Rmem]
        );
        assert_eq!(g.stride(), Layout::with_strides((8, 16), (k, 1)).stride());
        assert_eq!((g.dtype(), s.element_type(), r.dtype()), (DType::F32, DType::F16, DType::I32));
        assert_eq!((g.leading_dim(), r.leading_dim(), s.leading_dim()), (Some(vec![1]), Some(vec![0]), Some(vec![1])));
        assert!(matches!(r.iterator(), TensorIterator::Registers));
        let TensorIterator::Ptr(p) = g.slice((1, ..)).iterator() else { panic!("a memory tensor has a pointer") };
        assert_eq!(p.memspace(), AddressSpace::Gmem);
        // Row 1 starts k (a multiple of 4) floats in: 16 bytes.
        assert_eq!((g.ptr().max_alignment(), p.max_alignment(), p.offset(2).max_alignment()), (16, 16, 8));
        // The swizzled tensor's pointer carries its swizzle in bytes,
        // S<3,4,3> for S<3,3,3> on f16, which keeps runs of 16 bytes
        // together: its alignment, as CuTe DSL's swizzled pointers have.
        assert_eq!((s.ptr().swizzle(), s.ptr().max_alignment()), (Some(make_swizzle(3, 4, 3)), 16));
        let id = make_identity_tensor((4, 8)).slice((1, ..));
        let TensorIterator::Coord(origin) = id.iterator() else { panic!("a coordinate tensor has an origin") };
        assert_eq!(origin.to_string(), "(1,0)");
        // make_fragment_like follows the source's stride order; the dtype
        // form takes a layout's fragment layout.
        assert_eq!(make_fragment_like::<f32, f32>(&g).layout(), &Layout::new((8, 16), (16, 1)));
        assert_eq!(make_fragment_like::<f32, F16>(&s).layout(), &Layout::new((8, 64), (64, 1)));
        assert_eq!(make_fragment_like::<f32, Ident>(&id).layout(), &Layout::col_major(id.shape().clone()));
        let frag = make_fragment_like_dtype::<f32>(&Layout::new((4, 8), (8, 1)));
        assert_eq!((frag.layout(), frag.memspace()), (&Layout::new((4, 8), (1, 4)), AddressSpace::Rmem));
    });
}

/// The kernels declared through the kernel argument protocol
/// (`feature_kernels::param_cases`): at several problem sizes from host
/// values, the simulator computes the plain-Rust results, and each kernel
/// compiles for sm_80 (when `mlir-opt` is available).
#[test]
fn parameter_protocol_kernels_match_rust() {
    let toolchain = Toolchain::discover();
    for case in feature_kernels::params::param_cases() {
        let run = |k: &Kernel, cfg: LaunchConfig, args: &mut Vec<KernelArg>| {
            k.simulate(cfg, args).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        };
        (case.check)(&run);
        if let Some(tc) = &toolchain {
            for k in &case.kernels {
                k.compile_ptx(tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
            }
        }
    }
}

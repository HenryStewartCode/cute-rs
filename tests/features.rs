//! Every type, operation and intrinsic: the simulator is held to a reference
//! written in Rust, and every kernel must compile to PTX (when `mlir-opt` is
//! available) so that each path of the MLIR emitter is exercised.

mod feature_kernels;

use cute::prelude::*;
use feature_kernels::{Case, all, inputs_with_len, sm90};

fn simulate(case: &Case, small: bool) -> Vec<KernelArg> {
    let mut args = inputs_with_len(&case.kernel, small, 256);
    case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    args
}

#[test]
fn simulator_matches_rust_on_small_integers() {
    for case in all().into_iter().chain(sm90()) {
        (case.check)(&simulate(&case, true));
    }
}

#[test]
fn simulator_matches_rust_on_random_data() {
    for case in all().into_iter().chain(sm90()).filter(|c| !c.small_only) {
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
        .filter_map(|(c, arch)| c.kernel.compile_ptx(&tc, arch).err().map(|e| format!("{}: {e}", c.name)))
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

/// Shuffling from a lane that is not running is an error.
#[test]
fn shuffle_from_inactive_lane_is_reported() {
    let kernel = Kernel::builder("bad_shuffle").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        if_(t.lt(4), || out.store(t, shuffle_sync(t, 20)));
    });
    let mut args = vec![KernelArg::i32(vec![0; 32])];
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("not active"), "{e}");
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

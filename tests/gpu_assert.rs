//! A failing device assertion on a real GPU. It has its own test binary:
//! `CUDA_ERROR_ASSERT` is sticky, and poisons the process's primary context
//! for every other test until it is reset. The kernel runs in a child
//! process, whose stderr holds the device's `Assertion ... failed` line.
//! `CudaContext::reset` does not recover from it (the driver keeps refusing
//! the process), which the test checks too.
//! Needs the `cuda` feature; skipped (with a note) without an NVIDIA GPU or
//! an `mlir-opt`.

mod support;

use cute::cuda::CudaContext;
use cute::prelude::*;
use support::child;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

/// The line of the assertion, for the device's message.
const ASSERT_LINE: u32 = line!() + 6;

fn asserting_kernel() -> Kernel {
    Kernel::builder("asserting").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<i32>("x");
        let t = thread_idx().x;
        cute::testing::assert_(x.load_at(t).ge(0), "inputs are non-negative");
        x.store_at(t, x.load_at(t) + 1);
    })
}

fn run(ctx: &CudaContext, tc: &Toolchain, k: &Kernel, input: Vec<i32>, ptxas: bool) -> Result<Vec<i32>, String> {
    let opts = CompileOptions::new().enable_assertions(true).use_ptxas(ptxas);
    let mut args = [KernelArg::i32(input)];
    k.run_cuda_with(ctx, tc, &opts, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).map_err(|e| e.to_string())?;
    Ok(args[0].as_i32().to_vec())
}

#[test]
fn failing_device_assertion() {
    if child::is_child() {
        let Some((mut ctx, tc)) = device() else { return };
        let k = asserting_kernel();
        // Passing assertions, through the driver's JIT (which resolves
        // __assertfail as it resolves vprintf) and, where there is one,
        // through ptxas.
        let ptxas = tc.cuda_tools().ptxas.is_some();
        for with_ptxas in [false, ptxas] {
            assert_eq!(run(&ctx, &tc, &k, vec![1; 32], with_ptxas).unwrap(), vec![2; 32]);
        }
        child::begin();
        let mut input = vec![1; 32];
        input[5] = -1;
        let e = run(&ctx, &tc, &k, input, false).unwrap_err();
        child::end();
        println!("LAUNCH ERROR: {e}");
        // The context is dead.
        assert!(run(&ctx, &tc, &k, vec![1; 32], false).is_err());
        // A reset does not bring it back: the driver refuses the process
        // with the same error (if a future driver recovers, the kernel runs).
        match ctx.reset() {
            Err(e) => {
                assert!(e.is_device_assert(), "{e}");
                println!("RESET REFUSED: {e}");
            }
            Ok(()) => {
                assert_eq!(run(&ctx, &tc, &k, vec![3; 32], false).unwrap(), vec![4; 32]);
                println!("RESET OK");
            }
        }
        return;
    }
    if device().is_none() {
        return;
    }
    let out = child::run_child("failing_device_assertion");
    assert!(out.success, "the child failed:\n{}\n{}", out.stdout, out.stderr);
    // The driver writes a failed device assertion to stderr.
    let assert_line = out
        .stderr
        .lines()
        .find(|l| l.contains("Assertion"))
        .unwrap_or_else(|| panic!("{}\n{}", out.stdout, out.stderr));
    assert!(assert_line.contains("inputs are non-negative"), "{assert_line}");
    assert!(assert_line.contains(&format!("{}:{ASSERT_LINE}", file!())), "{assert_line}");
    assert!(assert_line.contains("block: [0,0,0], thread: [5,0,0]"), "{assert_line}");
    let launch = out.stdout.lines().find(|l| l.starts_with("LAUNCH ERROR: ")).expect("the launch error");
    assert!(launch.contains("CUDA_ERROR_ASSERT: device-side assert triggered"), "{launch}");
    assert!(launch.contains("run kernels that may assert in a child process"), "{launch}");
    assert!(out.stdout.contains("RESET OK") || out.stdout.contains("RESET REFUSED"), "{}", out.stdout);
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! The device a GPU test of one architecture family runs on, and a kernel compiled and loaded
//! once and launched many times (for the timings).
//!
//! A test asks for a compute capability major (9 for Hopper, 10 for datacenter Blackwell) and is
//! skipped, with a note, on any other device, without a GPU or without an `mlir-opt`. On the
//! machine the tests are meant for, set `CUTE_RS_EXPECT_SM` to its compute capability (`90`,
//! `100`, `103`): a test of that family then fails instead of skipping, so a misconfigured run
//! cannot pass by testing nothing.

// Shared by several test binaries; each uses part of it.
#![allow(dead_code)]

use cute::cuda::{CudaContext, CudaFunction, HardwareInfo};
use cute::prelude::*;

pub struct Device {
    pub ctx: CudaContext,
    pub tc: Toolchain,
    pub arch: Arch,
}

/// The device, if it is of compute capability `major.x`.
pub fn device(major: i32, what: &str) -> Option<Device> {
    let expected = std::env::var("CUTE_RS_EXPECT_SM").ok().and_then(|s| s.parse::<i32>().ok());
    let must_run = expected.is_some_and(|sm| sm / 10 == major);
    let skip = |why: String| -> Option<Device> {
        assert!(!must_run, "{what}: CUTE_RS_EXPECT_SM={} but {why}", expected.unwrap());
        eprintln!("skipping {what}: {why}");
        None
    };
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => return skip(format!("no CUDA device ({e})")),
    };
    let (ma, mi) = ctx.compute_capability();
    if ma != major {
        return skip(format!("{} is sm_{ma}{mi}, not sm_{major}x", ctx.name()));
    }
    if let Some(sm) = expected
        && must_run
    {
        assert_eq!(sm, ma * 10 + mi, "{what}: CUTE_RS_EXPECT_SM={sm} but {} is sm_{ma}{mi}", ctx.name());
    }
    let Some(tc) = Toolchain::discover() else {
        return skip("no mlir-opt found (set CUTE_MLIR_OPT)".into());
    };
    let Some(arch) = ctx.best_target(&tc) else {
        return skip(format!("no target for sm_{ma}{mi} with {}", tc.mlir_opt.display()));
    };
    Some(Device { ctx, tc, arch })
}

impl Device {
    /// Compile `kernel` for this device, run it once with `args` (copied in and back) and
    /// return the host-timed seconds of the launch.
    pub fn run(&self, kernel: &Kernel, cfg: LaunchConfig, args: &mut [KernelArg]) -> f64 {
        let loaded = self.load(kernel);
        loaded.launch(cfg, args)
    }

    pub fn load<'d>(&'d self, kernel: &'d Kernel) -> Loaded<'d> {
        // Without contraction (cute-rs's default fuses `a * b + c` into `fma`): the tests check
        // results bit for bit against the simulator's CPU references, which round every
        // operation.
        // A kernel built for an architecture of its own (`KernelOptions::target`) is compiled
        // for that one: the driver compiles its PTX for this device at load time (an sm_90
        // kernel on an sm_100 GPU), and the load fails if the device cannot run it.
        let arch = kernel.target().unwrap_or(self.arch);
        let options = CompileOptions::new().gpu_arch(arch).fp_contract(false);
        let compiled = kernel
            .compile_with(&self.tc, &options)
            .unwrap_or_else(|e| panic!("{}: compiling for {arch}: {e}", kernel.name()));
        let function = self.ctx.load(&compiled).unwrap_or_else(|e| panic!("{}: loading: {e}", kernel.name()));
        Loaded { device: self, kernel, function }
    }

    pub fn sm_count(&self) -> u32 {
        HardwareInfo::new(0).and_then(|h| h.get_device_multiprocessor_count()).expect("the SM count")
    }

    pub fn max_active_clusters(&self, cluster_size: u32) -> u32 {
        HardwareInfo::new(0).and_then(|h| h.get_max_active_clusters(cluster_size, None)).expect("max active clusters")
    }
}

pub struct Loaded<'d> {
    device: &'d Device,
    kernel: &'d Kernel,
    function: CudaFunction<'d>,
}

impl Loaded<'_> {
    /// One launch: `args` copied in, the kernel run, the buffers copied back; the seconds the
    /// launch took (a host timer around launch and synchronisation, copies excluded).
    pub fn launch(&self, cfg: LaunchConfig, args: &mut [KernelArg]) -> f64 {
        self.device
            .ctx
            .launch(self.kernel, &self.function, cfg, args)
            .unwrap_or_else(|e| panic!("{}: launch: {e}", self.kernel.name()))
            .kernel_seconds
    }

    /// The fastest of `reps` launches after one warm-up, in seconds.
    pub fn best_of(&self, reps: usize, cfg: LaunchConfig, args: &mut [KernelArg]) -> f64 {
        self.launch(cfg, args);
        (0..reps).map(|_| self.launch(cfg, args)).fold(f64::INFINITY, f64::min)
    }
}

/// A timing line the run script collects: `BENCH name shape seconds TFLOP/s`.
pub fn report(name: &str, shape: &str, flops: f64, seconds: f64) {
    println!("BENCH {name} {shape} {:.1}us {:.1} TFLOP/s", seconds * 1e6, flops / seconds / 1e12);
}

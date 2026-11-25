// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/tutorial/elementwise_add_autotune.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   (the kernel `elementwise_add_kernel` and host function `elementwise_add_autotune`, the
//   tuned `copy_bits` of 64 and 128, the unimplementable 1, `update_on_change=["M", "N"]`
//   and the problem sizes), and examples/python/CuTeDSL/cute/notebooks/benchmark_autotune.ipynb
//   (the 1024 x 1024 problem and the throughput report), a notebook with no header of its
//   own: the README.md of its folder carries
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and CUTLASS's LICENSE.txt
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file is cute-rs's
// own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text and lists what
// each part ports.

//! Autotuning the tiled element-wise addition, ported from CuTe DSL's
//! `elementwise_add_autotune.py`.
//!
//! `C = A + B` on `M x N` row-major Float32 tensors whose extents are launch-time values. A
//! [`TileConfig`] chooses the tile: a thread layout (4 x 32 threads in the example) and a
//! value layout of 4 rows of `copy_bits / 32` elements, the width of each thread's vector
//! copies. The tensors are `zipped_divide`d by the TV tiler, each block takes one tile, and
//! an identity tensor predicates the copies at the edges, as in `elementwise_add.py`. The
//! row length `N` (which is also the row stride) is a multiple of the vector width, so the
//! wide copies stay aligned.
//!
//! Every candidate tile, including one with `copy_bits = 1`, which cannot be built (as the
//! example's `copy_bits=1` cannot be implemented), is checked on the simulator. With the
//! `cuda` feature and a GPU, `cute::testing::tune` times the candidates and skips the
//! unbuildable one, and an `Autotuner` keyed on the kernel's spec and `M` and `N`
//! (`update_on_change=["M", "N"]`) tunes each new problem size once and reuses the winner.
//!
//! ```text
//! cargo run --release --example elementwise_add_autotune
//! cargo run --release --features cuda --example elementwise_add_autotune
//! ```

use cute::prelude::*;

/// One candidate: the thread layout (rows x columns of threads, row-major) and the bits each
/// thread copies at once along a row (`copy_bits` of the example).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileConfig {
    pub threads: (usize, usize),
    pub copy_bits: usize,
}

impl TileConfig {
    /// Elements per vector copy (`copy_bits // dtype.width`).
    pub fn vector(&self) -> usize {
        self.copy_bits / 32
    }

    /// The tile a block covers: `(4 * thread rows, vector * thread columns)`.
    pub fn tile(&self) -> (usize, usize) {
        (4 * self.threads.0, self.vector() * self.threads.1)
    }

    pub fn block_threads(&self) -> u32 {
        (self.threads.0 * self.threads.1) as u32
    }

    /// The grid for an `m x n` problem: one block per tile (`size(gC, mode=[1])`).
    pub fn grid(&self, m: usize, n: usize) -> [u32; 3] {
        let (tm, tn) = self.tile();
        [(m.div_ceil(tm) * n.div_ceil(tn)) as u32, 1, 1]
    }

    /// The launch for an `m x n` problem.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        LaunchConfig::new(self.grid(m, n), [self.block_threads(), 1, 1])
    }
}

/// The candidates: the example's `copy_bits` of 64 and 128 (and 32) on 4 x 32 threads,
/// 128-bit copies on 8 x 16 and 2 x 64 threads, and the unbuildable `copy_bits = 1`.
pub const CANDIDATES: [TileConfig; 6] = [
    TileConfig { threads: (4, 32), copy_bits: 1 },
    TileConfig { threads: (4, 32), copy_bits: 32 },
    TileConfig { threads: (4, 32), copy_bits: 64 },
    TileConfig { threads: (4, 32), copy_bits: 128 },
    TileConfig { threads: (8, 16), copy_bits: 128 },
    TileConfig { threads: (2, 64), copy_bits: 128 },
];

/// The kernel parameters: A, B and C.
pub type AddParams = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// The spec of an `M x N` row-major tensor whose `N` (also its row stride) is a multiple of
/// `vector`, starting at an address aligned to the vector: one `M` and one `N` for the three
/// tensors.
pub fn spec(vector: usize) -> TensorSpec {
    let vector = vector.max(1);
    let (m, n) = (sym_int32(1, Some("M")), sym_int32(vector, Some("N")));
    make_fake_tensor::<f32>((m, n), (n, 1)).with_assumed_align(4 * vector)
}

/// The kernel for `cfg` (`elementwise_add_autotune` with its `copy_bits`, and the kernel it
/// launches). Panics for a `copy_bits` below one element, as building the value layout
/// fails.
pub fn kernel(cfg: TileConfig) -> TypedKernel<AddParams> {
    let vector = cfg.vector();
    let t = spec(vector);
    Kernel::builder(format!("elementwise_add_{}x{}_{}b", cfg.threads.0, cfg.threads.1, cfg.copy_bits))
        .block_dim([cfg.block_threads(), 1, 1])
        .jit((t.clone(), t.clone(), t), move |m_a: Tensor<f32>, m_b: Tensor<f32>, m_c: Tensor<f32>| {
            assert!(vector > 0, "copy_bits = {} copies less than one Float32", cfg.copy_bits);
            let thr_layout = make_ordered_layout(cfg.threads, &[1, 0]);
            let val_layout = make_ordered_layout((4, vector), &[1, 0]);
            let (tiler_mn, _tv_layout) = make_layout_tv(&thr_layout, &val_layout);
            let g_a = zipped_divide(&m_a, &tiler_mn); // ((TileM,TileN),(RestM,RestN))
            let g_b = zipped_divide(&m_b, &tiler_mn);
            let g_c = zipped_divide(&m_c, &tiler_mn);
            let c_c = zipped_divide(&make_identity_tensor(m_c.shape()), &tiler_mn);
            let shape = m_c.shape().clone();

            let tidx = thread_idx().x;
            let blk_coord = ((.., ..), block_idx().x);
            let (blk_a, blk_b, blk_c) = (g_a.slice(blk_coord), g_b.slice(blk_coord), g_c.slice(blk_coord));
            let blk_crd = c_c.slice(blk_coord);

            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: cfg.copy_bits });
            let tiled_copy = make_tiled_copy_tv(&atom, &thr_layout, &val_layout);
            let thr = tiled_copy.get_slice(tidx);
            let (thr_a, thr_b, thr_c) = (thr.partition_s(&blk_a), thr.partition_s(&blk_b), thr.partition_d(&blk_c));
            let frg_a = make_fragment_like::<f32, _>(&thr_a);
            let frg_b = make_fragment_like::<f32, _>(&thr_b);
            let frg_c = make_fragment_like::<f32, _>(&thr_c);

            let thr_crd = thr.partition_s(&blk_crd);
            let frg_pred = make_fragment::<bool>(thr_crd.shape().clone());
            for i in 0..frg_pred.size() {
                frg_pred.store(i, elem_less(thr_crd.coord(i), shape.clone()));
            }

            copy_if(&tiled_copy, &frg_pred, &thr_a, &frg_a);
            copy_if(&tiled_copy, &frg_pred, &thr_b, &frg_b);
            frg_c.store_ssa(&(frg_a.load_ssa() + frg_b.load_ssa()));
            copy_if(&tiled_copy, &frg_pred, &frg_c, &thr_c);
        })
}

/// Whether `cfg` can serve an `m x n` problem (the example's `can_implement`): its rows must
/// be a whole number of vectors.
pub fn can_implement(cfg: TileConfig, n: usize) -> Result<(), String> {
    match cfg.vector() {
        0 => Err(format!("copy_bits = {} is less than one Float32", cfg.copy_bits)),
        v if !n.is_multiple_of(v) => Err(format!("N = {n} is not a multiple of the {v}-element vector")),
        _ => Ok(()),
    }
}

/// The inputs of an `m x n` problem.
pub fn inputs(m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * n).map(|i| (i % 97) as f32 * 0.5 + 0.125).collect();
    let b = (0..m * n).map(|i| (i % 13) as f32 - 6.0).collect();
    (a, b)
}

/// The host tensors of an `m x n` problem.
pub fn host(a: &[f32], b: &[f32], m: usize, n: usize) -> (HostTensor, HostTensor, HostTensor) {
    let t = |v: &[f32]| HostTensor::from_slice(v, Layout::row_major((m, n)));
    (t(a), t(b), t(&vec![0.0; m * n]))
}

/// Simulate `cfg` on an `m x n` problem and check C; the simulator's report.
pub fn simulate(cfg: TileConfig, m: usize, n: usize) -> Result<SimReport, String> {
    let k = std::panic::catch_unwind(|| kernel(cfg)).map_err(|_| format!("{cfg:?} cannot be built"))?;
    can_implement(cfg, n)?;
    let (a, b) = inputs(m, n);
    let mut h = host(&a, &b, m, n);
    let report = k.simulate(cfg.launch(m, n), &mut h).map_err(|e| e.to_string())?;
    let c = h.2.to_vec::<f32>();
    for i in 0..m * n {
        assert_eq!(c[i], a[i] + b[i], "{cfg:?} {m}x{n}: mismatch at {i}");
    }
    Ok(report)
}

#[cfg(feature = "cuda")]
pub mod gpu {
    //! Tuning on the GPU.

    use std::error::Error;

    use cute::cuda::{CudaContext, CudaStream, DeviceArg, DeviceBuffer};
    use cute::prelude::*;
    use cute::testing::{Autotuner, BenchmarkOptions, CantImplement, TuneKey, Tuned, tune};

    use super::{AddParams, TileConfig, can_implement, host, inputs, kernel};

    /// A problem on the device: A, B and C, and the host values the arguments are packed
    /// from.
    pub struct Problem<'c> {
        pub m: usize,
        pub n: usize,
        pub a: Vec<f32>,
        pub b: Vec<f32>,
        pub buffers: [DeviceBuffer<'c>; 3],
    }

    impl<'c> Problem<'c> {
        pub fn new(ctx: &'c CudaContext, m: usize, n: usize) -> Result<Problem<'c>, Box<dyn Error>> {
            let (a, b) = inputs(m, n);
            let buffers = [ctx.upload_slice(&a)?, ctx.upload_slice(&b)?, ctx.alloc::<f32>(m * n)?];
            Ok(Problem { m, n, a, b, buffers })
        }

        /// C, checked against A + B.
        pub fn check(&self) -> Result<(), Box<dyn Error>> {
            let c = self.buffers[2].to_vec::<f32>()?;
            match (0..self.m * self.n).find(|&i| c[i] != self.a[i] + self.b[i]) {
                Some(i) => Err(format!("{}x{}: c[{i}] = {} != {}", self.m, self.n, c[i], self.a[i] + self.b[i]).into()),
                None => Ok(()),
            }
        }
    }

    /// The launch values of `k` for an `m x n` problem, packed from host tensors of that
    /// layout: placeholders for A, B and C, and `M` and `N`.
    pub fn packed(k: &TypedKernel<AddParams>, m: usize, n: usize) -> Result<Vec<KernelArg>, Box<dyn Error>> {
        let zeros = vec![0.0f32; m * n];
        Ok(k.pack(&host(&zeros, &zeros, m, n))?)
    }

    /// The device arguments of `k` on problem `p`: its buffers `[a, b, c]` for the pointer
    /// parameters and the scalars of `packed`.
    pub fn device_args<'a>(k: &TypedKernel<AddParams>, packed: &[KernelArg], p: &'a Problem<'a>) -> Vec<DeviceArg<'a>> {
        let mut buffers = p.buffers.iter();
        k.signature()
            .params
            .iter()
            .zip(packed)
            .map(|(param, arg)| match param.kind {
                ParamKind::Pointer { .. } => DeviceArg::Buffer(buffers.next().expect("three buffers")),
                _ => DeviceArg::Scalar(arg.clone()),
            })
            .collect()
    }

    /// Compile and load `cfg`'s kernel and return its launch for [`tune`] (the closure
    /// `compile_class` of the example returns): a panic when the kernel cannot be built
    /// (`copy_bits = 1`), `CantImplement` when the row length is not a whole number of
    /// vectors.
    #[allow(clippy::type_complexity)]
    pub fn build<'a>(
        ctx: &'a CudaContext,
        tc: &Toolchain,
        p: &'a Problem<'a>,
        cfg: &TileConfig,
    ) -> Result<impl FnMut(&CudaStream<'a>, usize) -> Result<(), Box<dyn Error>> + use<'a>, Box<dyn Error>> {
        let k = kernel(*cfg);
        can_implement(*cfg, p.n).map_err(CantImplement)?;
        let arch = ctx.best_target(tc).ok_or("no target for this GPU")?;
        let function = ctx.load(&k.compile_with(tc, &CompileOptions::new().gpu_arch(arch))?)?;
        let args = device_args(&k, &packed(&k, p.m, p.n)?, p);
        let launch = cfg.launch(p.m, p.n);
        let kernel = k.into_kernel();
        Ok(move |s: &CudaStream<'a>, _workspace: usize| -> Result<(), Box<dyn Error>> {
            Ok(ctx.launch_builder(&kernel, &function, launch).stream(s).args(&args).submit()?)
        })
    }

    /// `tune` over the candidates on problem `p` (the example's `tune_class`).
    pub fn tune_problem<'a>(
        ctx: &'a CudaContext,
        tc: &Toolchain,
        stream: &CudaStream<'a>,
        p: &'a Problem<'a>,
        options: &BenchmarkOptions,
    ) -> Result<Tuned<TileConfig>, Box<dyn Error>> {
        Ok(tune(stream, super::CANDIDATES, options, |cfg| build(ctx, tc, p, cfg))?)
    }

    /// The key of problem `p` for an [`Autotuner`]: the kernels' spec (the same for every
    /// size) and `M` and `N`.
    pub fn key(p: &Problem<'_>) -> TuneKey {
        TuneKey::new(&super::spec(1).cache_key()).part("M", p.m).part("N", p.n)
    }

    /// The `autotune_jit` of the example: an autotuner over the candidates that tunes again
    /// when `M` or `N` changes.
    pub fn autotuner(options: BenchmarkOptions) -> Autotuner<TileConfig> {
        Autotuner::new(super::CANDIDATES).update_on_change(&["M", "N"]).options(options)
    }
}

#[allow(dead_code)]
fn main() {
    // Every candidate on the simulator: the buildable ones compute C = A + B, edges included.
    for (m, n) in [(3, 12), (37, 64), (64, 128)] {
        for cfg in CANDIDATES {
            match simulate(cfg, m, n) {
                Ok(r) => println!("{m}x{n} {cfg:?}: correct, {} blocks, {} global sectors", r.blocks, r.global_sectors),
                Err(e) => println!("{m}x{n} {cfg:?}: skipped ({e})"),
            }
        }
    }
    #[cfg(feature = "cuda")]
    if let Err(e) = run_on_gpu() {
        eprintln!("GPU part skipped: {e}");
    }
}

#[cfg(feature = "cuda")]
fn run_on_gpu() -> Result<(), Box<dyn std::error::Error>> {
    use cute::cuda::CudaContext;
    use cute::testing::BenchmarkOptions;
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found (set CUTE_MLIR_OPT)")?;
    let stream = ctx.stream()?;
    let options = BenchmarkOptions::new().warmup_iterations(10).iterations(100);
    let mut tuner = gpu::autotuner(options.clone());
    for (m, n) in [(1024, 512), (1024, 1024), (1024, 512)] {
        let p = gpu::Problem::new(&ctx, m, n)?;
        let best = tuner.get_or_tune(&stream, &gpu::key(&p), |cfg| gpu::build(&ctx, &tc, &p, cfg))?.clone();
        for t in &best.trials {
            println!("  {m}x{n} {:?}: {:?}", t.config, t.outcome);
        }
        let mut launch = gpu::build(&ctx, &tc, &p, &best.config)?;
        launch(&stream, 0)?;
        stream.synchronize()?;
        p.check()?;
        let gbs = (3 * m * n * 4) as f64 / (best.time_us * 1e-6) / 1e9;
        println!(
            "{m}x{n}: best {:?}, {:.2} us, {gbs:.1} GB/s ({} tunings)",
            best.config,
            best.time_us,
            tuner.tunings()
        );
    }
    Ok(())
}

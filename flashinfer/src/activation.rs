// SPDX-License-Identifier: Apache-2.0

//! `silu_and_mul`, `gelu_and_mul` and `gelu_tanh_and_mul`, after FlashInfer's
//! `act_and_mul_kernel` and `act_and_mul_split_kernel` (`include/flashinfer/activation.cuh`)
//! and their launcher (`flashinfer/jit/activation.py`). This is cute-rs's own code written
//! after the published design; no FlashInfer source was translated.
//!
//! `input` is `[tokens, 2 * d]` (`x` in the first `d` columns, `y` in the last `d`), `out` is
//! `[tokens, d]`, and `out = act(x) * y` with the activation in Float32. Float16 and
//! BFloat16, 16-byte (8-element) vector accesses.
//!
//! * **Geometry** ([`plan`]): `threads = min(d / 8, 256)`. A batch of `tokens >= 2 * SMs`
//!   rows uses one block per row looping over the row. A smaller batch splits each row over
//!   `split` blocks (about two per SM, up to four per SM until each thread has at most two
//!   vectors), and a thread then loads `unroll = 2` vectors of each half together before
//!   computing, as FlashInfer's launcher does. One kernel ([`kernel`]) covers both: the
//!   one-block-per-row kernel is `split = 1, unroll = 1`.
//! * **Activations**: `silu(x) = x / (1 + exp(-x))` with `ex2.approx.ftz` for the
//!   exponential (`__expf`), `gelu(x) = 0.5 x (1 + erf(x / sqrt 2))` with a 1.5e-7 polynomial `erf` (CUDA's
//!   `erff` is far slower here), and `gelu_tanh(x) = 0.5
//!   x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` with `tanh.approx.f32`.
//!
//! Not yet covered: `d` that is not a multiple of 8 (FlashInfer narrows the vector width
//! to 4, 2 or 1), `tokens * 2 * d` beyond 2^31 elements (offsets are 32-bit here), and
//! programmatic dependent launch.
//!
//! `examples/activation.rs` runs it on the simulator and, with `--features cuda` and
//! `GPU=1`, on the GPU against the reference, then benchmarks it.

use cute::prelude::*;

pub use crate::norm::Dtype;

/// Elements per thread per vector access (16 bytes of a 16-bit type).
pub const VEC: usize = 8;

/// Which activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    Silu,
    Gelu,
    GeluTanh,
}

impl Act {
    pub fn name(self) -> &'static str {
        match self {
            Act::Silu => "silu",
            Act::Gelu => "gelu",
            Act::GeluTanh => "gelu_tanh",
        }
    }

    /// The activation in f64 (the reference).
    pub fn reference(self, x: f64) -> f64 {
        match self {
            Act::Silu => x / (1.0 + (-x).exp()),
            Act::Gelu => x * 0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2)),
            Act::GeluTanh => x * 0.5 * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh()),
        }
    }

    fn apply(self, x: Float32) -> Float32 {
        match self {
            Act::Silu => {
                // `__expf(-x)`: `ex2.approx.ftz(-x * log2(e))`.
                let e = cute::math::exp2(x * Float32::from(-std::f32::consts::LOG2_E), true);
                cute::math::div(x, e + Float32::from(1.0f32), true)
            }
            Act::Gelu => {
                let t = erf_approx(x * Float32::from(std::f32::consts::FRAC_1_SQRT_2));
                x * Float32::from(0.5f32) * (t + Float32::from(1.0f32))
            }
            Act::GeluTanh => {
                let inner = Float32::from(0.797_884_6_f32) * (x + Float32::from(0.044715f32) * x * x * x);
                let cdf = Float32::from(0.5f32) * (cute::math::tanh(inner, true) + Float32::from(1.0f32));
                x * cdf
            }
        }
    }
}

/// `erf` to 1.5e-7 (Abramowitz and Stegun 7.1.26): `1 - (a1 t + .. + a5 t^5) e^(-z^2)`,
/// `t = 1 / (1 + p |z|)`, odd. About 20 instructions where the library's `erf` (accurate to
/// 1 ulp, which a 16-bit output cannot use) takes several times that.
fn erf_approx(z: Float32) -> Float32 {
    let c = |v: f32| Float32::from(v);
    let a = select(z.lt(0.0f32), c(0.0) - z, z);
    let t = cute::math::rcp(c(1.0) + c(0.3275911) * a, true);
    let poly =
        ((((c(1.061_405_4) * t + c(-1.453_152)) * t + c(1.421_413_8)) * t + c(-0.284_496_74)) * t + c(0.254_829_6)) * t;
    let y = c(1.0) - poly * cute::math::exp2(a * a * c(-std::f32::consts::LOG2_E), true);
    select(z.lt(0.0f32), c(0.0) - y, y)
}

/// `erf` in f64: the Taylor series (to about 1e-9 for `|x| < 4`), 1 beyond.
fn erf(x: f64) -> f64 {
    if x.abs() >= 4.5 {
        return x.signum();
    }
    let (mut term, mut sum) = (x, x);
    for n in 1..80 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// How a launch is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plan {
    /// Threads per block.
    pub threads: usize,
    /// Blocks per row.
    pub split: usize,
    /// Vectors of each half a thread loads together.
    pub unroll: usize,
}

/// FlashInfer's launch choice for `tokens` rows of `d` outputs on a GPU with `num_sms` SMs.
pub fn plan(tokens: usize, d: usize, num_sms: usize) -> Plan {
    assert!(d >= VEC && d.is_multiple_of(VEC), "d = {d} must be a positive multiple of {VEC}");
    let nv = d / VEC;
    let threads = nv.min(256);
    let (mut split, mut unroll) = (1, 1);
    if tokens > 0 && tokens < 2 * num_sms {
        let spread = 2 * num_sms / tokens;
        let fit = nv.div_ceil(2 * threads).min(4 * num_sms / tokens);
        split = spread.max(fit).min(nv.div_ceil(threads));
        unroll = if nv.div_ceil(threads * split) >= 2 { 2 } else { 1 };
    }
    Plan { threads, split, unroll }
}

/// Parameter order of [`kernel`]: `input, output, d, split`.
pub fn kernel(dtype: Dtype, act: Act, threads: usize, unroll: usize) -> Kernel {
    match dtype {
        Dtype::F16 => kernel_of::<F16>(dtype, act, threads, unroll),
        Dtype::BF16 => kernel_of::<BF16>(dtype, act, threads, unroll),
    }
}

fn kernel_of<T: Numeric>(dtype: Dtype, act: Act, threads: usize, unroll: usize) -> Kernel {
    assert!((1..=1024).contains(&threads) && unroll >= 1);
    let name = format!("flashinfer_{}_and_mul_{}_t{threads}_u{unroll}", act.name(), dtype.name());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<T>("input");
        let output = param_ptr::<T>("output");
        let d = param_scalar::<i32>("d");
        let split = param_scalar::<i32>("split");

        let tid = thread_idx().x;
        let bid = block_idx().x;
        let row = bid / split;
        let chunk = bid - row * split;
        let v = VEC as i32;
        let nv = d / v;
        let nt = threads as i32;
        let stride = nt * split;
        let first = chunk * nt + tid;
        // This thread's vectors are `first + j * stride < nv`; each step takes `unroll`.
        // Offsets in vectors; every access below is `vector * 8`, which cute-rs proves
        // 8-element aligned for the 16-byte accesses.
        let x_row = row * (2 * nv);
        let o_row = row * nv;
        for_range(first, nv, stride * unroll as i32, |at| {
            let cols: Vec<Int32> = (0..unroll).map(|k| at + k as i32 * stride).collect();
            let oks: Vec<Boolean> = cols.iter().map(|c| c.lt(nv)).collect();
            // Every load first (columns past the row read the last vector), then the math.
            #[allow(clippy::type_complexity)]
            let loads: Vec<([Value<T>; VEC], [Value<T>; VEC])> = cols
                .iter()
                .map(|c| {
                    let c = c.min(nv - 1);
                    let x = load_vector::<T, VEC>(&input.offset((x_row + c) * v), LoadOptions::default());
                    let y = load_vector::<T, VEC>(&input.offset((x_row + nv + c) * v), LoadOptions::default());
                    (x, y)
                })
                .collect();
            for ((col, ok), (x, y)) in cols.iter().zip(&oks).zip(&loads) {
                let out: [Value<T>; VEC] =
                    std::array::from_fn(|i| (act.apply(x[i].to::<f32>()) * y[i].to::<f32>()).to::<T>());
                if_(*ok, || store_vector::<T, VEC>(&output.offset((o_row + *col) * v), out, StoreOptions::default()));
            }
        });
    })
}

/// A test problem: `[tokens, 2 * d]` of input, every value exactly representable in `dtype`.
pub struct Problem {
    pub dtype: Dtype,
    pub act: Act,
    pub tokens: usize,
    pub d: usize,
    pub input: Vec<f32>,
    /// The SM count the launch plan assumes.
    pub num_sms: usize,
}

impl Problem {
    pub fn new(dtype: Dtype, act: Act, tokens: usize, d: usize, seed: u64) -> Problem {
        assert!(tokens * 2 * d < 1 << 31, "offsets are 32-bit");
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let input = (0..tokens * 2 * d)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (state >> 33) as f32 / (1u64 << 31) as f32;
                dtype.round((u * 2.0 - 1.0) * 4.0)
            })
            .collect();
        Problem { dtype, act, tokens, d, input, num_sms: 22 }
    }

    pub fn plan(&self) -> Plan {
        plan(self.tokens, self.d, self.num_sms)
    }

    pub fn kernel(&self) -> Kernel {
        let p = self.plan();
        kernel(self.dtype, self.act, p.threads, p.unroll)
    }

    pub fn launch(&self) -> LaunchConfig {
        let p = self.plan();
        LaunchConfig::new([(self.tokens * p.split) as u32, 1, 1], [p.threads as u32, 1, 1])
    }

    /// The kernel's arguments: `input`, `output`, `d`, `split`.
    pub fn args(&self) -> Vec<KernelArg> {
        vec![
            self.dtype.buffer(&self.input),
            self.dtype.buffer(&vec![0.0; self.tokens * self.d]),
            KernelArg::ScalarI32(self.d as i32),
            KernelArg::ScalarI32(self.plan().split as i32),
        ]
    }

    /// `act(x) * y` in f64.
    pub fn reference(&self) -> Vec<f64> {
        let d = self.d;
        let mut out = Vec::with_capacity(self.tokens * d);
        for row in self.input.chunks(2 * d) {
            for k in 0..d {
                out.push(self.act.reference(row[k] as f64) * row[d + k] as f64);
            }
        }
        out
    }

    /// Check `got` against the reference (absolute plus relative, as [`Dtype::tolerance`],
    /// plus the approximate `exp2`/`tanh`); returns the largest scaled error.
    pub fn check(&self, got: &[f32]) -> f64 {
        let tol = self.dtype.tolerance();
        let mut worst = 0.0f64;
        for (i, (&g, r)) in got.iter().zip(self.reference()).enumerate() {
            let e = (g as f64 - r).abs() / (1.0 + r.abs());
            assert!(
                e <= tol,
                "{}: element {i}: {g} against {r} ({} tokens {} d {})",
                self.act.name(),
                self.dtype.name(),
                self.tokens,
                self.d
            );
            worst = worst.max(e);
        }
        worst
    }

    /// Bytes read and written, as `bench_act_and_mul.py` counts them.
    pub fn bytes(&self) -> f64 {
        (self.tokens * 3 * self.d * 2) as f64
    }
}

/// Simulate the kernel on `p` and check it against the f64 reference; returns the largest
/// scaled error.
pub fn simulate(p: &Problem) -> f64 {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&p.dtype.read(&args[1]))
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::collections::HashMap;
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Act, Dtype, Problem};

    /// Kernels compiled and loaded on first use, per `(dtype, act, threads, unroll)`.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<(Dtype, Act, usize, usize), (Kernel, CudaFunction<'c>)>,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> Kernels<'c> {
            Kernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(&mut self, p: &Problem) -> Result<&(Kernel, CudaFunction<'c>), Box<dyn Error>> {
            let plan = p.plan();
            let key = (p.dtype, p.act, plan.threads, plan.unroll);
            if !self.loaded.contains_key(&key) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let k = p.kernel();
                let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                self.loaded.insert(key, (k, f));
            }
            Ok(&self.loaded[&key])
        }
    }

    /// Run `p` once; if `check`, compare with the f64 reference and return the largest
    /// scaled error; if `time`, also return the average GPU time in microseconds of
    /// launches replayed from a CUDA graph.
    pub fn run(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        p: &Problem,
        check: bool,
        time: bool,
    ) -> Result<(f64, Option<f64>), Box<dyn Error>> {
        let (k, f) = ks.get(p)?;
        let host = p.args();
        let mut bufs = Vec::new();
        for a in &host {
            if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => DeviceArg::Scalar(a.clone()),
                _ => DeviceArg::Buffer(bi.next().unwrap()),
            })
            .collect();
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(k, f, p.launch()).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let mut err = 0.0;
        if check {
            let got: Vec<f32> = match p.dtype {
                Dtype::F16 => bufs[1].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                Dtype::BF16 => bufs[1].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect(),
            };
            err = p.check(&got);
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((err, t))
    }
}

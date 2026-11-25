// SPDX-License-Identifier: Apache-2.0

//! POD attention: a prefill batch and a decode batch in one launch, after FlashInfer's
//! `PODWithPagedKVCacheWrapper` (`flashinfer/pod.py`) and its kernel
//! `PODWithKVCacheTensorKernel` (`include/flashinfer/attention/pod.cuh`). This is cute-rs's
//! own code, written after FlashInfer's published design; no FlashInfer source was
//! translated.
//!
//! A serving step that mixes a chunked prefill with ongoing decodes would otherwise launch
//! two kernels; the prefill is compute-bound and the decodes bandwidth-bound, so running
//! their CTAs side by side on the same SMs overlaps the two (POD-Attention, Kamath et al.).
//!
//! * **The two halves** are the [`prefill`](crate::prefill) kernel's per-CTA body
//!   (`prefill::body`), built twice, as FlashInfer's POD kernel calls
//!   `SinglePrefillWithKVCacheDevice` and `BatchPrefillWithPagedKVCacheDevice`:
//!   * the prefill: contiguous (ragged) K/V, FlashInfer's single-prefill plan
//!     ([`prefill::Plan::single`]: query tile from `qo_len * group`, split-KV only when
//!     the tiles leave the GPU idle), causal or not;
//!   * the decode: a paged cache with one query per request, planned by FlashInfer's batch
//!     prefill planner ([`prefill::Plan::new`]), which picks the 16-row tile (1 x 4 warps)
//!     for `group <= 16`: the tensor-core decode FlashInfer's POD uses, never causal.
//!
//!   Both use 128 threads; the launch is `prefill CTAs + decode CTAs` blocks of 128
//!   threads with the larger of the two shared-memory footprints.
//! * **SM-aware scheduler** (FlashInfer's): thread 0 of each CTA reads its SM id and takes
//!   a ticket from a per-SM counter, so the CTAs landing on one SM alternate between the
//!   operations in the ratio of their CTA counts (`1 + decode / prefill` tags, one of them
//!   prefill, or the converse), then takes the next block index of that operation from a
//!   global counter, switching to the other operation once its blocks run out. The block
//!   index and operation go to the other threads through shared memory.
//! * The counters live in a small `i32` buffer. FlashInfer clears it with a `cudaMemset`
//!   before each launch; here the last CTA to finish clears it (a done counter), so a
//!   launch is a single kernel (plus the merges below), also in a CUDA graph.
//! * Split-KV partial states of either half are merged afterwards by the decode port's
//!   [`merge_kernel`](crate::decode::merge_kernel), as FlashInfer's `MergeStates` /
//!   `VariableLengthMergeStates` launches after its POD kernel.
//!
//! Float16 only (Q, K/V and output), head dim 128, no attention variants (soft-cap,
//! window, ALiBi, custom mask) and no FP8 cache, though the prefill body has them; the
//! prefill and decode must share the head counts (as FlashInfer asserts).

use cute::prelude::*;

use crate::prefill::{self, CtaCoords, KernelSpec, Params, Plan};

/// Threads per CTA (every prefill tile shape has four warps).
pub const THREADS: usize = 128;
/// Per-SM ticket counters in the scheduler buffer (SM ids are taken modulo this).
pub const SM_SLOTS: usize = 256;
/// Scheduler buffer entries: the next prefill and decode block, the done counter, and
/// [`SM_SLOTS`] per-SM tickets.
pub const SCHED_LEN: usize = 3 + SM_SLOTS;
/// Arguments per half (the prefill kernel's, [`prefill::p`]).
pub const HALF_ARGS: usize = 25;

/// Parameter positions of the POD kernel: the prefill half's prefill-kernel arguments
/// (`PREFILL + prefill::p::*`), the decode half's (`DECODE + prefill::p::*`), then the
/// scheduler buffer and the two halves' CTA counts per KV head.
pub mod p {
    pub const PREFILL: usize = 0;
    pub const DECODE: usize = super::HALF_ARGS;
    pub const SCHED: usize = 2 * super::HALF_ARGS;
    pub const PREFILL_CTAS: usize = SCHED + 1;
    pub const DECODE_CTAS: usize = SCHED + 2;
}

/// What a POD kernel is built for: the prefill and decode halves' prefill-kernel specs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PodSpec {
    pub prefill: KernelSpec,
    pub decode: KernelSpec,
}

impl PodSpec {
    pub fn name(&self) -> String {
        let strip = |s: String| s.trim_start_matches("flashinfer_batch_prefill_").to_string();
        format!("flashinfer_pod_{}__{}", strip(self.prefill.name()), strip(self.decode.name()))
    }

    /// Shared memory per CTA, in bytes: the larger half's.
    pub fn smem_bytes(&self) -> usize {
        self.prefill.tile.smem_bytes().max(self.decode.tile.smem_bytes())
    }
}

/// The POD kernel for `spec`; parameters in the order of [`p`].
pub fn kernel(spec: &PodSpec) -> Kernel {
    for s in [&spec.prefill, &spec.decode] {
        assert_eq!(s.tile.kv, prefill::KvDtype::F16, "POD is built for a Float16 cache");
        assert_eq!(s.variant, prefill::Variant::NONE, "POD is built without attention variants");
    }
    let (sp, sd) = (spec.prefill, spec.decode);
    let elems = (spec.smem_bytes() / 2).max(2 * 2);
    Kernel::builder(spec.name()).block_dim([THREADS as u32, 1, 1]).build(|| {
        let pp = Params::declare::<F16>("p_");
        let dp = Params::declare::<F16>("d_");
        let sched = param_ptr::<i32>("sched");
        let prefill_ctas = param_scalar::<i32>("prefill_ctas");
        let decode_ctas = param_scalar::<i32>("decode_ctas");
        let smem = make_smem_tensor_aligned::<F16>(Layout::col_major(elems), 16);
        let shared = recast_ptr::<i32, F16>(&smem.ptr(), None);
        let tid = thread_idx().x;
        let prefill_blocks = prefill_ctas * pp.num_kv_heads;
        let decode_blocks = decode_ctas * dp.num_kv_heads;
        let total = prefill_blocks + decode_blocks;

        // ---- The scheduler (thread 0): an operation for this CTA, from its SM's ticket,
        // then that operation's next block (or the other's, once it runs out).
        if_(tid.eq(0), || {
            let ticket = atomic_add(&sched.offset(3 + smid() % SM_SLOTS as i32), 1);
            let decode_first = prefill_blocks.le(decode_blocks);
            // prefill <= decode: 1 + decode / prefill tags, the first prefill; otherwise
            // prefill / decode prefill tags, then one decode.
            let few = prefill_blocks.min(decode_blocks);
            let tags = select(decode_first, decode_blocks / few + 1, prefill_blocks / few + 1);
            let tag = ticket % tags;
            let op = select(decode_first, select(tag.gt(0), Int32::from(1), Int32::from(0)), {
                select(tag.lt(tags - 1), Int32::from(0), Int32::from(1))
            });
            let bid = atomic_add(&sched.offset(op), 1);
            let limit = select(op.eq(0), prefill_blocks, decode_blocks);
            let state = Var::new((op, bid));
            if_(bid.ge(limit), || {
                let other = Int32::from(1) - op;
                state.set((other, atomic_add(&sched.offset(other), 1)));
            });
            let (op, bid) = state.get();
            shared.store_at(0, bid);
            shared.store_at(1, op);
        });
        sync_threads();
        let bid = shared.load_at(0);
        let op = shared.load_at(1);
        sync_threads();

        // ---- The CTA's work: plan entry `bid % ctas` of KV head `bid / ctas`.
        let coords = |spec: &KernelSpec, ctas: Int32| {
            let warp = tid / 32;
            let nwq = spec.tile.num_warps_q as i32;
            CtaCoords { lane: tid % 32, wq: warp % nwq, wkv: warp / nwq, kv_head: bid / ctas, cta: bid % ctas }
        };
        if_else(
            op.eq(0),
            || prefill::body::<F16>(&sp, &pp, &smem, coords(&sp, prefill_ctas)),
            || prefill::body::<F16>(&sd, &dp, &smem, coords(&sd, decode_ctas)),
        );

        // ---- The last CTA to finish clears the scheduler buffer for the next launch.
        sync_threads();
        if_(tid.eq(0), || {
            let done = atomic_add(&sched.offset(2), 1);
            shared.store_at(0, select(done.eq(total - 1), Int32::from(1), Int32::from(0)));
        });
        sync_threads();
        if_(shared.load_at(0).ne(0), || {
            for i in 0..SCHED_LEN.div_ceil(THREADS) {
                let at = tid + (i * THREADS) as i32;
                if_(at.lt(SCHED_LEN as i32), || {
                    let ptr = sched.offset(at);
                    let v = atomic_add(&ptr, 0);
                    atomic_add(&ptr, Int32::from(0) - v);
                });
            }
        });
    })
}

/// A POD problem: a prefill batch over contiguous K/V (one request for FlashInfer's
/// `PODWithPagedKVCacheWrapper`) and a decode batch over a paged cache, with the same head
/// counts.
pub struct Problem {
    /// Ragged (contiguous K/V); causal or not.
    pub prefill: prefill::Problem,
    /// Paged, one query per request, not causal.
    pub decode: prefill::Problem,
}

/// The two halves' plans.
#[derive(Clone, Debug)]
pub struct Plans {
    pub prefill: Plan,
    pub decode: Plan,
}

impl Plans {
    /// CTAs per KV head of the prefill and decode halves.
    pub fn ctas(&self) -> (usize, usize) {
        (self.prefill.num_ctas(), self.decode.num_ctas())
    }
}

impl Problem {
    /// Random inputs: a prefill of `qo_len_p` queries over `kv_len_p` tokens (causal if
    /// `causal_p`), and decodes over `decode_kv_lens` tokens in pages of `page_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qo_len_p: usize,
        kv_len_p: usize,
        decode_kv_lens: &[usize],
        page_size: usize,
        num_kv_heads: usize,
        group: usize,
        causal_p: bool,
        seed: u64,
    ) -> Problem {
        let prefill = prefill::Problem::new_ragged(&[qo_len_p], &[kv_len_p], num_kv_heads, group, causal_p, seed);
        let decode = prefill::Problem::new(
            &vec![1; decode_kv_lens.len()],
            decode_kv_lens,
            page_size,
            num_kv_heads,
            group,
            false,
            seed + 1,
        );
        Problem::from_parts(prefill, decode)
    }

    /// A POD problem from its halves (checked: ragged prefill, paged one-query decodes, the
    /// same head counts and Float16 without variants).
    pub fn from_parts(prefill: prefill::Problem, decode: prefill::Problem) -> Problem {
        assert!(!prefill.qo_lens.is_empty() && !decode.qo_lens.is_empty(), "both halves need a request");
        assert!(prefill.is_ragged(), "the prefill half reads contiguous K/V");
        assert!(
            !decode.is_ragged() && decode.qo_lens.iter().all(|&l| l == 1),
            "the decode half is paged, one query each"
        );
        assert!(!decode.causal, "the decode half is not causal");
        assert_eq!((prefill.num_kv_heads, prefill.group), (decode.num_kv_heads, decode.group), "head counts differ");
        for h in [&prefill, &decode] {
            assert_eq!(h.kv_dtype, prefill::KvDtype::F16, "POD is Float16 only");
            assert_eq!(h.variant, prefill::Variant::NONE, "POD has no attention variants");
        }
        Problem { prefill, decode }
    }

    /// FlashInfer's plans for `max_grid` concurrent CTAs: the single-prefill plan (the
    /// batch prefill plan if the prefill has several requests) and the batch prefill
    /// planner's for the decodes.
    pub fn plans(&self, max_grid: usize) -> Plans {
        self.plans_with_min_chunk(max_grid, (128 / self.decode.page_tokens()).max(1))
    }

    /// [`Problem::plans`] with another smallest decode chunk, in pages (tests use 1).
    pub fn plans_with_min_chunk(&self, max_grid: usize, min_chunk_pages: usize) -> Plans {
        let prefill = if self.prefill.qo_lens.len() == 1 {
            self.prefill.single_plan(max_grid)
        } else {
            self.prefill.plan(max_grid)
        };
        Plans { prefill, decode: self.decode.plan_with_min_chunk(max_grid, min_chunk_pages) }
    }

    pub fn spec(&self, plans: &Plans) -> PodSpec {
        PodSpec { prefill: self.prefill.spec(&plans.prefill), decode: self.decode.spec(&plans.decode) }
    }

    /// One CTA of [`THREADS`] per (plan entry, KV head) of either half.
    pub fn launch(&self, plans: &Plans) -> LaunchConfig {
        let (cp, cd) = plans.ctas();
        LaunchConfig::new([((cp + cd) * self.prefill.num_kv_heads) as u32, 1, 1], [THREADS as u32, 1, 1])
    }

    /// The POD kernel's arguments, in the order of [`p`] (a zeroed scheduler buffer).
    pub fn args(&self, plans: &Plans) -> Vec<KernelArg> {
        let mut args = self.prefill.args(&plans.prefill);
        args.extend(self.decode.args(&plans.decode));
        assert_eq!(args.len(), 2 * HALF_ARGS);
        let (cp, cd) = plans.ctas();
        args.push(KernelArg::i32(vec![0; SCHED_LEN]));
        args.push(KernelArg::ScalarI32(cp as i32));
        args.push(KernelArg::ScalarI32(cd as i32));
        args
    }

    /// The f64 references of both halves: `((o_p, lse_p), (o_d, lse_d))`.
    #[allow(clippy::type_complexity)]
    pub fn reference(&self) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
        (self.prefill.reference(), self.decode.reference())
    }

    /// Floating-point operations of both halves (`flashinfer.testing.attention_flops`).
    pub fn flops(&self) -> f64 {
        self.prefill.flops() + self.decode.flops()
    }

    /// Bytes of Q and K/V read (both halves), as FlashInfer's POD benchmark counts them.
    pub fn bytes(&self) -> f64 {
        let hd = prefill::HEAD_DIM * 2;
        let kv = |h: &prefill::Problem| h.kv_lens.iter().sum::<usize>() * h.num_kv_heads * hd * 2;
        let q = |h: &prefill::Problem| h.total_q() * h.num_qo_heads() * hd;
        (kv(&self.prefill) + kv(&self.decode) + q(&self.prefill) + q(&self.decode)) as f64
    }
}

/// The two halves' outputs: `o` (as f32) and `lse` of the prefill and of the decodes.
pub type Outputs = ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>));

/// Run `p` under `plans` on the simulator (the POD kernel, then the merges of any split
/// half); returns both halves' outputs.
pub fn execute(p: &Problem, plans: &Plans) -> Outputs {
    let mut args = p.args(plans);
    let k = kernel(&p.spec(plans));
    k.simulate(p.launch(plans), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    let merge = crate::decode::merge_kernel();
    let mut half = |h: &prefill::Problem, plan: &Plan, base: usize| {
        let a = &mut args[base..base + HALF_ARGS];
        if plan.split {
            let mut margs = prefill::merge_args(h, a);
            merge.simulate(h.merge_launch(), &mut margs).unwrap_or_else(|e| panic!("merge: {e}"));
            a[prefill::p::O] = margs[crate::decode::mp::O].clone();
            a[prefill::p::LSE] = margs[crate::decode::mp::LSE].clone();
        }
        let o: Vec<f32> = a[prefill::p::O].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect();
        (o, a[prefill::p::LSE].as_f32().to_vec())
    };
    let outp = half(&p.prefill, &plans.prefill, p::PREFILL);
    let outd = half(&p.decode, &plans.decode, p::DECODE);
    (outp, outd)
}

/// The largest errors of `out` against the reference: `[o_p, lse_p, o_d, lse_d]`.
pub fn errors(p: &Problem, out: &Outputs) -> [f32; 4] {
    let ((rop, rlp), (rod, rld)) = p.reference();
    let ((op, lp), (od, ld)) = out;
    use crate::decode::max_error;
    [max_error(op, &rop), max_error(lp, &rlp), max_error(od, &rod), max_error(ld, &rld)]
}

/// Simulate `p` for `max_grid` concurrent CTAs (smallest decode chunk `min_chunk_pages`)
/// and check it: the plans and `[o_p, lse_p, o_d, lse_d]` errors.
pub fn simulate(p: &Problem, max_grid: usize, min_chunk_pages: usize) -> (Plans, [f32; 4]) {
    let plans = p.plans_with_min_chunk(max_grid, min_chunk_pages);
    let out = execute(p, &plans);
    (plans.clone(), errors(p, &out))
}

/// Running on the GPU (`--features cuda`): the POD kernel and, for comparison, the same
/// two halves as two prefill-kernel launches.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::error::Error;
    use std::rc::Rc;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{HALF_ARGS, Outputs, Plans, PodSpec, Problem, kernel};
    use crate::prefill;

    type Loaded<'c> = Rc<(Kernel, CudaFunction<'c>)>;

    /// POD kernels compiled on first use, and the prefill kernels (with the merge) for the
    /// two-launch comparison.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: Toolchain,
        opts: CompileOptions,
        cache: RefCell<HashMap<PodSpec, Loaded<'c>>>,
        pub prefill: prefill::gpu::Kernels<'c>,
        /// CTAs the planners assume run at once: 2 per SM, as FlashInfer's.
        pub max_grid: usize,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(&tc).ok_or("no target for this GPU")?;
            let opts = CompileOptions::new().gpu_arch(arch);
            let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            let prefill = prefill::gpu::Kernels::new(ctx, tc.clone())?;
            Ok(Kernels { ctx, tc, opts, cache: RefCell::new(HashMap::new()), prefill, max_grid: 2 * sms })
        }

        /// The POD kernel for `spec`, compiled on first use.
        pub fn get(&self, spec: &PodSpec) -> Result<Loaded<'c>, Box<dyn Error>> {
            if let Some(k) = self.cache.borrow().get(spec) {
                return Ok(k.clone());
            }
            let k = kernel(spec);
            let f = self.ctx.load(&k.compile_with(&self.tc, &self.opts)?)?;
            let loaded = Rc::new((k, f));
            self.cache.borrow_mut().insert(*spec, loaded.clone());
            Ok(loaded)
        }
    }

    /// GPU times (microseconds, CUDA-graph replays): the POD launch (with its merges), and
    /// the same halves as two prefill-kernel launches (each with its merge).
    #[derive(Clone, Copy, Debug)]
    pub struct Times {
        pub pod: f64,
        pub two_launches: f64,
    }

    /// Run `p` once on the GPU under FlashInfer's plans; if `check`, compare both halves
    /// with the f64 reference (`[o_p, lse_p, o_d, lse_d]` errors); if `time`, time the POD
    /// launch and the two-launch equivalent, `iterations` replays each.
    #[allow(clippy::type_complexity)]
    pub fn run(
        ks: &Kernels<'_>,
        p: &Problem,
        check: bool,
        time: Option<usize>,
    ) -> Result<(Plans, [f32; 4], Option<Times>), Box<dyn Error>> {
        let plans = p.plans(ks.max_grid);
        let (out, t) = launch(ks, p, &plans, check, time)?;
        let errs = match out {
            Some(out) => super::errors(p, &out),
            None => [0.0; 4],
        };
        Ok((plans, errs, t))
    }

    /// Upload `p`, run the POD kernel once under `plans`, download both halves' outputs if
    /// `fetch`, and time it (and the two-launch equivalent) if `time` is `Some(iterations)`.
    pub fn launch(
        ks: &Kernels<'_>,
        p: &Problem,
        plans: &Plans,
        fetch: bool,
        time: Option<usize>,
    ) -> Result<(Option<Outputs>, Option<Times>), Box<dyn Error>> {
        let ctx = ks.ctx;
        let spec = p.spec(plans);
        let loaded = ks.get(&spec)?;
        let host = p.args(plans);
        let mut bufs = Vec::new();
        for a in &host {
            bufs.push(match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => None,
                _ => Some(ctx.upload(a)?),
            });
        }
        let dev = |i: usize| match &bufs[i] {
            Some(b) => DeviceArg::Buffer(b),
            None => DeviceArg::Scalar(host[i].clone()),
        };
        let args: Vec<DeviceArg> = (0..host.len()).map(dev).collect();
        // A half's merge arguments (its split-KV scratch into its `o` and `lse`).
        use prefill::p as pp;
        let merge_args = |base: usize, h: &prefill::Problem| {
            vec![
                dev(base + pp::TMP_O),
                dev(base + pp::TMP_LSE),
                dev(base + pp::MERGE_INDPTR),
                dev(base + pp::O),
                dev(base + pp::LSE),
                DeviceArg::Scalar(KernelArg::ScalarI32(h.num_qo_heads() as i32)),
            ]
        };
        let halves = [(&p.prefill, &plans.prefill, super::p::PREFILL), (&p.decode, &plans.decode, super::p::DECODE)];
        let margs: Vec<Vec<DeviceArg>> = halves.iter().map(|(h, _, base)| merge_args(*base, h)).collect();
        let merge = &ks.prefill.merge;
        let merges = |s: &cute::cuda::CudaStream<'_>| -> Result<(), Box<dyn Error>> {
            for ((h, plan, _), ma) in halves.iter().zip(&margs) {
                if plan.split {
                    ctx.launch_builder(&merge.0, &merge.1, h.merge_launch()).stream(s).args(ma).submit()?;
                }
            }
            Ok(())
        };
        let mut pod = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&loaded.0, &loaded.1, p.launch(plans)).stream(s).args(&args).submit()?;
            merges(s)
        };
        let stream = ctx.stream()?;
        pod(&stream, 0)?;
        stream.synchronize()?;
        let out = if fetch {
            let get = |base: usize| -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
                let o = bufs[base + pp::O].as_ref().unwrap().to_vec::<F16>()?;
                let lse = bufs[base + pp::LSE].as_ref().unwrap().to_vec::<f32>()?;
                Ok((o.into_iter().map(|x| x.to_f32()).collect(), lse))
            };
            Some((get(super::p::PREFILL)?, get(super::p::DECODE)?))
        } else {
            None
        };
        let t = match time {
            Some(iterations) => {
                let opts = BenchmarkOptions::new().warmup_iterations(3).iterations(iterations).use_cuda_graphs(true);
                let pod_t = benchmark(&stream, &opts, &mut pod)?;
                let kp = ks.prefill.get(&spec.prefill)?;
                let kd = ks.prefill.get(&spec.decode)?;
                let two = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
                    for ((h, plan, base), k) in halves.iter().zip([&kp, &kd]) {
                        ctx.launch_builder(&k.0, &k.1, h.launch(plan))
                            .stream(s)
                            .args(&args[*base..*base + HALF_ARGS])
                            .submit()?;
                    }
                    merges(s)
                };
                let two_t = benchmark(&stream, &opts, two)?;
                Some(Times { pod: pod_t, two_launches: two_t })
            }
            None => None,
        };
        Ok((out, t))
    }
}

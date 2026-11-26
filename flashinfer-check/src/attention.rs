// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::attention` (the port of `BatchAttention`) on the inputs of `reference/cases_attention.py`: mixed
//! prefill and decode requests over a paged cache. The port's `gpu::run` returns only an error figure, so
//! the launch is built here from its public pieces.

use cute::cuda::DeviceArg;
use flashinfer::attention::gpu::Kernels;
use flashinfer::attention::{AttentionSpec, Problem, p as pp};
use flashinfer::prefill::Problem as Batch;

use crate::case::{Case, Gpu, Res, Tensor};
use cute::prelude::*;

fn f16s(t: &Tensor) -> Vec<F16> {
    t.f32s().into_iter().map(F16::from_f32).collect()
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let (hq, hkv, ps) = (case.int("hq"), case.int("hkv"), case.int("page_size"));
    let (qo_lens, kv_lens) = (case.ints("qo_lens"), case.ints("kv_lens"));
    let mut b = Batch::new(&qo_lens, &kv_lens, ps, hkv, hq / hkv, case.flag("causal"), 0);
    // The page table, lengths and cache contents are FlashInfer's own; its cache holds K then V per page.
    b.q = f16s(&case.tensor("q"));
    b.kv_indptr = case.tensor("kv_indptr").i32s();
    b.kv_indices = case.tensor("kv_indices").i32s();
    b.kv_last_page_len = case.tensor("last").i32s();
    let cache = f16s(&case.tensor("cache"));
    let page = ps * hkv * 128;
    let (mut kc, mut vc) = (Vec::new(), Vec::new());
    for pg in cache.chunks(2 * page) {
        kc.extend_from_slice(&pg[..page]);
        vc.extend_from_slice(&pg[page..]);
    }
    (b.k_cache, b.v_cache) = (kc, vc);
    let p = Problem::from_batch(b);

    let ks = Kernels::new(&gpu.ctx, gpu.tc.clone())?;
    let plan = p.plan(ks.num_sm);
    let loaded = ks.get(&AttentionSpec { alternate: ks.alternate, ..p.spec(&plan) })?;
    let host = p.args(&plan);
    let bufs: Vec<_> = host
        .iter()
        .map(|a| match a {
            KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => Ok(None),
            _ => gpu.ctx.upload(a).map(Some),
        })
        .collect::<Result<_, _>>()?;
    let args: Vec<DeviceArg> = host
        .iter()
        .zip(&bufs)
        .map(|(a, buf)| match buf {
            Some(b) => DeviceArg::Buffer(b),
            None => DeviceArg::Scalar(a.clone()),
        })
        .collect();
    let stream = gpu.ctx.stream()?;
    gpu.ctx.launch_builder(&loaded.0, &loaded.1, plan.launch()).stream(&stream).args(&args).submit()?;
    stream.synchronize()?;
    let o: Vec<f32> = bufs[pp::O].as_ref().unwrap().to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect();
    case.write_f32("o", &o)?;
    case.write_f32("lse", &bufs[pp::LSE].as_ref().unwrap().to_vec::<f32>()?)
}

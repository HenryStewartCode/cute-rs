// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::mla` on the inputs of `reference/cases_mla.py`.
//!
//! The port's `Problem` makes its own random data and page table; a case builds one for its
//! shape (the port's planner then makes the work list) and replaces the queries, cache and
//! page table with the case's bytes.

use cute::cuda::DeviceArg;
use cute::prelude::*;
use flashinfer::mla::gpu::Kernels;
use flashinfer::mla::{Problem, p as arg};

use crate::case::{Case, Gpu, Res, Tensor};

fn f16_arg(t: &Tensor) -> KernelArg {
    KernelArg::F16(t.bytes.chunks(2).map(|b| F16(u16::from_le_bytes([b[0], b[1]]))).collect())
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let mut p = Problem::new(
        &case.ints("qo_lens"),
        &case.ints("kv_lens"),
        case.int("page_size"),
        case.int("heads"),
        case.flag("causal"),
        1,
    );
    p.sm_scale = case.float("sm_scale") as f32;
    let ks = Kernels::new(&gpu.ctx, gpu.tc.clone())?;
    let plan = p.plan(ks.num_sm);
    let loaded = ks.get(&p.spec())?;
    let mut host = p.args(&plan);
    host[arg::Q_NOPE] = f16_arg(&case.tensor("q_nope"));
    host[arg::Q_PE] = f16_arg(&case.tensor("q_pe"));
    host[arg::CKV] = f16_arg(&case.tensor("ckv"));
    host[arg::KPE] = f16_arg(&case.tensor("kpe"));
    host[arg::KV_INDICES] = KernelArg::i32(case.tensor("kv_indices").i32s());

    let ctx = &gpu.ctx;
    let mut bufs = Vec::new();
    for a in &host {
        if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
            bufs.push(ctx.upload(a)?);
        }
    }
    let mut args: Vec<DeviceArg> = bufs.iter().map(DeviceArg::Buffer).collect();
    args.push(DeviceArg::Scalar(KernelArg::ScalarF32(p.sm_scale * std::f32::consts::LOG2_E)));
    let stream = ctx.stream()?;
    ctx.launch_builder(&loaded.0, &loaded.1, plan.launch()).stream(&stream).args(&args).submit()?;
    stream.synchronize()?;
    let o: Vec<u8> = bufs[arg::O].to_vec::<u16>()?.into_iter().flat_map(u16::to_le_bytes).collect();
    case.write("o", &o)?;
    case.write_f32("lse", &bufs[arg::LSE].to_vec::<f32>()?)
}

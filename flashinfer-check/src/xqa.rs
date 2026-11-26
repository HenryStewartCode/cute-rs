// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::xqa` on the inputs of `reference/cases_xqa.py`.
//!
//! The port's `Problem` makes its own random data; a case builds one for its shape and then
//! replaces the query, cache, page table, sequence lengths and sinks with the case's bytes.

use cute::cuda::DeviceArg;
use cute::prelude::*;
use flashinfer::xqa::gpu::XqaKernel;
use flashinfer::xqa::{CacheDtype, Config, InputDtype, Problem, p as arg};

use crate::case::{Case, Gpu, Res, Tensor};

fn half_arg(t: &Tensor, bf16: bool) -> KernelArg {
    let bits = t.bytes.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    if bf16 {
        KernelArg::buffer(&bits.map(BF16).collect::<Vec<_>>())
    } else {
        KernelArg::buffer(&bits.map(F16).collect::<Vec<_>>())
    }
}

fn cache_arg(t: &Tensor, kv: CacheDtype, bf16: bool) -> KernelArg {
    match kv {
        CacheDtype::Input => half_arg(t, bf16),
        CacheDtype::E4M3 => KernelArg::buffer(&t.bytes.iter().map(|&b| F8E4M3FN(b)).collect::<Vec<_>>()),
    }
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let bf16 = case.str("dtype") == "bf16";
    let input = if bf16 { InputDtype::BF16 } else { InputDtype::F16 };
    let cache = if case.str("kv_dtype") == "e4m3" { CacheDtype::E4M3 } else { CacheDtype::Input };
    let mut cfg = Config::new(input, cache, case.int("group"), case.int("page_size"));
    let window = case.int("window");
    cfg.sliding_window = window > 0;
    cfg.sinks = case.flag("sinks");
    let table = case.tensor("page_table");
    let lens = case.ints("seq_lens");
    let mut p = Problem::new(cfg, &lens, case.int("kv_heads"), 1)
        .with_scales(case.float("q_scale") as f32, case.float("kv_scale") as f32);
    if window > 0 {
        p = p.with_window(window);
    }
    p.max_pages = table.shape[1];

    let k = XqaKernel::new(&gpu.ctx, &gpu.tc, cfg)?;
    let nb_sub = p.nb_sub(k.sm_count);
    let mut host = p.args(nb_sub);
    host[arg::Q] = half_arg(&case.tensor("q"), bf16);
    host[arg::K] = cache_arg(&case.tensor("k_cache"), cache, bf16);
    host[arg::V] = cache_arg(&case.tensor("v_cache"), cache, bf16);
    host[arg::PAGE_TABLE] = KernelArg::i32(table.i32s());
    host[arg::SEQ_LENS] = KernelArg::i32(case.tensor("seq_lens").i32s());
    if cfg.sinks {
        host[arg::SINKS] = KernelArg::f32(case.tensor("sinks").f32s());
    }

    let ctx = &gpu.ctx;
    let mut bufs = Vec::new();
    for a in &host {
        if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
            bufs.push(ctx.upload(a)?);
        }
    }
    let mut next = bufs.iter();
    let args: Vec<DeviceArg> = host
        .iter()
        .map(|a| match a {
            KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => DeviceArg::Scalar(a.clone()),
            _ => DeviceArg::Buffer(next.next().unwrap()),
        })
        .collect();
    let stream = ctx.stream()?;
    ctx.launch_builder(&k.kernel, &k.func, p.launch(nb_sub)).stream(&stream).args(&args).submit()?;
    stream.synchronize()?;
    let out: Vec<u8> = bufs[arg::OUT].to_vec::<u16>()?.into_iter().flat_map(u16::to_le_bytes).collect();
    case.write("out", &out)
}

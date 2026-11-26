// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::decode` on the inputs of `reference/cases_decode.py`: batch decode over a paged
//! cache (plan, decode kernel, merge) and single decode over contiguous K/V.
//!
//! The port's `Problem` makes its own random data and page table, so a case builds one for its
//! shape and then replaces every buffer FlashInfer was given (query, cache, page table) with the
//! case's bytes; the port's own plan decides the split.

use cute::cuda::DeviceArg;
use cute::prelude::*;
use flashinfer::decode::gpu::Kernels;
use flashinfer::decode::{KvDtype, Problem, STAGES, mp, p as arg};

use crate::case::{Case, Gpu, Res, Tensor};

/// The raw 16-bit elements of `t` as a buffer argument of the case's query type.
fn half_arg(t: &Tensor, bf16: bool) -> KernelArg {
    let bits = t.bytes.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    if bf16 {
        KernelArg::buffer(&bits.map(BF16).collect::<Vec<_>>())
    } else {
        KernelArg::buffer(&bits.map(F16).collect::<Vec<_>>())
    }
}

fn cache_arg(t: &Tensor, kv: KvDtype, bf16: bool) -> KernelArg {
    match kv {
        KvDtype::F16 => half_arg(t, bf16),
        KvDtype::E4M3 => KernelArg::buffer(&t.bytes.iter().map(|&b| F8E4M3FN(b)).collect::<Vec<_>>()),
        KvDtype::E5M2 => KernelArg::buffer(&t.bytes.iter().map(|&b| F8E5M2(b)).collect::<Vec<_>>()),
    }
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    let hd = case.int("head_dim");
    let hkv = case.int("kv_heads");
    let bf16 = case.str("dtype") == "bf16";
    let kv = match case.str("kv_dtype") {
        "e4m3" => KvDtype::E4M3,
        "e5m2" => KvDtype::E5M2,
        _ => KvDtype::F16,
    };
    let single = case.str("op") == "single";
    let lens = case.ints("kv_lens");
    let mut p = if single {
        Problem::new_ragged_hd(&lens, hkv, hd, 1)
    } else {
        Problem::new_hd(&lens, case.int("page_size"), hkv, hd, 1)
    };
    p.bf16 = bf16;
    p.kv_dtype = kv;
    p.k_scale = case.float("k_scale") as f32;
    p.v_scale = case.float("v_scale") as f32;
    p.sm_scale = case.float("sm_scale") as f32;
    let cap = case.float("soft_cap") as f32;
    if cap > 0.0 {
        p = p.with_soft_cap(cap);
    }
    if case.i64("window") >= 0 {
        p = p.with_window(case.i64("window") as i32);
    }
    if case.flag("alibi") {
        p = p.with_alibi();
    }

    let ks = Kernels::with_shape(&gpu.ctx, &gpu.tc, STAGES, single, kv, p.variant, hd, bf16)?;
    let plan = if single { p.single_plan(ks.max_grid) } else { p.plan(ks.max_grid) };
    let mut host = p.args(&plan);
    host[arg::Q] = half_arg(&case.tensor("q"), bf16);
    if single {
        host[arg::K] = cache_arg(&case.tensor("k"), kv, bf16);
        host[arg::V] = cache_arg(&case.tensor("v"), kv, bf16);
    } else {
        host[arg::K] = cache_arg(&case.tensor("k_cache"), kv, bf16);
        host[arg::V] = cache_arg(&case.tensor("v_cache"), kv, bf16);
        host[arg::KV_INDPTR] = KernelArg::i32(case.tensor("kv_indptr").i32s());
        host[arg::KV_INDICES] = KernelArg::i32(case.tensor("kv_indices").i32s());
        host[arg::LAST_PAGE_LEN] = KernelArg::i32(case.tensor("kv_last_page_len").i32s());
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
    let (dk, df) = if plan.split { (&ks.split.0, &ks.split.1) } else { (&ks.decode.0, &ks.decode.1) };
    ctx.launch_builder(dk, df, p.launch(&plan)).stream(&stream).args(&args).submit()?;
    if plan.split {
        // The merge kernel's parameters, in the order of `mp`.
        let merge = [
            DeviceArg::Buffer(&bufs[arg::TMP_O]),
            DeviceArg::Buffer(&bufs[arg::TMP_LSE]),
            DeviceArg::Buffer(&bufs[arg::O_INDPTR]),
            DeviceArg::Buffer(&bufs[arg::O]),
            DeviceArg::Buffer(&bufs[arg::LSE]),
            DeviceArg::Scalar(KernelArg::ScalarI32(p.num_qo_heads() as i32)),
        ];
        assert_eq!((mp::TMP_O, mp::LSE, mp::NUM_QO_HEADS), (0, 4, 5));
        ctx.launch_builder(&ks.merge.0, &ks.merge.1, p.merge_launch()).stream(&stream).args(&merge).submit()?;
    }
    stream.synchronize()?;
    let o: Vec<u8> = bufs[arg::O].to_vec::<u16>()?.into_iter().flat_map(u16::to_le_bytes).collect();
    case.write("o", &o)?;
    let lse = bufs[arg::LSE].to_vec::<f32>()?;
    case.write_f32("lse", &lse)
}

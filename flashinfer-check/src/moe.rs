// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::moe` on the inputs of `reference/cases_moe.py`: the fused MoE layer and the
//! DeepSeek-V3 routing kernel.

use cute::cuda::DeviceArg;
use cute::prelude::*;
use flashinfer::moe::{MoeShape, Problem, RouteProblem, Routing, ScoreType, buf};

use crate::case::{Case, Gpu, Res, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    match case.str("op") {
        "fused" => fused(gpu, case),
        "route_deepseek" => route(gpu, case),
        other => Err(format!("op {other}").into()),
    }
}

fn fused(gpu: &Gpu, case: &Case) -> Res<()> {
    use flashinfer::moe::Dtype;
    let dtype = match case.str("dtype") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("dtype {other}").into()),
    };
    let shape = MoeShape::new(case.int("hidden"), case.int("inter"), case.int("experts"), case.int("top_k"));
    // The problem picks the GEMM tiles for this shape; its own random data is replaced by the file's.
    let mut p = Problem::new(dtype, shape, case.int("tokens"), 0);
    p.x = case.tensor("x").f32s();
    p.w1 = case.tensor("w1").f32s();
    p.w2 = case.tensor("w2").f32s();
    p.selected = case.tensor("selected").i32s();
    p.scales = case.tensor("scales").f32s();

    let arch = gpu.ctx.best_target(&gpu.tc).ok_or("no target for this GPU")?;
    let opts = CompileOptions::new().gpu_arch(arch);
    let bufs = p.buffers().iter().map(|a| gpu.ctx.upload(a)).collect::<Result<Vec<_>, _>>()?;
    let stream = gpu.ctx.stream()?;
    for step in p.pipeline() {
        let f = gpu.ctx.load(&step.kernel.compile_with(&gpu.tc, &opts)?)?;
        let mut args: Vec<DeviceArg> = step.buffers.iter().map(|&b| DeviceArg::Buffer(&bufs[b])).collect();
        args.extend(step.scalars.iter().cloned().map(DeviceArg::Scalar));
        gpu.ctx.launch_builder(&step.kernel, &f, step.launch).stream(&stream).args(&args).submit()?;
    }
    stream.synchronize()?;
    let out: Vec<u16> = bufs[buf::OUT].to_vec::<u16>()?;
    case.write("out", &out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
}

fn route(gpu: &Gpu, case: &Case) -> Res<()> {
    let score = match case.str("dtype") {
        "f32" => ScoreType::F32,
        "bf16" => ScoreType::BF16,
        other => return Err(format!("dtype {other}").into()),
    };
    let (tokens, experts, k) = (case.int("tokens"), case.int("experts"), case.int("top_k"));
    let routing = Routing::DeepSeekV3 { n_group: case.int("n_group"), topk_group: case.int("topk_group") };
    let mut p = RouteProblem::new(routing, score, tokens, experts, k, 0);
    p.logits = case.tensor("logits").f32s();
    p.bias = case.tensor("bias").f32s();
    p.routed_scaling_factor = case.float("scaling") as f32;
    let args = launch(gpu, &p.kernel(), p.launch(), p.args())?;
    let mut idx = args[2].to_vec::<i32>();
    let mut w = args[3].to_vec::<f32>();
    // Each row in expert order, with its weights, as the reference side does.
    for t in 0..tokens {
        let row = t * k..(t + 1) * k;
        let mut pairs: Vec<(i32, f32)> = idx[row.clone()].iter().copied().zip(w[row.clone()].iter().copied()).collect();
        pairs.sort_by_key(|p| p.0);
        for (i, (e, v)) in pairs.into_iter().enumerate() {
            idx[t * k + i] = e;
            w[t * k + i] = v;
        }
    }
    case.write("indices", &idx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    // FlashInfer returns them in the logits' type: Float32 as is, BFloat16 rounded once.
    case.write_f32("weights", &w)
}

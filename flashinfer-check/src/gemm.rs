// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::gemm` on the inputs of `reference/cases_gemm.py`: the segment GEMM and `bmm_fp8`.

use cute::prelude::*;
use flashinfer::gemm::{BmmFp8Problem, Dtype, Fp8, SegmentProblem, segment_tile};

use crate::case::{Case, Gpu, Res, launch};

fn dtype(case: &Case) -> Res<Dtype> {
    match case.str("dtype") {
        "f16" => Ok(Dtype::F16),
        "bf16" => Ok(Dtype::BF16),
        other => Err(format!("dtype {other}").into()),
    }
}

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    match case.str("op") {
        "segment" => segment(gpu, case),
        "bmm_fp8" => bmm_fp8(gpu, case),
        other => Err(format!("op {other}").into()),
    }
}

fn segment(gpu: &Gpu, case: &Case) -> Res<()> {
    let dt = dtype(case)?;
    let lens = case.ints("seg_lens");
    let indices: Option<Vec<usize>> = case.flag("indexed").then(|| case.tensor("weight_indices").usizes());
    // The problem makes its own data; every vector is replaced by the file's.
    let mut p = SegmentProblem::new(
        dt,
        &lens,
        case.int("d_in"),
        case.int("d_out"),
        case.flag("col_major"),
        case.int("num_weights"),
        indices,
        false,
        0,
    );
    p.x = case.tensor("x").f32s();
    p.w = case.tensor("w").f32s();
    p.y0 = case.tensor("y0").f32s();
    p.tile = segment_tile(p.rows(), lens.len());
    let args = launch(gpu, &p.kernel(), p.launch(), p.args())?;
    case.write("y", &cute::host::arg_bytes(&args[2]))
}

fn bmm_fp8(gpu: &Gpu, case: &Case) -> Res<()> {
    let fp8 = match case.str("fp8") {
        "e4m3" => Fp8::E4M3,
        "e5m2" => Fp8::E5M2,
        other => return Err(format!("fp8 {other}").into()),
    };
    let mut p =
        BmmFp8Problem::new(fp8, dtype(case)?, case.int("batch"), case.int("m"), case.int("n"), case.int("k"), 0);
    p.a = case.tensor("a").f32s();
    p.b = case.tensor("b").f32s();
    p.a_scale = case.tensor("a_scale").f32s()[0];
    p.b_scale = case.tensor("b_scale").f32s()[0];
    // FP8 `mma.sync` lowers through NVVM only from LLVM 23; older toolchains need inline PTX.
    let arch = gpu.ctx.best_target(&gpu.tc).ok_or("no target for this GPU")?;
    let kernel = p.kernel();
    let old = kernel.min_llvm_version().is_some_and(|need| gpu.tc.llvm_version().is_some_and(|have| have < need));
    let f = gpu.ctx.load(&kernel.compile_with(&gpu.tc, &CompileOptions::new().gpu_arch(arch).mma_inline_ptx(old))?)?;
    let mut args = p.args();
    gpu.ctx.launch(&kernel, &f, p.launch(), &mut args)?;
    case.write("out", &cute::host::arg_bytes(&args[2]))
}

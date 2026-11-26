// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::quantization` on the inputs of `reference/cases_quantization.py`.

use cute::prelude::*;
use flashinfer::norm::Dtype;
use flashinfer::quantization::{GroupQuantProblem, Order, Out8, QuantMode, THREADS, pack_kernel, segment_kernel};

use crate::case::{Case, Gpu, Res, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    match case.str("op") {
        "packbits" | "segment_packbits" => run_pack(gpu, case),
        "group_quant" => run_group_quant(gpu, case),
        other => Err(format!("op {other}").into()),
    }
}

fn run_pack(gpu: &Gpu, case: &Case) -> Res<()> {
    let order = match case.str("order") {
        "big" => Order::Big,
        "little" => Order::Little,
        other => return Err(format!("order {other}").into()),
    };
    let bits: Vec<u8> = case.tensor("x").bytes;
    let input = KernelArg::Bytes { elem: DType::U8, data: bits.clone() };
    if case.str("op") == "packbits" {
        let n = bits.len();
        let args = vec![
            input,
            KernelArg::Bytes { elem: DType::U8, data: vec![0xAA; n.div_ceil(8)] },
            KernelArg::Scalar { ty: DType::I64, bits: n as u64 },
        ];
        let cfg = LaunchConfig::new([n.div_ceil(8 * THREADS).max(1) as u32, 1, 1], [THREADS as u32, 1, 1]);
        let args = launch(gpu, &pack_kernel(order), cfg, args)?;
        return case.write("out", &cute::host::arg_bytes(&args[1]));
    }
    // `output_indptr` is the prefix sum of `ceil(len / 8)`: the port takes it as an input,
    // FlashInfer returns it, so compute it from the same indptr.
    let indptr = case.tensor("indptr").i32s();
    let mut out_indptr = vec![0i32];
    for w in indptr.windows(2) {
        out_indptr.push(out_indptr.last().unwrap() + (w[1] - w[0] + 7) / 8);
    }
    let total = *out_indptr.last().unwrap() as usize;
    let args = vec![
        input,
        KernelArg::Bytes { elem: DType::U8, data: vec![0xAA; total] },
        KernelArg::I32(indptr.clone()),
        KernelArg::I32(out_indptr.clone()),
    ];
    let cfg = LaunchConfig::new([(indptr.len() - 1).max(1) as u32, 1, 1], [THREADS as u32, 1, 1]);
    let args = launch(gpu, &segment_kernel(order), cfg, args)?;
    case.write("out", &cute::host::arg_bytes(&args[1]))?;
    let bytes: Vec<u8> = match case.ref_dtype("new_indptr") {
        "int32" => out_indptr.iter().flat_map(|v| v.to_le_bytes()).collect(),
        "int64" => out_indptr.iter().flat_map(|&v| (v as i64).to_le_bytes()).collect(),
        other => return Err(format!("new_indptr dtype {other}").into()),
    };
    case.write("new_indptr", &bytes)
}

fn run_group_quant(gpu: &Gpu, case: &Case) -> Res<()> {
    let dtype = match case.str("dtype") {
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        other => return Err(format!("dtype {other}").into()),
    };
    let out = match case.str("out") {
        "e4m3" => Out8::E4M3,
        "e5m2" => Out8::E5M2,
        "int8" => Out8::Int8,
        other => return Err(format!("out {other}").into()),
    };
    let mode = QuantMode { column_major: case.flag("column_major"), ue8m0: case.flag("ue8m0") };
    let mut p =
        GroupQuantProblem::new(dtype, out, case.int("tokens"), case.int("hidden"), case.int("group_size"), mode, 0);
    p.x = case.tensor("x").f32s();
    p.eps = case.float("eps") as f32;
    let args = launch(gpu, &p.kernel(), p.launch(), p.args())?;
    case.write("q", &p.out_bytes(&args[1]))?;
    case.write("s", &cute::host::arg_bytes(&args[2]))
}

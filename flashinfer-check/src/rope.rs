// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::rope` on the inputs of `reference/cases_rope.py`: the in-place RoPE functions
//! and the paged KV append.

use cute::prelude::*;
use flashinfer::rope::{
    AppendProblem, CacheElem, FreqConfig, Half, Variant, append_kernel, bdx, bdy, freq_kernel, kernel,
    pos_ids_from_indptr,
};

use crate::case::{Case, Gpu, Res, Tensor, launch};

pub fn run(gpu: &Gpu, case: &Case) -> Res<()> {
    match (case.str("op"), case.str("dtype")) {
        ("append", "f16") => append::<F16>(gpu, case),
        ("append", "bf16") => append::<BF16>(gpu, case),
        ("append", "e4m3") => append::<F8E4M3FN>(gpu, case),
        ("append", "e5m2") => append::<F8E5M2>(gpu, case),
        (_, "f16") => rope::<F16>(gpu, case),
        (_, "bf16") => rope::<BF16>(gpu, case),
        (op, dtype) => Err(format!("op {op}, dtype {dtype}").into()),
    }
}

/// A tensor's elements as the type `T` of the same width.
fn elems<T: CacheElem>(t: &Tensor) -> Vec<T> {
    t.bytes.chunks(T::BYTES).map(T::from_le_bytes).collect()
}

/// A launch shape by token count, so that all three of the port's shapes get checked.
fn shape(nnz: usize, head_dim: usize, heads: usize) -> (Variant, [u32; 3]) {
    let nblks = nnz.div_ceil(bdy(head_dim)).max(1) as u32;
    if nnz <= 32 {
        (Variant::HeadParallel, [nblks, heads as u32, 1])
    } else if nnz <= 400 {
        (Variant::Split, [nblks, heads.min(3) as u32, 1])
    } else {
        (Variant::Unsplit, [nblks, 1, 1])
    }
}

fn rope<T: Half>(gpu: &Gpu, case: &Case) -> Result<(), Box<dyn std::error::Error>> {
    let (hd, nnz) = (case.int("head_dim"), case.int("nnz"));
    let (hq, hk) = (case.int("hq"), case.int("hk"));
    let interleave = case.flag("interleave");
    let (variant, grid) = shape(nnz, hd, hq + hk);
    let block = [bdx(hd) as u32, bdy(hd) as u32, 1];
    let cfg = LaunchConfig::new(grid, block);
    let q: Vec<T> = elems(&case.tensor("q"));
    let k: Vec<T> = elems(&case.tensor("k"));
    let (qa, ka) = (KernelArg::buffer(&q), KernelArg::buffer(&k));
    let counts = [KernelArg::ScalarI32(nnz as i32), KernelArg::ScalarI32(hq as i32), KernelArg::ScalarI32(hk as i32)];
    let op = case.str("op");
    let args = if op == "cache" {
        let pos = case.tensor("positions").i32s();
        let mut a = vec![qa, ka, KernelArg::f32(case.tensor("cos_sin_cache").f32s()), KernelArg::buffer(&pos)];
        a.extend(counts);
        launch(gpu, &kernel::<T, i32>(hd, interleave, variant), cfg, a)?
    } else {
        let pos = if op.starts_with("pos_ids") {
            case.tensor("pos_ids").i32s()
        } else {
            pos_ids_from_indptr(&case.tensor("indptr").i32s(), &case.tensor("offsets").i32s())
        };
        let rd = case.int("rotary_dim");
        let fc = if op.ends_with("llama31") { FreqConfig::llama31(rd) } else { FreqConfig::new(rd) };
        let mut a = vec![qa, ka, KernelArg::buffer(&pos)];
        a.extend(counts);
        a.extend(fc.args());
        launch(gpu, &freq_kernel::<T, i32>(hd, rd, interleave, variant), cfg, a)?
    };
    case.write("q", &cute::host::arg_bytes(&args[0]))?;
    case.write("k", &cute::host::arg_bytes(&args[1]))
}

fn append<T: CacheElem>(gpu: &Gpu, case: &Case) -> Res<()> {
    let p = AppendProblem::<T> {
        page_size: case.int("page_size"),
        num_kv_heads: case.int("kv_heads"),
        head_dim: case.int("head_dim"),
        num_pages: case.tensor("cache").shape[0],
        append_k: elems(&case.tensor("append_k")),
        append_v: elems(&case.tensor("append_v")),
        kv_cache: elems(&case.tensor("cache")),
        kv_indptr: case.tensor("kv_indptr").i32s(),
        kv_indices: case.tensor("kv_indices").i32s(),
        batch_indices: case.tensor("batch_indices").i32s(),
        positions: case.tensor("positions").i32s(),
    };
    let grid = p.nnz().clamp(1, 64);
    let args = launch(gpu, &append_kernel::<T>(p.num_kv_heads, p.head_dim), p.launch(grid), p.args())?;
    case.write("cache", &cute::host::arg_bytes(&args[2]))
}

// SPDX-License-Identifier: Apache-2.0

//! Runs cute-flashinfer's kernels on inputs saved from FlashInfer.
//!
//! `reference/gen.py` runs FlashInfer itself on seeded inputs and saves the inputs and its
//! outputs, one directory per case. This program gives each case's inputs to the cute-rs
//! port of the same function, on the GPU, and saves what it returns beside them;
//! `reference/compare.py` then compares the two. See `README.md`.
//!
//! `fi-check <cases-dir> [filter]` runs every case whose name contains `filter`.

mod case;

mod activation;
mod attention;
mod cascade;
mod decode;
mod gdn;
mod gemm;
mod kda;
mod mamba;
mod mla;
mod moe;
mod norm;
mod pod;
mod prefill;
mod quantization;
mod rope;
mod sampling;
mod sparse;
mod xqa;

use std::panic::{AssertUnwindSafe, catch_unwind};

use case::{Case, Gpu, Res};
use cute::cuda::CudaContext;
use cute::prelude::*;

/// The port of a case's module.
fn run(gpu: &'static Gpu, case: &Case) -> Res<()> {
    match case.str("module") {
        "activation" => activation::run(gpu, case),
        "attention" => attention::run(gpu, case),
        "cascade" => cascade::run(gpu, case),
        "decode" => decode::run(gpu, case),
        "gdn" => gdn::run(gpu, case),
        "gemm" => gemm::run(gpu, case),
        "kda" => kda::run(gpu, case),
        "mamba" => mamba::run(gpu, case),
        "mla" => mla::run(gpu, case),
        "moe" => moe::run(gpu, case),
        "norm" => norm::run(gpu, case),
        "pod" => pod::run(gpu, case),
        "prefill" => prefill::run(gpu, case),
        "quantization" => quantization::run(gpu, case),
        "rope" => rope::run(gpu, case),
        "sampling" => sampling::run(gpu, case),
        "sparse" => sparse::run(gpu, case),
        "xqa" => xqa::run(gpu, case),
        other => Err(format!("no runner for module `{other}`").into()),
    }
}

fn main() -> Res<()> {
    let mut args = std::env::args().skip(1);
    let root = args.next().ok_or("usage: fi-check <cases-dir> [filter]")?;
    let filter = args.next().unwrap_or_default();
    // One device and compiler for the whole run, so a runner can keep compiled kernels between cases.
    let gpu: &'static Gpu = Box::leak(Box::new(Gpu {
        ctx: CudaContext::new(0)?,
        tc: Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?,
    }));
    println!("device: {}", gpu.ctx.name());
    let mut dirs: Vec<_> = std::fs::read_dir(&root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.txt").is_file())
        .filter(|p| p.file_name().unwrap().to_string_lossy().contains(&filter))
        .collect();
    dirs.sort();
    let (mut ran, mut failed) = (0, 0);
    for dir in dirs {
        let case = Case::load(&dir)?;
        // A rerun must not leave an earlier run's outputs or error behind.
        for entry in std::fs::read_dir(&dir)?.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("out.") || name == "error.txt" {
                std::fs::remove_file(entry.path())?;
            }
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| run(gpu, &case))).unwrap_or_else(|panic| {
            let text =
                panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()));
            Err(format!("panic: {}", text.unwrap_or_default()).into())
        });
        ran += 1;
        match outcome {
            Ok(()) => println!("ran    {}", case.name),
            Err(e) => {
                failed += 1;
                println!("ERROR  {}: {e}", case.name);
                std::fs::write(dir.join("error.txt"), e.to_string())?;
            }
        }
    }
    println!("{ran} cases, {failed} could not run");
    Ok(())
}

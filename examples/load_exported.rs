//! Load kernels exported by the `export_to_c` example and launch them, without their Rust code
//! or an `mlir-opt`: `CudaContext::load_exported` reads each manifest, checks its format
//! version, target and PTX version against this machine, and the launch checks the arguments
//! against the exported signature.
//!
//! A port of CUTLASS 4.8.1's `examples/python/CuTeDSL/dsl_tutorials/export/load_in_python.py`,
//! Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! BSD-3-Clause; see THIRD-PARTY-NOTICES.md. CuTe DSL loads the object files or the shared
//! library; cute-rs loads the manifest (`<name>.sig`) and the PTX and cubin beside it, which the
//! object files embed (a C program links those instead). The tensors are 10 x 16 here, row-major.
//!
//! ```text
//! cargo run --release --example export_to_c -- build
//! cargo run --release --features cuda --example load_exported -- build
//! ```

use std::error::Error;
use std::path::PathBuf;

use cute::cuda::{CudaContext, DeviceArg};
use cute::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "build".to_string()));
    let ctx = CudaContext::new(0)?;
    let (m, n) = (16usize, 10usize);
    let a = KernelArg::f32((0..m * n).map(|i| i as f32).collect());
    let mut b = KernelArg::f32(vec![1.0; m * n]);
    let (da, db) = (ctx.upload(&a)?, ctx.upload(&b)?);
    let shape =
        || [DeviceArg::Scalar(KernelArg::ScalarI32(m as i32)), DeviceArg::Scalar(KernelArg::ScalarI32(n as i32))];
    let cfg = LaunchConfig::new([1, 1, 1], [1, 1, 1]);

    let print_tensor = ctx.load_exported(dir.join("print_tensor_example.sig"))?;
    let [am, an] = shape();
    print_tensor.launch_builder(cfg)?.args(&[DeviceArg::Buffer(&da), am, an]).submit()?;
    ctx.synchronize()?;

    let add_one = ctx.load_exported(dir.join("add_one_example.sig"))?;
    let ([am, an], [bm, bn]) = (shape(), shape());
    add_one.launch_builder(cfg)?.args(&[DeviceArg::Buffer(&da), am, an, DeviceArg::Buffer(&db), bm, bn]).submit()?;
    ctx.synchronize()?;
    let mut out = a.clone();
    ctx.download(&da, &mut out)?;
    ctx.download(&db, &mut b)?;
    assert_eq!(out.as_f32()[0], b.as_f32()[0] + 1.0);
    println!("a[0] = {} = b[0] + 1", out.as_f32()[0]);
    Ok(())
}

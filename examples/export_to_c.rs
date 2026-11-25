//! Export compiled kernels to C: a header, a C source with the kernel embedded, an object file
//! when a C compiler and `cuda.h` are found, and a manifest that `CudaContext::load_exported`
//! reads back (see the `load_exported` example).
//!
//! A port of CUTLASS 4.8.1's `examples/python/CuTeDSL/dsl_tutorials/export/export_to_c.py`,
//! Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! BSD-3-Clause; see THIRD-PARTY-NOTICES.md. As there, two kernels on tensors of launch-time
//! shape (the first extent a multiple of 16) are exported, and their object files are linked into
//! one shared library. The exports use the CUDA driver API (link with `-lcuda`) where CuTe DSL's
//! use the runtime API, and a launch takes a configuration struct where CuTe DSL's wrapper takes
//! the stream.
//!
//! ```text
//! cargo run --release --example export_to_c [output directory, default ./build]
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use cute::prelude::*;

/// A tensor of `m x n` f32 elements, row-major, with `m` a multiple of 16 and `n` any size,
/// both known at launch.
fn dynamic_tensor(name: &str) -> Tensor<f32> {
    let ptr = param_ptr::<f32>(name);
    let m = param_extent(&format!("{name}_m"), 16);
    let n = param_extent(&format!("{name}_n"), 1);
    make_tensor(&ptr, Layout::row_major((m, n)))
}

/// One thread prints the tensor's layout and first element (CuTe DSL's kernel prints the whole
/// tensor, which cute-rs's `printf` does only for static shapes).
pub fn print_tensor_kernel() -> Kernel {
    Kernel::builder("print_tensor_kernel").block_dim([1, 1, 1]).build(|| {
        let a = dynamic_tensor("a");
        printf("a: {} a[0] = {}\n", &[a.layout().into(), a.load((0, 0)).into()]);
    })
}

/// `a[0] = b[0] + 1`.
pub fn add_one_kernel() -> Kernel {
    Kernel::builder("add_one_kernel").block_dim([1, 1, 1]).build(|| {
        let a = dynamic_tensor("a");
        let b = dynamic_tensor("b");
        a.store((0, 0), b.load((0, 0)) + 1.0f32);
    })
}

/// Compile both kernels for `arch` and export them to `dir`; returns the object files built.
pub fn export(toolchain: &Toolchain, arch: Arch, dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let options = CompileOptions::new().gpu_arch(arch).with_env()?;
    let mut objects = vec![];
    for (kernel, name, prefix) in [
        (print_tensor_kernel(), "print_tensor_example", "print_tensor"),
        (add_one_kernel(), "add_one_example", "add_one"),
    ] {
        let compiled = kernel.compile_with(toolchain, &options)?;
        let report = compiled.export_c(dir, name, prefix)?;
        println!("{}: {} and {}", kernel.ir().name, report.header.display(), report.source.display());
        for note in &report.notes {
            println!("  note: {note}");
        }
        objects.extend(report.object);
    }
    Ok(objects)
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "build".to_string()));
    let toolchain = Toolchain::discover().ok_or("no mlir-opt found (set CUTE_MLIR_OPT)")?;
    let arch = match std::env::var("CUTE_RS_ARCH") {
        Ok(a) => a.parse()?,
        Err(_) => Arch::Sm80,
    };
    let objects = export(&toolchain, arch, &dir)?;
    if objects.len() < 2 {
        println!("no object files (no C compiler or cuda.h): skipping the shared library");
        return Ok(());
    }
    // Link the object files into one shared library, as the tutorial does.
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let lib = dir.join("libexport_example.so");
    let status = Command::new(&cc).arg("-shared").arg("-o").arg(&lib).args(&objects).arg("-lcuda").status()?;
    if !status.success() {
        return Err(format!("`{cc} -shared` failed").into());
    }
    println!("{}", lib.display());
    Ok(())
}

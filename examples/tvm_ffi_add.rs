// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// This example follows two CuTe DSL examples of NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/jit_and_use_in_torch.py and
//   examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/error_reporting.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and takes their kernel (`b[i] = a[i] + 1` on one thread over `a`'s extent) and the
// layouts it is compiled for (dynamic, and the static `(10,)` of error_reporting.py) from
// them. Those parts are under the BSD-3-Clause licence, and the rest of the file is
// cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text and
// describes what is taken.

//! TVM-FFI functions in a shared library: CuTe DSL's `jit_and_use_in_torch.py` and
//! `error_reporting.py` as a Rust `cdylib` (`crate-type = ["cdylib"]`) that exports
//!
//! * `__tvm_ffi_add_one`: `b = a + 1` on two 1-D Float32 CUDA tensors of any (equal) length,
//!   the layout `mark_layout_dynamic()` gives them (CuTe DSL gives `a` and `b` separate
//!   extents; one shared extent here makes a shorter `b` an error instead of a write past
//!   it);
//! * `__tvm_ffi_add_one_static`: the same kernel for the static layout `(10,):(1,)` of
//!   `error_reporting.py`, which refuses a tensor of 5 elements with a `ValueError`.
//!
//! The kernels are compiled with `mlir-opt` on each thread's first call (the target is the
//! one for CUDA device 0), and launched on the stream TVM-FFI reports for the tensors'
//! device. From Python (`examples/tvm_ffi/*.py` are the two tutorials' scripts):
//!
//! ```text
//! cargo build --release --features cuda --example tvm_ffi_add
//! python examples/tvm_ffi/jit_and_use_in_torch.py target/release/examples/libtvm_ffi_add.so
//! python examples/tvm_ffi/error_reporting.py target/release/examples/libtvm_ffi_add.so
//! ```

use cute::prelude::*;

/// The kernel's parameters: `a` and `b`, each a DLPack tensor whose host value is `H`
/// (a [`DlpackTensor`] for the TVM-FFI call; any [`DlpackSource`] for the simulator).
pub type AddOneParams<H> = (DlTensor<f32, H>, DlTensor<f32, H>);

/// `device_add_one` and `add_one` of the tutorials: one thread adds 1 to each element of `a`
/// into `b`. `len` is the static extent (`error_reporting.py`) or `None` for a dynamic one.
pub fn add_one_kernel<H: DlpackSource>(len: Option<usize>) -> TypedKernel<AddOneParams<H>> {
    let spec = match len {
        Some(n) => make_fake_tensor::<f32>((n,), (1,)),
        None => make_fake_tensor::<f32>((sym_int32(1, Some("n")),), (1,)),
    };
    let name = if len.is_some() { "add_one_static" } else { "add_one" };
    Kernel::builder(name).block_dim([1, 1, 1]).jit((spec.clone(), spec), |a: DlTensor<f32, H>, b: DlTensor<f32, H>| {
        let n: Int32 = size(a.tensor(), &[0]).into_value();
        for_range(0, n, 1, |i| b.store(i, a.load(i) + 1.0f32));
    })
}

/// The launch of both: one block of one thread (`launch(grid=(1, 1, 1), block=(1, 1, 1))`).
pub fn one_thread<H>(_: &H) -> Result<LaunchConfig, String> {
    Ok(LaunchConfig::new([1, 1, 1], [1, 1, 1]))
}

/// The callable: the kernel compiled with `--enable-tvm-ffi` (CuTe DSL's
/// `cute.compile(add_one, a, b, options="--enable-tvm-ffi")`).
fn callable(len: Option<usize>) -> Result<TvmFfiFunction<AddOneParams<DlpackTensor>>, Box<dyn std::error::Error>> {
    let toolchain = Toolchain::discover().ok_or("no mlir-opt found (set CUTE_MLIR_OPT)")?;
    let options = CompileOptions::new().with_env()?;
    let f = TvmFfiFunction::compile(add_one_kernel(len), &toolchain, &options, one_thread)?;
    Ok(f.with_arg_names(&["a", "b"]))
}

cute::tvm_ffi_export!(add_one, callable(None));
cute::tvm_ffi_export!(add_one_static, callable(Some(10)));

// `ctypes.CDLL(path).cute_tvm_ffi_register_all()` after `import tvm_ffi` makes the two functions
// global (`tvm_ffi.get_global_func("cute_rs.add_one")`), the counterpart of
// `tvm_ffi.register_global_func(name, compiled)`.
cute::tvm_ffi_register_all!("cute_rs.add_one" => add_one, "cute_rs.add_one_static" => add_one_static);

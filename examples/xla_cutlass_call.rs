// SPDX-License-Identifier: MIT OR Apache-2.0

//! A shared library with the XLA FFI handler for JAX (`crate-type = ["cdylib"]`): the Rust
//! counterpart of `cutlass.jax.cutlass_call`. It exports the handler symbols
//! (`cute_xla_cutlass_call`, `cute_xla_cutlass_call_nograph`, `cute_xla_cutlass_call_bundle`)
//! and `cute_rs_register_targets`, which registers the target `add_one` (`b = a + 1` over
//! 1-D float32 arrays; the output may alias the input when the call has the attribute `alias`). From Python, `python/cute_rs_jax`
//! loads it:
//!
//! ```text
//! cargo build --release --features cuda --example xla_cutlass_call
//! python examples/xla/cutlass_call_basic.py target/release/examples/libxla_cutlass_call.so
//! ```

use cute::prelude::*;
use cute::tvm_ffi::TvmFfiFunction;
use cute::xla_ffi::{XlaCallSpec, XlaTarget, register_cutlass_call};

cute::xla_ffi_export!();

type Params = (DlTensor<f32>, DlTensor<f32>);

/// One thread adds 1 to each of `a`'s elements into `b`, over a flat compact 1-D view.
fn add_one_kernel() -> TypedKernel<Params> {
    let spec = make_fake_tensor::<f32>((sym_int32(1, Some("n")),), (1,));
    Kernel::builder("xla_add_one").block_dim([128, 1, 1]).jit(
        (spec.clone(), spec),
        |a: DlTensor<f32>, b: DlTensor<f32>| {
            let n: Int32 = size(a.tensor(), &[0]).into_value();
            for_range(thread_idx().x + block_idx().x * 128, n, grid_dim().x * 128, |i| b.store(i, a.load(i) + 1.0f32));
        },
    )
}

/// The builder: one compiled kernel per call spec (the handler caches it).
fn build(spec: &XlaCallSpec) -> Result<XlaTarget, String> {
    if spec.args.len() != 1 || spec.rets.len() != 1 {
        return Err(format!(
            "add_one takes one array and returns one, got {} and {}",
            spec.args.len(),
            spec.rets.len()
        ));
    }
    let toolchain = Toolchain::discover().ok_or("no mlir-opt found (set CUTE_MLIR_OPT)")?;
    let f = TvmFfiFunction::compile(
        add_one_kernel(),
        &toolchain,
        &CompileOptions::new().with_env().map_err(|e| e.to_string())?,
        |h| {
            let n: i64 = h.0.shape().iter().product();
            Ok(LaunchConfig::new([(n as u32).div_ceil(128).clamp(1, 1024), 1, 1], [128, 1, 1]))
        },
    )
    .map_err(|e| e.to_string())?;
    let target = XlaTarget::new(Box::new(f.with_arg_names(&["a", "b"])));
    Ok(if spec.attr("alias").is_some() { target.alias(0, 0) } else { target })
}

/// Register the library's targets (Python calls this once after loading the library).
#[unsafe(no_mangle)]
pub extern "C" fn cute_rs_register_targets() {
    register_cutlass_call("add_one", build);
}

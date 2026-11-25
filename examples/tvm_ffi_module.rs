//! Two TVM-FFI functions from one module of two kernels, in one shared library
//! (`crate-type = ["cdylib"]`):
//!
//! * `__tvm_ffi_mk_stash`: `mk_stash[i] = 2 * a[i]` for the elements of a 1-D Float32 CUDA
//!   tensor `a` (at most 1024), and `mk_count = len(a)`, in two `__device__` globals;
//! * `__tvm_ffi_mk_unstash`: `b[i] = mk_stash[i] + 1` for `i < mk_count`.
//!
//! Both kernels are compiled into one module (`KernelModule`), and each thread loads it once
//! for both functions, so the second sees what the first left in the module's globals. The
//! plain C function `cute_rs_tvm_ffi_module_loads` says how many loads there have been
//! (`cute::tvm_ffi::shared_module_loads`). tests/gpu_multi_kernel.rs calls both from C:
//!
//! ```text
//! cargo build --release --features cuda --example tvm_ffi_module
//! CUTE_RS_TVM_FFI_MODULE_LIB=target/release/examples/libtvm_ffi_module.so \
//!     cargo test --release --features cuda --test gpu_multi_kernel
//! ```

use cute::prelude::*;

/// The most elements the module's globals hold.
pub const CAPACITY: usize = 1024;

/// A function's one parameter: a 1-D Float32 DLPack tensor whose host value is `H`.
pub type OneTensor<H> = DlTensor<f32, H>;

fn spec() -> TensorSpec {
    make_fake_tensor::<f32>((sym_int32(1, Some("n")),), (1,))
}

fn globals() -> (Tensor<f32>, Tensor<i32>) {
    (
        module_global::<f32>(Some("mk_stash"), CAPACITY, AddressSpace::Gmem, None, false),
        module_global::<i32>(Some("mk_count"), 1, AddressSpace::Gmem, None, false),
    )
}

/// One thread stashes `2 * a` and its length in the module's globals.
pub fn stash_kernel<H: DlpackSource>() -> TypedKernel<OneTensor<H>> {
    Kernel::builder("mk_stash_kernel").block_dim([1, 1, 1]).jit(spec(), |a: DlTensor<f32, H>| {
        let (stash, count) = globals();
        let n: Int32 = size(a.tensor(), &[0]).into_value();
        let n = n.min(CAPACITY as i32);
        for_range(0, n, 1, |i| stash.store(i, a.load(i) * 2.0f32));
        count.store(0, n);
    })
}

/// One thread writes `stash + 1` into `b`, for as many elements as were stashed.
pub fn unstash_kernel<H: DlpackSource>() -> TypedKernel<OneTensor<H>> {
    Kernel::builder("mk_unstash_kernel").block_dim([1, 1, 1]).jit(spec(), |b: DlTensor<f32, H>| {
        let (stash, count) = globals();
        let n: Int32 = size(b.tensor(), &[0]).into_value();
        for_range(0, count.load(0).min(n), 1, |i| b.store(i, stash.load(i) + 1.0f32));
    })
}

/// One block of one thread.
pub fn one_thread<H>(_: &H) -> Result<LaunchConfig, String> {
    Ok(LaunchConfig::new([1, 1, 1], [1, 1, 1]))
}

/// The module of both kernels, compiled for CUTE_RS_ARCH or (feature `cuda`) CUDA device 0.
fn compiled() -> Result<CompiledModule, Box<dyn std::error::Error>> {
    let toolchain = Toolchain::discover().ok_or("no mlir-opt found (set CUTE_MLIR_OPT)")?;
    let options = CompileOptions::new().with_env()?.enable_tvm_ffi(true);
    #[cfg(feature = "cuda")]
    let options = match options.get_gpu_arch() {
        Some(_) => options,
        None => {
            let ctx = cute::cuda::CudaContext::new(0)?;
            options.gpu_arch(ctx.best_target(&toolchain).ok_or("no target runs on CUDA device 0")?)
        }
    };
    let module = KernelModule::new(
        "mk_tvm",
        [stash_kernel::<DlpackTensor>().kernel().clone(), unstash_kernel::<DlpackTensor>().kernel().clone()],
    )?;
    Ok(module.compile_with(&toolchain, &options)?)
}

fn stash() -> Result<TvmFfiFunction<OneTensor<DlpackTensor>>, Box<dyn std::error::Error>> {
    Ok(TvmFfiFunction::from_module(stash_kernel(), &compiled()?, one_thread)?.with_arg_names(&["a"]))
}

fn unstash() -> Result<TvmFfiFunction<OneTensor<DlpackTensor>>, Box<dyn std::error::Error>> {
    Ok(TvmFfiFunction::from_module(unstash_kernel(), &compiled()?, one_thread)?.with_arg_names(&["b"]))
}

cute::tvm_ffi_export!(mk_stash, stash());
cute::tvm_ffi_export!(mk_unstash, unstash());

/// How many times this library has loaded a module for its TVM-FFI functions.
#[unsafe(no_mangle)]
pub extern "C" fn cute_rs_tvm_ffi_module_loads() -> u64 {
    cute::tvm_ffi::shared_module_loads()
}

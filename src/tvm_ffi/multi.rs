//! TVM-FFI functions of the kernels of one module ([`TvmFfiFunction::from_module`]): each is
//! exported as its own `__tvm_ffi_<name>` symbol, and on each thread and device the module is
//! loaded once for all of them, so they share its globals as the kernels of one CuTe DSL
//! compiled function do.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{FromTvmFfi, Image, TvmFfiFunction};
use crate::aot::ExportedModule;
use crate::backend::{CompiledKernel, CompiledModule};
use crate::params::{KernelParam, TypedKernel};
use crate::sim::LaunchConfig;

/// Module loads made for TVM-FFI functions of modules in this process (one per thread, device
/// and module, however many of its kernels are called).
static LOADS: AtomicU64 = AtomicU64::new(0);

/// How many times a module of TVM-FFI functions ([`TvmFfiFunction::from_module`]) has been
/// loaded in this process: once per thread, device and module.
pub fn shared_module_loads() -> u64 {
    LOADS.load(Ordering::Relaxed)
}

enum Source {
    Compiled(CompiledModule),
    Exported(ExportedModule),
}

/// A module whose kernels' TVM-FFI functions share one load per thread and device.
pub(super) struct SharedModule {
    /// The module's identity: a hash of its kernels' names and its images.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    key: [u8; 32],
    source: Source,
}

impl SharedModule {
    fn new(source: Source) -> SharedModule {
        let (names, ptx, cubin): (Vec<&str>, Option<&str>, Option<&[u8]>) = match &source {
            Source::Compiled(m) => (m.kernel_names(), Some(m.ptx()), m.cubin()),
            Source::Exported(m) => (m.kernel_names(), m.ptx.as_deref(), m.cubin.as_deref()),
        };
        let mut bytes = names.join("\0").into_bytes();
        bytes.extend(b"\0ptx\0");
        bytes.extend(ptx.unwrap_or("").as_bytes());
        bytes.extend(b"\0cubin\0");
        bytes.extend(cubin.unwrap_or(&[]));
        SharedModule { key: crate::backend::cache::sha256(&bytes), source }
    }

    /// The compiled kernel `name`, when the module was compiled here.
    pub(super) fn compiled(&self, name: &str) -> Option<&CompiledKernel> {
        match &self.source {
            Source::Compiled(m) => m.kernel(name),
            Source::Exported(_) => None,
        }
    }

    /// Kernel `name`'s function in the module loaded on `device` (loaded on this thread's first
    /// call of any of the module's functions).
    #[cfg(feature = "cuda")]
    pub(super) fn function(
        &self,
        ctx: &'static crate::cuda::CudaContext,
        device: i32,
        name: &str,
    ) -> Result<&'static crate::cuda::CudaFunction<'static>, crate::cuda::CudaError> {
        type Loaded = (i32, [u8; 32], &'static crate::cuda::CudaModule<'static>);
        thread_local! {
            static MODULES: std::cell::RefCell<Vec<Loaded>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let module = MODULES.with(|ms| {
            if let Some((_, _, m)) = ms.borrow().iter().find(|(d, k, _)| *d == device && *k == self.key) {
                return Ok(*m);
            }
            let m = match &self.source {
                Source::Compiled(m) => ctx.load_kernel_module(m)?,
                Source::Exported(m) => ctx.load_exported_module_from(m)?,
            };
            LOADS.fetch_add(1, Ordering::Relaxed);
            // Leaked with its context: TLS destructors at exit must not unload modules.
            let m: &'static crate::cuda::CudaModule<'static> = Box::leak(Box::new(m));
            ms.borrow_mut().push((device, self.key, m));
            Ok::<_, crate::cuda::CudaError>(m)
        })?;
        module.function(name)
    }
}

impl<P: KernelParam> TvmFfiFunction<P>
where
    P::Host: FromTvmFfi,
{
    /// The callable of `kernel`, compiled as kernel `kernel.name()` of `module`
    /// ([`KernelModule::compile_with`](crate::KernelModule::compile_with)). Callables of one
    /// module's kernels load it once per thread and device, and share its globals.
    pub fn from_module(
        kernel: TypedKernel<P>,
        module: &CompiledModule,
        config: impl Fn(&P::Host) -> Result<LaunchConfig, String> + 'static,
    ) -> Result<TvmFfiFunction<P>, String> {
        let compiled = module.kernel(kernel.name()).ok_or_else(|| {
            format!(
                "module `{}` has no kernel `{}` (it has {})",
                module.name(),
                kernel.name(),
                module.kernel_names().join(", ")
            )
        })?;
        let signature = compiled.signature().clone();
        let image = Image::Module(Box::new(SharedModule::new(Source::Compiled(module.clone()))));
        TvmFfiFunction::with_image(kernel, signature, image, config)
    }

    /// [`from_module`](Self::from_module) from an exported module ([`ExportedModule`]): no
    /// `mlir-opt` is needed when it is called.
    pub fn from_exported_module(
        kernel: TypedKernel<P>,
        module: &ExportedModule,
        config: impl Fn(&P::Host) -> Result<LaunchConfig, String> + 'static,
    ) -> Result<TvmFfiFunction<P>, String> {
        let signature = module.signatures.iter().find(|s| s.name == kernel.name()).cloned().ok_or_else(|| {
            format!(
                "exported module `{}` has no kernel `{}` (it has {})",
                module.name,
                kernel.name(),
                module.kernel_names().join(", ")
            )
        })?;
        let image = Image::Module(Box::new(SharedModule::new(Source::Exported(module.clone()))));
        TvmFfiFunction::with_image(kernel, signature, image, config)
    }
}

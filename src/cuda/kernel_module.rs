//! A module of several kernels loaded once ([`CudaContext::load_kernel_module`]), its kernels
//! looked up by name, as CuTe DSL's compiled function loads its module and finds each kernel it
//! launches.

use super::{CudaContext, CudaError, CudaFunction, DeviceGlobal, ModuleImage, err};
use crate::aot::{ExportedModule, ImageChoice};
use crate::backend::CompiledModule;

/// A loaded module: one `CUmodule`, and a [`CudaFunction`] per kernel, each carrying its
/// kernel's signature (so its launches are checked and packed as any compiled kernel's are).
/// The functions share the module's globals ([`global`](Self::global)).
pub struct CudaModule<'a> {
    name: String,
    functions: Vec<CudaFunction<'a>>,
}

impl CudaContext {
    /// Load a compiled module once ([`load`](Self::load)'s choice of the cubin or the PTX, and
    /// its per-context module cache) and look up every kernel by name.
    pub fn load_kernel_module(&self, module: &CompiledModule) -> Result<CudaModule<'_>, CudaError> {
        let (loaded, info) = self.load_compiled_image(&module.kernels()[0])?;
        let mut functions = Vec::with_capacity(module.kernels().len());
        for k in module.kernels() {
            let mut f = self.function(loaded.clone(), k.name(), info.clone())?;
            f.signature = Some(k.signature().clone());
            functions.push(f);
        }
        Ok(CudaModule { name: module.name().to_string(), functions })
    }

    /// Load an exported module ([`CompiledModule::export_c`]: format 2, or a single kernel's
    /// format 1) from its manifest and the images beside it, and look up every kernel by name.
    pub fn load_exported_module(&self, path: impl AsRef<std::path::Path>) -> Result<CudaModule<'_>, CudaError> {
        let module = ExportedModule::read(path).map_err(|e| err(e.to_string()))?;
        self.load_exported_module_from(&module)
    }

    /// [`load_exported_module`](Self::load_exported_module) from an [`ExportedModule`] in memory.
    pub fn load_exported_module_from(&self, module: &ExportedModule) -> Result<CudaModule<'_>, CudaError> {
        let (major, minor) = self.compute_capability();
        let cc = (major.max(0) as u32, minor.max(0) as u32);
        let choice = module.choose_image(cc, self.driver_ptx().ok()).map_err(|e| err(e.to_string()))?;
        let use_cache = !crate::backend::cache::no_cache_env();
        let first = &module.signatures[0].name;
        let (loaded, info) = match (choice, &module.cubin, &module.ptx) {
            (ImageChoice::Cubin, Some(cubin), _) => {
                self.load_image(ModuleImage::Cubin(cubin), &[], first, use_cache)?
            }
            (_, _, Some(ptx)) => self.load_image(ModuleImage::Ptx(ptx), &[], first, use_cache)?,
            _ => return Err(err(format!("exported module `{}` has no image this device loads", module.name))),
        };
        let mut functions = Vec::with_capacity(module.signatures.len());
        for sig in &module.signatures {
            let mut f = self.function(loaded.clone(), &sig.name, info.clone())?;
            f.signature = Some(sig.clone());
            functions.push(f);
        }
        Ok(CudaModule { name: module.name.clone(), functions })
    }
}

impl<'a> CudaModule<'a> {
    /// The module's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The kernels' functions, in the module's order.
    pub fn functions(&self) -> &[CudaFunction<'a>] {
        &self.functions
    }

    /// The kernels' names, in order.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.functions.iter().filter_map(|f| f.signature().map(|s| s.name.as_str())).collect()
    }

    /// The function of kernel `name`.
    pub fn function(&self, name: &str) -> Result<&CudaFunction<'a>, CudaError> {
        self.functions.iter().find(|f| f.signature().is_some_and(|s| s.name == name)).ok_or_else(|| {
            err(format!("module `{}` has no kernel `{name}` (it has {})", self.name, self.kernel_names().join(", ")))
        })
    }

    /// Module global `name`, shared by every kernel of the module, to read and write from the
    /// host between launches ([`CudaFunction::global`]).
    pub fn global(&self, name: &str) -> Result<DeviceGlobal<'_>, CudaError> {
        let f = self
            .functions
            .iter()
            .find(|f| f.signature().is_some_and(|s| s.global(name).is_some()))
            .unwrap_or(&self.functions[0]);
        f.global(name)
    }

    /// Whether every function was looked up in one `CUmodule` (always, for a module this made).
    pub fn is_one_module(&self) -> bool {
        self.functions.windows(2).all(|w| std::rc::Rc::ptr_eq(&w[0].module, &w[1].module))
    }
}

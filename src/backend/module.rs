//! Several kernels in one module: what CuTe DSL's `cute.compile` of a `@cute.jit` host function
//! that launches several `@cute.kernel`s produces (one GPU module, one PTX and cubin, whose
//! kernels the compiled function looks up by name when it launches them).
//!
//! A [`KernelModule`] is built from [`Kernel`]s with distinct names. [`KernelModule::compile_with`]
//! emits every kernel into one `gpu.module`, lowers it once and assembles it once, giving a
//! [`CompiledModule`]: one [`CompiledKernel`] per kernel, all carrying the module's PTX and
//! cubin, each with its own name (its PTX entry point), signature, IR and diagnostics.
//! Loading any of them loads the module; [`CudaContext::load_kernel_module`] loads it once and
//! looks every kernel up by name, [`KernelModule::sim_module`] simulates them over shared module
//! globals, [`CompiledModule::export_c`] exports them together and
//! [`TvmFfiFunction::from_module`] exports each as a TVM-FFI function sharing one module load.
//!
//! What a module shares and what stays per kernel:
//!
//! * the target, the PTX ISA version, the optimization level, line information and every other
//!   [`CompileOptions`] setting are the module's. Kernels built for different targets
//!   ([`KernelOptions::target`]) cannot share one; the PTX ISA version is the newest any kernel
//!   needs;
//! * the block size (`.reqntid`, `.maxntid`), `.minnctapersm`, parameters and shared memory
//!   (including the size of the dynamic window) are each kernel's own. Two kernels' static
//!   shared buffers of the same name become two symbols (the later kernel's is renamed in the
//!   PTX);
//! * module globals ([`module_global`](crate::module_global)) are the module's: kernels that
//!   declare a global of the same name share it (one `__device__` or `__constant__` symbol), and
//!   must declare it alike (element type, length, memory space, initial value, read-only and
//!   alignment), or the module is refused. An unnamed global's automatic name is the same in
//!   every kernel, so two kernels with unnamed globals of the same automatic name are refused
//!   rather than silently sharing one;
//! * printf formats and assertion messages are module-level strings the lowering names apart.
//!
//! A module of one kernel compiles exactly as [`Kernel::compile_with`] does: the same MLIR,
//! PTX, cubin and cache key.
//!
//! [`CudaContext::load_kernel_module`]: crate::cuda::CudaContext::load_kernel_module
//! [`TvmFfiFunction::from_module`]: crate::tvm_ffi::TvmFfiFunction::from_module
//! [`KernelOptions::target`]: crate::KernelOptions::target

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::Instant;

use super::mlir::emit_mlir_in_module;
use super::{
    Arch, Artifact, ArtifactNote, CompileError, CompileOptions, CompiledKernel, MlirOptions, OptionError, Toolchain,
    VERSION, byval_hook, cache, cudatools, dump, jit_options_for_ptxas_flags, nvvm_needs, options, ptx_reason_text,
    ptxas_max_ptx,
};
use crate::builder::Kernel;
use crate::ir::{DYNAMIC_SMEM_SYMBOL, KernelIr, MemSpace};
use crate::log::Level;

/// Why kernels cannot share a module ([`KernelModule::new`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleError(String);

impl ModuleError {
    fn new(message: impl Into<String>) -> ModuleError {
        ModuleError(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModuleError {}

impl From<ModuleError> for CompileError {
    fn from(e: ModuleError) -> CompileError {
        CompileError::Options(OptionError::new(e.0))
    }
}

/// Several kernels compiled into one module (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct KernelModule {
    name: String,
    kernels: Vec<Kernel>,
}

impl From<Kernel> for KernelModule {
    /// The module of one kernel, named after it.
    fn from(kernel: Kernel) -> KernelModule {
        KernelModule { name: kernel.name().to_string(), kernels: vec![kernel] }
    }
}

/// The automatic name of an unnamed module global (`__cute_global_<n>`, see `abi::module_global`).
const UNNAMED_GLOBAL_PREFIX: &str = "__cute_global_";

/// The parts of a module global's declaration that kernels sharing it must agree on.
fn global_decl(ir: &KernelIr, g: &crate::ir::ModuleGlobal) -> String {
    let d = ir.buffer(g.buffer);
    let space = if d.space == MemSpace::Constant { "__constant__" } else { "__device__" };
    format!(
        "{space} {} {}[{}] align {}{}{}",
        d.elem.cute_name(),
        d.name,
        d.len.unwrap_or(0),
        d.align,
        if g.readonly { " read-only" } else { "" },
        match &g.init {
            Some(b) => format!(" = 0x{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>()),
            None => String::new(),
        }
    )
}

impl KernelModule {
    /// The module `name` (an identifier, which names its artifacts and export) of `kernels`,
    /// in order. Fails when there is no kernel, two kernels share a name, a kernel is named
    /// like a module global, kernels were built for different targets, or kernels declare a
    /// module global of the same name differently.
    pub fn new(
        name: impl Into<String>,
        kernels: impl IntoIterator<Item = Kernel>,
    ) -> Result<KernelModule, ModuleError> {
        let name = name.into();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(ModuleError::new(format!("module name {name:?} must be a non-empty identifier")));
        }
        let kernels: Vec<Kernel> = kernels.into_iter().collect();
        if kernels.is_empty() {
            return Err(ModuleError::new(format!("module `{name}` has no kernels")));
        }
        let mut names = BTreeSet::new();
        for k in &kernels {
            if !names.insert(k.name()) {
                return Err(ModuleError::new(format!(
                    "module `{name}` has two kernels named `{}`: kernels in one module are looked up by name, so \
                     their names must differ",
                    k.name()
                )));
            }
        }
        let mut target: Option<(Arch, &str)> = None;
        for k in &kernels {
            match (target, k.target()) {
                (Some((a, first)), Some(t)) if a != t => {
                    return Err(ModuleError::new(format!(
                        "module `{name}`: kernel `{first}` was built for {a} and kernel `{}` for {t} \
                         (KernelOptions::target), but a module has one target",
                        k.name()
                    )));
                }
                (None, Some(t)) => target = Some((t, k.name())),
                _ => {}
            }
        }
        let mut globals: BTreeMap<String, (String, &str)> = BTreeMap::new();
        for k in &kernels {
            let ir = k.ir();
            for g in &ir.globals {
                let gname = &ir.buffer(g.buffer).name;
                if names.contains(gname.as_str()) {
                    return Err(ModuleError::new(format!(
                        "module `{name}`: kernel `{}` declares a module global named `{gname}`, which is also a \
                         kernel's name: they would be one symbol",
                        k.name()
                    )));
                }
                let decl = global_decl(ir, g);
                // An unnamed global (`module_global(None, ..)`) is its kernel's own, but its
                // automatic name is the same in every kernel: refuse rather than merge them.
                if gname.starts_with(UNNAMED_GLOBAL_PREFIX)
                    && let Some((_, by)) = globals.get(gname)
                {
                    return Err(ModuleError::new(format!(
                        "module `{name}`: kernels `{by}` and `{}` both have an unnamed module global `{gname}`, \
                         which would be one symbol of the module: give the globals names (module_global(Some(..), \
                         ..)), the same name to share one",
                        k.name()
                    )));
                }
                match globals.get(gname) {
                    Some((other, by)) if *other != decl => {
                        return Err(ModuleError::new(format!(
                            "module `{name}`: kernels `{by}` and `{}` declare module global `{gname}` differently \
                             (`{other}` and `{decl}`): kernels share a module's globals, so they must declare them \
                             alike",
                            k.name()
                        )));
                    }
                    Some(_) => {}
                    None => {
                        globals.insert(gname.clone(), (decl, k.name()));
                    }
                }
            }
        }
        // External functions are the module's: one prototype per symbol (m11-extern-distributed).
        let mut externs: BTreeMap<&str, (String, &str)> = BTreeMap::new();
        for k in &kernels {
            for d in &k.ir().externs {
                if names.contains(d.symbol.as_str()) {
                    return Err(ModuleError::new(format!(
                        "module `{name}`: kernel `{}` calls external function `{}`, which is also a kernel's name",
                        k.name(),
                        d.symbol
                    )));
                }
                let proto = d.prototype();
                match externs.get(d.symbol.as_str()) {
                    Some((other, by)) if *other != proto => {
                        return Err(ModuleError::new(format!(
                            "module `{name}`: kernels `{by}` and `{}` declare external function `{}` differently \
                             (`{other}` and `{proto}`)",
                            k.name(),
                            d.symbol
                        )));
                    }
                    Some(_) => {}
                    None => {
                        externs.insert(d.symbol.as_str(), (proto, k.name()));
                    }
                }
            }
        }
        Ok(KernelModule { name, kernels })
    }

    /// The module's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The kernels, in order.
    pub fn kernels(&self) -> &[Kernel] {
        &self.kernels
    }

    /// The kernel named `name`.
    pub fn kernel(&self, name: &str) -> Option<&Kernel> {
        self.kernels.iter().find(|k| k.name() == name)
    }

    /// The target the kernels were built for, if any was ([`Kernel::target`]).
    pub fn target(&self) -> Option<Arch> {
        self.kernels.iter().find_map(Kernel::target)
    }

    /// The module as MLIR (upstream dialects only): one `gpu.module` with every kernel.
    pub fn to_mlir(&self) -> String {
        self.to_mlir_with(&MlirOptions::default())
    }

    /// The module as MLIR, with options (e.g. source locations).
    pub fn to_mlir_with(&self, opts: &MlirOptions) -> String {
        match self.kernels.as_slice() {
            [k] => k.to_mlir_with(opts),
            _ => merge_mlir(&self.module_irs(), opts),
        }
    }

    /// Compile to PTX for `arch` at the default options, without `ptxas` (as
    /// [`Kernel::compile_ptx`]).
    pub fn compile_ptx(&self, toolchain: &Toolchain, arch: Arch) -> Result<String, CompileError> {
        self.compile_with(toolchain, &CompileOptions::new().gpu_arch(arch).use_ptxas(false))
            .map(|m| m.ptx().to_string())
    }

    /// The kernels' IR as emitted into the module: static shared buffers whose names another
    /// symbol of the module has are renamed.
    fn module_irs(&self) -> Vec<KernelIr> {
        let mut taken: BTreeSet<String> = self.kernels.iter().map(|k| k.name().to_string()).collect();
        taken.insert(DYNAMIC_SMEM_SYMBOL.to_string());
        for k in &self.kernels {
            let ir = k.ir();
            taken.extend(ir.globals.iter().map(|g| ir.buffer(g.buffer).name.clone()));
        }
        let mut out = Vec::with_capacity(self.kernels.len());
        for k in &self.kernels {
            let mut ir = k.ir().clone();
            for b in ir.buffers.iter_mut().filter(|b| b.space == MemSpace::Shared) {
                if taken.contains(&b.name) {
                    let mut renamed = format!("{}_{}", ir.name, b.name);
                    while taken.contains(&renamed) {
                        renamed.push('_');
                    }
                    b.name = renamed;
                }
                taken.insert(b.name.clone());
            }
            out.push(ir);
        }
        out
    }

    /// Compile with `options` (CuTe DSL's `cute.compile` of a host function that launches these
    /// kernels): one MLIR module, one `mlir-opt` lowering, one `ptxas` run. A module of one
    /// kernel is [`Kernel::compile_with`]. The target is the options', else the kernels'; it
    /// must suit every kernel ([`CompileError::Unsupported`] names the kernel that cannot run
    /// on it).
    pub fn compile_with(
        &self,
        toolchain: &Toolchain,
        options: &CompileOptions,
    ) -> Result<CompiledModule, CompileError> {
        if let [k] = self.kernels.as_slice() {
            let compiled = k.compile_with(toolchain, options)?;
            return Ok(CompiledModule {
                name: self.name.clone(),
                kernels: vec![compiled],
                notes: Vec::new(),
                whole: None,
            });
        }
        compile_module(self, toolchain, options)
    }
}

/// A [`KernelModule`] compiled: one PTX (and cubin), and one [`CompiledKernel`] per kernel,
/// each carrying the module's images under its own name, signature and diagnostics.
#[derive(Clone, Debug)]
pub struct CompiledModule {
    name: String,
    kernels: Vec<CompiledKernel>,
    notes: Vec<ArtifactNote>,
    /// The module as one compiled image, named after it, whose artifacts are `notes` (none
    /// for a module of one kernel).
    whole: Option<Box<CompiledKernel>>,
}

impl CompiledModule {
    /// The module's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The compiled kernels, in the module's order. Each loads the whole module.
    pub fn kernels(&self) -> &[CompiledKernel] {
        &self.kernels
    }

    /// The compiled kernel named `name`.
    pub fn kernel(&self, name: &str) -> Option<&CompiledKernel> {
        self.kernels.iter().find(|k| k.name() == name)
    }

    /// The kernels' names (their PTX entry points), in order.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.kernels.iter().map(CompiledKernel::name).collect()
    }

    /// The compiled kernels.
    pub fn into_kernels(self) -> Vec<CompiledKernel> {
        self.kernels
    }

    fn first(&self) -> &CompiledKernel {
        &self.kernels[0]
    }

    /// The module's PTX, with one `.entry` per kernel.
    pub fn ptx(&self) -> &str {
        self.first().ptx()
    }

    /// The module's cubin, when `ptxas` ran.
    pub fn cubin(&self) -> Option<&[u8]> {
        self.first().cubin()
    }

    /// The architecture the cubin's SASS is for.
    pub fn cubin_target(&self) -> Option<Arch> {
        self.first().cubin_target()
    }

    /// The target it was compiled for.
    pub fn arch(&self) -> Arch {
        self.first().arch()
    }

    /// The PTX ISA version (`10 * major + minor`): the newest any kernel needs.
    pub fn ptx_version(&self) -> u32 {
        self.first().ptx_version()
    }

    /// The options it was compiled with.
    pub fn options(&self) -> &CompileOptions {
        self.first().options()
    }

    /// The module's MLIR.
    pub fn mlir(&self) -> &str {
        self.first().mlir()
    }

    /// The `ptxas` log, when `ptxas` ran (every kernel's resources).
    pub fn ptxas_log(&self) -> Option<&str> {
        self.first().ptxas_log()
    }

    /// The artifacts kept for the module ([`CompileOptions::keep`]), named after the module.
    /// A module of one kernel keeps them as the kernel does ([`CompiledKernel::artifacts`]).
    pub fn artifacts(&self) -> &[ArtifactNote] {
        if self.kernels.len() == 1 { self.first().artifacts() } else { &self.notes }
    }

    /// Supply the module's cubin, built for `target`, when `ptxas` did not run
    /// ([`CompiledKernel::set_cubin`] for every kernel).
    pub fn set_cubin(&mut self, cubin: Vec<u8>, target: Arch, toolchain: &Toolchain) {
        for k in &mut self.kernels {
            k.cubin = Some(cubin.clone());
            k.cubin_target = Some(target);
            if self.whole.is_none() {
                k.refresh_artifacts(toolchain);
            }
        }
        if let Some(whole) = &mut self.whole {
            whole.cubin = Some(cubin);
            whole.cubin_target = Some(target);
            whole.refresh_artifacts(toolchain);
            self.notes = whole.notes.clone();
        }
    }
}

/// One kernel's MLIR, split for merging: the debug-info header, the module-level declarations
/// (`memref.global`) and the kernel function.
struct Parts {
    header: Vec<String>,
    decls: Vec<String>,
    func: Vec<String>,
    closing: Vec<String>,
}

fn split_parts(text: &str) -> Parts {
    let mut parts = Parts { header: vec![], decls: vec![], func: vec![], closing: vec![] };
    let mut stage = 0;
    for line in text.lines() {
        match stage {
            0 if line.starts_with("module attributes") => stage = 1,
            0 => parts.header.push(line.to_string()),
            1 if line.starts_with("  gpu.module @kernels") => stage = 2,
            2 if line.starts_with("    memref.global ") => parts.decls.push(line.to_string()),
            // External device functions' prototypes (m11-extern-distributed).
            2 if line.starts_with("    func.func private ") => parts.decls.push(line.to_string()),
            2 if line.starts_with("  }") => {
                stage = 3;
                parts.closing.push(line.to_string());
            }
            2 => parts.func.push(line.to_string()),
            _ => parts.closing.push(line.to_string()),
        }
    }
    parts
}

/// A declaration without its trailing location, to compare declarations across kernels.
fn without_loc(line: &str) -> &str {
    match line.rfind(" loc(") {
        Some(i) if line.ends_with(')') => &line[..i],
        _ => line,
    }
}

/// The symbol a `memref.global` line declares.
fn decl_symbol(line: &str) -> &str {
    let at = line.find('@').map_or(line.len(), |i| i + 1);
    let rest = &line[at..];
    &rest[..rest.find([' ', ':', '(']).unwrap_or(rest.len())]
}

/// The `alignment` of a `memref.global` line.
fn decl_alignment(line: &str) -> u64 {
    line.split("alignment = ").nth(1).and_then(|r| r.split(' ').next()).and_then(|n| n.parse().ok()).unwrap_or(0)
}

/// Kernel `i`'s debug-info aliases renamed apart from the other kernels' (`#di_k<i>_`), with
/// the module's one compile unit and subroutine type (kernel 0's) and a distinct subprogram id.
/// Only outside string literals: a printf format or an assertion message keeps its text.
fn rename_debug_info(line: &str, i: usize) -> String {
    if i == 0 || !line.contains("#di_") {
        return line.to_string();
    }
    let rename = |code: &str| {
        code.replace("#di_", &format!("#di_k{i}_"))
            .replace(&format!("#di_k{i}_cu"), "#di_cu")
            .replace(&format!("#di_k{i}_type"), "#di_type")
            .replace("distinct[1]<>", &format!("distinct[{}]<>", i + 1))
    };
    // Split at the quotes of MLIR string literals (`\"` escapes a quote inside one).
    let mut out = String::with_capacity(line.len() + 16);
    let (mut start, mut in_string, mut escaped) = (0, false, false);
    for (at, c) in line.char_indices() {
        if in_string && escaped {
            escaped = false;
        } else if in_string && c == '\\' {
            escaped = true;
        } else if c == '"' {
            let segment = &line[start..at];
            out.push_str(&if in_string { segment.to_string() } else { rename(segment) });
            start = at;
            in_string = !in_string;
        }
    }
    let rest = &line[start..];
    out.push_str(&if in_string { rest.to_string() } else { rename(rest) });
    out
}

/// The MLIR of one module with every kernel of `irs` (two or more).
fn merge_mlir(irs: &[KernelIr], opts: &MlirOptions) -> String {
    // Module globals declared by more than one kernel are declared as bytes by each, so that
    // their declarations agree whatever types each kernel accesses them with.
    let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    for ir in irs {
        for g in &ir.globals {
            *count.entry(ir.buffer(g.buffer).name.as_str()).or_default() += 1;
        }
    }
    let shared: BTreeSet<String> = count.into_iter().filter(|&(_, n)| n > 1).map(|(s, _)| s.to_string()).collect();
    let mut header: Vec<String> = Vec::new();
    let mut decls: Vec<String> = Vec::new();
    let mut funcs: Vec<String> = Vec::new();
    let mut closing = Vec::new();
    for (i, ir) in irs.iter().enumerate() {
        let parts = split_parts(&emit_mlir_in_module(ir, opts, &shared));
        for line in parts.header {
            let line = rename_debug_info(&line, i);
            // One compile unit and one subroutine type (kernel 0's) serve every kernel.
            if i > 0 && (line.starts_with("#di_cu ") || line.starts_with("#di_type ")) {
                continue;
            }
            header.push(line);
        }
        funcs.extend(parts.func.iter().map(|f| rename_debug_info(f, i)));
        for d in parts.decls {
            let sym = decl_symbol(&d).to_string();
            match decls.iter().position(|e| decl_symbol(e) == sym) {
                None => decls.push(d),
                Some(at) if sym == DYNAMIC_SMEM_SYMBOL => {
                    // The dynamic window: one extern, at the strictest alignment any kernel needs.
                    if decl_alignment(&d) > decl_alignment(&decls[at]) {
                        decls[at] = d;
                    }
                }
                Some(at) => debug_assert_eq!(
                    without_loc(&decls[at]),
                    without_loc(&d),
                    "KernelModule::new checks shared globals and module_irs renames shared buffers"
                ),
            }
        }
        if i == 0 {
            closing = parts.closing;
        }
    }
    let mut out = String::new();
    let lines = header
        .iter()
        .map(String::as_str)
        .chain(["module attributes {gpu.container_module} {", "  gpu.module @kernels {"])
        .chain(decls.iter().map(String::as_str))
        .chain(funcs.iter().map(String::as_str))
        .chain(closing.iter().map(String::as_str));
    for l in lines {
        out.push_str(l);
        out.push('\n');
    }
    out
}

/// [`KernelModule::compile_with`] for two or more kernels.
fn compile_module(
    module: &KernelModule,
    toolchain: &Toolchain,
    options: &CompileOptions,
) -> Result<CompiledModule, CompileError> {
    let mname = module.name();
    let arch = match (options.get_gpu_arch(), module.target()) {
        (Some(a), Some(t)) if a != t => {
            return Err(CompileError::Options(OptionError::new(format!(
                "module `{mname}` was built for {t} (KernelOptions::target), so it cannot be compiled for {a}"
            ))));
        }
        (Some(a), _) | (None, Some(a)) => a,
        (None, None) => {
            return Err(CompileError::Options(OptionError::new(format!(
                "no target architecture to compile module `{mname}` for: set one with CompileOptions::gpu_arch, \
                 `--gpu-arch sm_XX` or KernelOptions::target (compile_with does not read the environment; \
                 CompileOptions::with_env adds {})",
                super::ENV_ARCH
            ))));
        }
    };
    // The module's PTX ISA version is the newest any kernel needs.
    let mut need = 0;
    let mut why: Option<String> = None;
    for k in module.kernels() {
        let (n, w) = crate::validate::check_target_needs(k.ir(), arch).map_err(|e| {
            CompileError::Unsupported(format!(
                "kernel `{}` of module `{mname}` cannot run on {}: {e}",
                k.name(),
                arch.chip()
            ))
        })?;
        if n > need || why.is_none() && w.is_some() && n >= need {
            need = n;
            why = w.map(|w| format!("{w} in kernel `{}`", k.name()));
        }
    }
    let (chip, ptx_version, note) = toolchain
        .emission(arch, need, why.as_deref())
        .map_err(|e| CompileError::Unsupported(format!("module `{mname}`: {e}")))?;
    let line_info = options.effective_line_info();
    let opt_level = options.effective_opt_level();
    let mma_inline_ptx = options.get_mma_inline_ptx().unwrap_or(false);
    let llvm_major = toolchain.llvm_version();
    if !mma_inline_ptx {
        for k in module.kernels() {
            if let (Some((llvm, what)), Some(have)) = (nvvm_needs(k.ir()), toolchain.llvm_version())
                && have < llvm
            {
                return Err(CompileError::Unsupported(format!(
                    "kernel `{}` of module `{mname}` issues {what}, which the NVVM dialect has from LLVM {llvm}, but \
                     {} is LLVM {have}: set CUTE_MLIR_OPT to the mlir-opt of LLVM {llvm} or newer, or emit it as \
                     inline PTX with CompileOptions::mma_inline_ptx(true)",
                    k.name(),
                    toolchain.mlir_opt.display()
                )));
            }
        }
    }
    let started = Instant::now();
    crate::cute_log!(
        Level::Info,
        "compiling module `{mname}` ({} kernels) for {chip} (PTX ISA {}.{}, O{opt_level}{})",
        module.kernels().len(),
        ptx_version / 10,
        ptx_version % 10,
        if line_info { ", line info" } else { "" }
    );
    let ptxas_flags = options.get_ptxas_options().unwrap_or(&[]);
    let tools = toolchain.cuda_tools();
    let wanted = options
        .get_use_ptxas()
        .unwrap_or(!ptxas_flags.is_empty() || options.keeps(Artifact::Cubin) || options.keeps(Artifact::Sass));
    let ptxas = tools.ptxas.as_ref().filter(|_| wanted);
    if ptxas.is_none() {
        jit_options_for_ptxas_flags(ptxas_flags)?;
    }
    let assertions = options.effective_enable_assertions();
    let fp_contract = options.effective_fp_contract();
    let irs = module.module_irs();
    let link_libraries = super::link_libraries(&irs.iter().collect::<Vec<_>>(), options)?;
    let mlir_opts =
        |locations: bool| MlirOptions { locations, line_info, mma_inline_ptx, assertions, llvm_major, fp_contract };
    let mlir = merge_mlir(&irs, &mlir_opts(line_info));
    if options.get_print_ir() == Some(true) {
        println!("{mlir}");
    }
    let signatures: Vec<crate::KernelSignature> = module.kernels().iter().map(Kernel::signature).collect();
    let cacheable = options.get_no_cache() != Some(true)
        && !cache::no_cache_env()
        && options.hooks().is_empty()
        && !Artifact::ALL.into_iter().any(|a| options.keeps(a));
    let key = cacheable.then(|| {
        let ptxas_id = ptxas.map_or_else(
            || "none".to_string(),
            |t| format!("{} {:?} {}", t.path.display(), t.release, cache::file_identity(&t.path)),
        );
        let sigs: String = signatures.iter().map(|s| format!("signature {s}\n")).collect();
        format!(
            "cute-rs {VERSION}\nmodule {mname} of {} kernels\nmlir-opt {} {} LLVM {:?}\nptxas {ptxas_id}\npipeline \
             {:?}\ntarget {chip} {arch:?} PTX {ptx_version} O{opt_level} lineinfo {line_info} mma_inline_ptx \
             {mma_inline_ptx} assertions {assertions} ptxas_flags {ptxas_flags:?}\noptions {}\n{sigs}{mlir}",
            module.kernels().len(),
            toolchain.mlir_opt.display(),
            cache::file_identity(&toolchain.mlir_opt),
            toolchain.llvm_version(),
            toolchain.lowering_pipeline_for(chip, ptx_version, u32::from(opt_level)),
            options.cache_key(),
        ) + &super::link_key(&link_libraries)
    });
    let whole_ir: String =
        module.kernels().iter().map(|k| format!("// kernel `{}`\n{}", k.name(), dump(k.ir()))).collect();
    // The module's images, from the memory cache, the disk cache or a fresh compile.
    let cached = key.as_ref().and_then(|k| cache::memory_get(k));
    let whole = match cached {
        Some(hit) => {
            cache::note_hit(false, mname);
            (*hit).clone()
        }
        None => {
            let disk = key
                .as_ref()
                .and_then(|k| cache::cache_dir().map(|d| (d, format!("{k}\nbuild {}", cache::build_identity()))));
            let from_disk = disk.as_ref().and_then(|(d, k)| cache::disk_get(d, k));
            match (&key, &from_disk) {
                (Some(_), Some(_)) => cache::note_hit(true, mname),
                (Some(_), None) => cache::note_miss(mname),
                (None, _) => cache::count_bypass(),
            }
            let mut hooks: Vec<options::StageHook> = irs.iter().filter_map(byval_hook).collect();
            hooks.extend(options.hooks().iter().cloned());
            let lower = |mlir: &str| {
                toolchain.lower(
                    mlir,
                    chip,
                    ptx_version,
                    u32::from(opt_level),
                    mname,
                    &hooks,
                    options.keeps(Artifact::Llvm),
                    &link_libraries,
                )
            };
            let fresh = || -> Result<_, CompileError> {
                let (ptx, llvm) = match lower(&mlir) {
                    Err(CompileError::Mlir(m)) if !line_info => {
                        return Err(match lower(&merge_mlir(&irs, &mlir_opts(true))) {
                            Err(CompileError::Mlir(with_loc)) => CompileError::Mlir(with_loc),
                            _ => CompileError::Mlir(m),
                        });
                    }
                    r => r?,
                };
                let (mut cubin, mut ptxas_log, mut ptxas_command) = (None, None, None);
                if let Some(tool) = ptxas {
                    if let Some((major, minor)) = tool.release {
                        let max = ptxas_max_ptx(major, minor);
                        if ptx_version > max {
                            return Err(CompileError::Ptxas(format!(
                                "module `{mname}` is PTX ISA {}.{} (for {}), but {} is CUDA release {major}.{minor}, \
                                 which assembles PTX ISA {}.{} at most: use a newer toolkit, point CUTE_RS_PTXAS at \
                                 one, or turn ptxas off with CompileOptions::use_ptxas(false) to leave the PTX to the \
                                 driver",
                                ptx_version / 10,
                                ptx_version % 10,
                                ptx_reason_text(arch, why.as_deref()),
                                tool.path.display(),
                                max / 10,
                                max % 10
                            )));
                        }
                    }
                    let out = cudatools::run_ptxas(tool, chip, &ptx, u32::from(opt_level), line_info, ptxas_flags)?;
                    (cubin, ptxas_log, ptxas_command) = (Some(out.cubin), Some(out.log), Some(out.command));
                }
                Ok((ptx, llvm, cubin, ptxas_log, ptxas_command))
            };
            let (ptx, llvm, cubin, ptxas_log, ptxas_command) = match from_disk.clone() {
                Some(e) => (e.ptx, None, e.cubin, e.ptxas_log, e.ptxas_command),
                None => fresh()?,
            };
            let whole = CompiledKernel {
                name: mname.to_string(),
                arch,
                chip,
                note: note.clone(),
                ptx_reason: why.clone(),
                ptx_version,
                opt_level,
                line_info,
                options: options.clone(),
                ptx,
                ir_text: whole_ir.clone(),
                mlir: mlir.clone(),
                llvm,
                cubin_target: cubin.as_ref().map(|_| arch),
                cubin,
                ptxas_log,
                ptxas_command,
                sass: None,
                notes: Vec::new(),
                diagnostics: Vec::new(),
                resource_usage: None,
                signature: signatures[0].clone(),
            };
            if let Some(k) = &key {
                cache::memory_put(k, std::sync::Arc::new(whole.clone()));
                if let (Some((dir, disk_key)), None) = (&disk, &from_disk) {
                    let entry = cache::DiskEntry {
                        ptx: whole.ptx.clone(),
                        cubin: whole.cubin.clone(),
                        ptxas_log: whole.ptxas_log.clone(),
                        ptxas_command: whole.ptxas_command.clone(),
                    };
                    cache::disk_put(dir, disk_key, &entry);
                }
            }
            whole
        }
    };
    let mut whole = CompiledKernel { options: options.clone(), ir_text: whole_ir, mlir, ..whole };
    if options.keeps(Artifact::Mlir) {
        whole.mlir = merge_mlir(&irs, &mlir_opts(true));
    }
    let ptxas_ran = tools.ptxas.is_some() && ptxas.is_some();
    let mut kernels = Vec::with_capacity(module.kernels().len());
    for (k, sig) in module.kernels().iter().zip(signatures) {
        let mut c = CompiledKernel {
            name: k.name().to_string(),
            ir_text: dump(k.ir()),
            signature: crate::KernelSignature { target: Some(arch), ptx_version: Some(ptx_version), ..sig },
            ..whole.clone()
        };
        c.collect_diagnostics(k.ir().loc, ptxas_ran)?;
        kernels.push(c);
    }
    whole.refresh_artifacts(toolchain);
    crate::cute_log!(
        Level::Info,
        "module `{mname}`: compiled {} kernels in {:.1} ms",
        kernels.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    Ok(CompiledModule { name: mname.to_string(), kernels, notes: whole.notes.clone(), whole: Some(Box::new(whole)) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaration_helpers() {
        let d = "    memref.global \"private\" @smem : memref<4xf32, 3> {alignment = 16 : i64} loc(\"a.rs\":1:2)";
        assert_eq!(decl_symbol(d), "smem");
        assert_eq!(decl_alignment(d), 16);
        assert!(without_loc(d).ends_with("i64}"));
        assert_eq!(
            rename_debug_info("#di_sp = x distinct[1]<> #di_cu #di_file0", 2),
            "#di_k2_sp = x distinct[3]<> #di_cu #di_k2_file0"
        );
        assert_eq!(
            rename_debug_info(r##"gpu.printf "#di_file \"#di_sp\" distinct[1]<>" loc(fused<#di_lbf0>[x])"##, 1),
            r##"gpu.printf "#di_file \"#di_sp\" distinct[1]<>" loc(fused<#di_k1_lbf0>[x])"##
        );
    }
}

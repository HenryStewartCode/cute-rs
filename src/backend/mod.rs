//! Code generation: a readable IR listing, MLIR, and PTX via `mlir-opt`.

mod artifacts;
pub mod cache;
mod cudatools;
mod mlir;
mod options;
mod passes;
mod stage;
mod toolchain;
// Several kernels in one module (m6b-multi-kernel).
mod module;
pub use module::{CompiledModule, KernelModule, ModuleError};

use std::fmt::Write;

pub use artifacts::{Artifact, ArtifactNote};
pub use cache::{CacheStats, CachedKernel, KernelCache, cache_stats};
pub use cudatools::{CudaTools, ENV_PTXAS, JitOption, Tool, jit_options_for_ptxas_flags, ptxas_max_ptx};
// m11-extern-distributed: relocatable device code.
pub(crate) use cudatools::run_ptxas;
pub use mlir::{MlirOptions, emit_mlir};
pub(crate) use options::parse_bool;
pub use options::{CompileOptions, DEFAULT_OPT_LEVEL, ENV_ARCH, ENV_COMPILER_OPT, ENV_LINEINFO, OptionError};
pub use passes::LoweringPipeline;
pub use stage::LoweredModule;
pub use toolchain::{
    Arch, CUTE_DSL_VERSION, CompileError, ParseArchError, TargetSupport, Toolchain, VERSION, best_target,
    driver_ptx_ceiling,
};

use std::path::PathBuf;
use std::time::Instant;

use crate::builder::Kernel;
use crate::ir::{Inst, KernelIr, LdMatrixForm, MemSpace, Op, Param, Region};
use crate::log::Level;

impl Kernel {
    /// The kernel as MLIR (upstream dialects only).
    pub fn to_mlir(&self) -> String {
        emit_mlir(self.ir(), &MlirOptions::default())
    }

    /// The kernel as MLIR, with options (e.g. source locations).
    pub fn to_mlir_with(&self, opts: &MlirOptions) -> String {
        emit_mlir(self.ir(), opts)
    }

    /// The oldest LLVM (major version) whose NVVM dialect has every operation this kernel
    /// emits in NVVM form, if that is newer than what cute-rs otherwise needs: 23 for an FP8
    /// `mma.sync`, 22 for `mma.sp` and the `.m16n16` and `.m8n16` forms of `ldmatrix`.
    /// [`CompileOptions::mma_inline_ptx`] emits those as inline PTX instead.
    pub fn min_llvm_version(&self) -> Option<u32> {
        nvvm_needs(self.ir()).map(|(v, _)| v)
    }

    /// Compile to PTX for `arch` at the default options: a wrapper around
    /// [`compile_with`](Self::compile_with). The kernel's parameters are
    /// passed as bare pointers (for buffers) and scalars, in declaration
    /// order. Fails with [`CompileError::Unsupported`] if an instruction needs
    /// a newer architecture than `arch`; the PTX ISA version is raised when an
    /// instruction needs it. Only the PTX is made: `ptxas` is not run.
    pub fn compile_ptx(&self, toolchain: &Toolchain, arch: Arch) -> Result<String, CompileError> {
        self.compile_with(toolchain, &CompileOptions::new().gpu_arch(arch).use_ptxas(false))
            .map(CompiledKernel::into_ptx)
    }

    /// Compile with `options` (CuTe DSL's `cute.compile`), which must name the target
    /// architecture. The options are used as given; add the environment with
    /// [`CompileOptions::with_env`].
    ///
    /// The kernel is checked against the target first: an instruction the architecture lacks
    /// fails with [`CompileError::Unsupported`], naming its source line, and the PTX ISA
    /// version is raised when an instruction needs a newer one. It is then lowered in the two
    /// stages of [`Toolchain::lowering_pipeline`], running the options' stage hooks in
    /// between.
    pub fn compile_with(
        &self,
        toolchain: &Toolchain,
        options: &CompileOptions,
    ) -> Result<CompiledKernel, CompileError> {
        let arch = match (options.get_gpu_arch(), self.target()) {
            (Some(a), Some(t)) if a != t => {
                return Err(CompileError::Options(OptionError::new(format!(
                    "kernel `{}` was built for {t} (KernelOptions::target), so it cannot be compiled for {a}",
                    self.name()
                ))));
            }
            (Some(a), _) | (None, Some(a)) => a,
            (None, None) => {
                return Err(CompileError::Options(OptionError::new(format!(
                    "no target architecture to compile kernel `{}` for: set one with CompileOptions::gpu_arch, \
                     `--gpu-arch sm_XX` or KernelOptions::target (compile_with does not read the environment; \
                     CompileOptions::with_env adds {ENV_ARCH})",
                    self.name()
                ))));
            }
        };
        let (need, why) = crate::validate::check_target_needs(self.ir(), arch).map_err(|e| {
            CompileError::Unsupported(format!("kernel `{}` cannot run on {}: {e}", self.name(), arch.chip()))
        })?;
        let (chip, ptx_version, note) = toolchain
            .emission(arch, need, why.as_deref())
            .map_err(|e| CompileError::Unsupported(format!("kernel `{}`: {e}", self.name())))?;
        let line_info = options.effective_line_info();
        let opt_level = options.effective_opt_level();
        let mma_inline_ptx = options.get_mma_inline_ptx().unwrap_or(false);
        let llvm_major = toolchain.llvm_version();
        if !mma_inline_ptx && let Some((llvm, what)) = nvvm_needs(self.ir()) {
            match toolchain.llvm_version() {
                Some(have) if have < llvm => {
                    return Err(CompileError::Unsupported(format!(
                        "kernel `{}` issues {what}, which the NVVM dialect has from LLVM {llvm}, but {} is LLVM \
                         {have}: set CUTE_MLIR_OPT to the mlir-opt of LLVM {llvm} or newer, or emit it as inline \
                         PTX with CompileOptions::mma_inline_ptx(true)",
                        self.name(),
                        toolchain.mlir_opt.display()
                    )));
                }
                _ => {}
            }
        }
        let started = Instant::now();
        crate::cute_log!(
            Level::Info,
            "compiling kernel `{}` for {chip} (PTX ISA {}.{}, O{opt_level}{})",
            self.name(),
            ptx_version / 10,
            ptx_version % 10,
            if line_info { ", line info" } else { "" }
        );
        // The user's ptxas flags are checked up front, so that a flag nothing here can honour
        // is reported on every machine and not only where the toolkit is missing.
        let ptxas_flags = options.get_ptxas_options().unwrap_or(&[]);
        let tools = toolchain.cuda_tools();
        // ptxas runs when asked for: `use_ptxas(true)`, ptxas flags, or a kept cubin or SASS.
        let wanted = options
            .get_use_ptxas()
            .unwrap_or(!ptxas_flags.is_empty() || options.keeps(Artifact::Cubin) || options.keeps(Artifact::Sass));
        let ptxas = tools.ptxas.as_ref().filter(|_| wanted);
        if ptxas.is_none() {
            jit_options_for_ptxas_flags(ptxas_flags)?;
        }
        let assertions = options.effective_enable_assertions();
        let fp_contract = options.effective_fp_contract();
        let link_libraries = link_libraries(&[self.ir()], options)?;
        let mlir = emit_mlir(
            self.ir(),
            &MlirOptions { locations: line_info, line_info, mma_inline_ptx, assertions, llvm_major, fp_contract },
        );
        crate::cute_log!(
            Level::Info,
            "kernel `{}`: emitted {} bytes of MLIR in {:.1} ms",
            self.name(),
            mlir.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
        if options.get_print_ir() == Some(true) {
            println!("{mlir}");
        }
        // The compile caches (see `cache`): bypassed for stage hooks, kept artifacts and
        // `no_cache`.
        let cacheable = options.get_no_cache() != Some(true)
            && !cache::no_cache_env()
            && options.hooks().is_empty()
            && !Artifact::ALL.into_iter().any(|a| options.keeps(a));
        let key = cacheable.then(|| {
            let ptxas_id = ptxas.map_or_else(
                || "none".to_string(),
                |t| format!("{} {:?} {}", t.path.display(), t.release, cache::file_identity(&t.path)),
            );
            format!(
                "cute-rs {VERSION}\nmlir-opt {} {} LLVM {:?}\nptxas {ptxas_id}\npipeline {:?}\ntarget {chip} {arch:?} PTX \
                 {ptx_version} O{opt_level} lineinfo {line_info} mma_inline_ptx {mma_inline_ptx} assertions \
                 {assertions} ptxas_flags {ptxas_flags:?}\noptions {}\nsignature {}\n{mlir}",
                toolchain.mlir_opt.display(),
                cache::file_identity(&toolchain.mlir_opt),
                toolchain.llvm_version(),
                toolchain.lowering_pipeline_for(chip, ptx_version, u32::from(opt_level)),
                options.cache_key(),
                self.signature()
            ) + &link_key(&link_libraries)
        });
        let refresh = |mut c: CompiledKernel| -> Result<CompiledKernel, CompileError> {
            c.ir_text = dump(self.ir());
            c.options = options.clone();
            c.signature =
                crate::KernelSignature { target: Some(arch), ptx_version: Some(ptx_version), ..self.signature() };
            c.collect_diagnostics(self.ir().loc, tools.ptxas.is_some() && ptxas.is_some())?;
            Ok(c)
        };
        if let Some(k) = &key
            && let Some(hit) = cache::memory_get(k)
        {
            cache::note_hit(false, self.name());
            return refresh((*hit).clone());
        }
        let disk = key
            .as_ref()
            .and_then(|k| cache::cache_dir().map(|d| (d, format!("{k}\nbuild {}", cache::build_identity()))));
        let from_disk = disk.as_ref().and_then(|(d, k)| cache::disk_get(d, k));
        match (&key, &from_disk) {
            (Some(_), Some(_)) => cache::note_hit(true, self.name()),
            (Some(_), None) => cache::note_miss(self.name()),
            (None, _) => cache::count_bypass(),
        }
        // By-value parameters become `.param` aggregates: `llvm.byval` (and
        // `nvvm.grid_constant`) on their pointers, which only the lowered module can carry.
        let mut hooks: Vec<options::StageHook> = Vec::new();
        if let Some(hook) = byval_hook(self.ir()) {
            hooks.push(hook);
        }
        hooks.extend(options.hooks().iter().cloned());
        let lower = |mlir: &str| {
            toolchain.lower(
                mlir,
                chip,
                ptx_version,
                u32::from(opt_level),
                self.name(),
                &hooks,
                options.keeps(Artifact::Llvm),
                &link_libraries,
            )
        };
        let fresh = || -> Result<_, CompileError> {
            let (ptx, llvm) = match lower(&mlir) {
                // Without locations mlir-opt cannot say which line failed: lower again with them,
                // so that the error names the user's line (CompileError::loc).
                Err(CompileError::Mlir(m)) if !line_info => {
                    let located = emit_mlir(
                        self.ir(),
                        &MlirOptions {
                            locations: true,
                            line_info,
                            mma_inline_ptx,
                            assertions,
                            llvm_major,
                            fp_contract,
                        },
                    );
                    return Err(match lower(&located) {
                        Err(CompileError::Mlir(with_loc)) => CompileError::Mlir(with_loc),
                        _ => CompileError::Mlir(m),
                    });
                }
                r => r?,
            };
            // m11-prims-mma-forms: an intrinsic the NVPTX backend does not know becomes a call
            // of an external function, which nothing can link.
            if let Some(why) = unknown_intrinsic(&ptx) {
                return Err(CompileError::Unsupported(format!("kernel `{}`: {why}", self.name())));
            }
            let (mut cubin, mut ptxas_log, mut ptxas_command) = (None, None, None);
            if let Some(tool) = ptxas {
                if let Some((major, minor)) = tool.release {
                    let max = ptxas_max_ptx(major, minor);
                    if ptx_version > max {
                        let why = ptx_reason_text(arch, why.as_deref());
                        return Err(CompileError::Ptxas(format!(
                            "kernel `{}` is PTX ISA {}.{} (for {why}), but {} is CUDA release {major}.{minor}, which \
                         assembles PTX ISA {}.{} at most: use a newer toolkit, point CUTE_RS_PTXAS at one, or turn \
                         ptxas off with CompileOptions::use_ptxas(false) to leave the PTX to the driver",
                            self.name(),
                            ptx_version / 10,
                            ptx_version % 10,
                            tool.path.display(),
                            max / 10,
                            max % 10
                        )));
                    }
                }
                let t = Instant::now();
                let out = cudatools::run_ptxas(tool, chip, &ptx, u32::from(opt_level), line_info, ptxas_flags)?;
                crate::cute_log!(
                    Level::Info,
                    "kernel `{}`: ptxas made {} bytes of cubin in {:.1} ms",
                    self.name(),
                    out.cubin.len(),
                    t.elapsed().as_secs_f64() * 1e3
                );
                (cubin, ptxas_log, ptxas_command) = (Some(out.cubin), Some(out.log), Some(out.command));
            }
            Ok((ptx, llvm, cubin, ptxas_log, ptxas_command))
        };
        let (ptx, llvm, cubin, ptxas_log, ptxas_command) = match from_disk.clone() {
            Some(e) => (e.ptx, None, e.cubin, e.ptxas_log, e.ptxas_command),
            None => fresh()?,
        };
        let mut compiled = CompiledKernel {
            name: self.name().to_string(),
            arch,
            chip,
            note,
            ptx_reason: why,
            ptx_version,
            opt_level,
            line_info,
            options: options.clone(),
            ptx,
            ir_text: dump(self.ir()),
            mlir,
            llvm,
            cubin_target: cubin.as_ref().map(|_| arch),
            cubin,
            ptxas_log,
            ptxas_command,
            sass: None,
            notes: Vec::new(),
            diagnostics: Vec::new(),
            resource_usage: None,
            signature: crate::KernelSignature {
                target: Some(arch),
                ptx_version: Some(ptx_version),
                ..self.signature()
            },
        };
        compiled.collect_diagnostics(self.ir().loc, tools.ptxas.is_some() && ptxas.is_some())?;
        if options.keeps(Artifact::Mlir) {
            // Kept MLIR carries locations whether or not line information is on.
            compiled.mlir = emit_mlir(
                self.ir(),
                &MlirOptions { locations: true, line_info, mma_inline_ptx, assertions, llvm_major, fp_contract },
            );
        }
        compiled.refresh_artifacts(toolchain);
        crate::cute_log!(
            Level::Info,
            "kernel `{}`: compiled in {:.1} ms",
            self.name(),
            started.elapsed().as_secs_f64() * 1e3
        );
        if let Some(k) = &key {
            cache::memory_put(k, std::sync::Arc::new(compiled.clone()));
            if let (Some((dir, disk_key)), None) = (&disk, &from_disk) {
                let entry = cache::DiskEntry {
                    ptx: compiled.ptx.clone(),
                    cubin: compiled.cubin.clone(),
                    ptxas_log: compiled.ptxas_log.clone(),
                    ptxas_command: compiled.ptxas_command.clone(),
                };
                cache::disk_put(dir, disk_key, &entry);
            }
        }
        Ok(compiled)
    }
}

/// The bitcode files to link into a module of `irs` compiled with `options`: the kernels'
/// external functions' sources, then the options' `link_libraries`, without repeats. Each
/// must exist (m11-extern-distributed).
fn link_libraries(irs: &[&KernelIr], options: &CompileOptions) -> Result<Vec<PathBuf>, CompileError> {
    let mut out: Vec<PathBuf> = Vec::new();
    let from_kernels = irs.iter().flat_map(|ir| ir.link_libraries.iter().map(move |p| (p, Some(ir.name.as_str()))));
    let from_options = options.get_link_libraries().iter().map(|p| (p, None));
    for (path, kernel) in from_kernels.chain(from_options) {
        if !path.is_file() {
            let from = match kernel {
                Some(k) => format!("an external function of kernel `{k}` names it as its source (BitCode)"),
                None => "CompileOptions::link_libraries names it".to_string(),
            };
            return Err(CompileError::Options(OptionError::new(format!(
                "bitcode file {} does not exist: {from}",
                path.display()
            ))));
        }
        if !out.contains(path) {
            out.push(path.clone());
        }
    }
    Ok(out)
}

/// The cache-key lines of linked bitcode: each file with its identity, so that rebuilding
/// one invalidates what was compiled with it.
fn link_key(libraries: &[PathBuf]) -> String {
    libraries.iter().map(|p| format!("\nlink {} {}", p.display(), cache::file_identity(p))).collect()
}

fn ptx_reason_text(arch: Arch, why: Option<&str>) -> String {
    why.map_or_else(|| format!("its target, {}", arch.chip()), str::to_string)
}

/// m11-prims-mma-forms: the diagnostic for PTX that declares an `llvm.*`
/// function (`.extern .func ... llvm.nvvm.<name>`): what `llc` makes of a
/// call to an intrinsic it does not know (a misspelt
/// `llvm.call_intrinsic` name, or one newer than the backend), which no
/// loader resolves.
pub(crate) fn unknown_intrinsic(ptx: &str) -> Option<String> {
    let line = ptx.lines().find(|l| l.trim_start().starts_with(".extern .func") && l.contains(" llvm."))?;
    let symbol = line.split(['(', ')', ' ', '\t', ';']).find(|w| w.starts_with("llvm.")).unwrap_or("?");
    Some(format!(
        "its PTX declares `{symbol}` as an external function: the NVPTX backend did not know this intrinsic \
         (check its name, or use an LLVM that has it)"
    ))
}

/// The newest NVVM operation `ir` emits, as `(LLVM major version, what)`:
/// FP8 `nvvm.mma.sync` (LLVM 23), and `nvvm.mma.sp.sync` and the 8-bit
/// shapes of `nvvm.ldmatrix` (LLVM 22).
fn nvvm_needs(ir: &crate::ir::KernelIr) -> Option<(u32, &'static str)> {
    fn walk(r: &crate::ir::Region, best: &mut Option<(u32, &'static str)>) {
        for inst in &r.insts {
            let need = match &inst.op {
                Op::Mma { kind, .. } if kind.is_fp8() => Some((23, "an FP8 mma.sync")),
                Op::Mma { kind, .. } if kind.is_sparse() => Some((22, "mma.sp")),
                // m10-sm120-blockscaled: nvvm.mma.block_scale arrived in LLVM 22.
                Op::MmaBlockScaled { .. } => Some((22, "a block-scaled mma.sync")),
                // LLVM 21's nvvm.ldmatrix has only .m8n8.b16; LLVM 22 adds
                // the shape and element-type attributes.
                Op::LdMatrix { form, .. } if *form != LdMatrixForm::M8n8B16 => {
                    Some((22, "an ldmatrix of shape .m16n16 or .m8n16"))
                }
                // LLVM 21's nvvm.stmatrix has only .m8n8.b16.
                Op::StMatrix { form: crate::ir::StMatrixForm::M16n8B8, .. } => Some((22, "stmatrix.m16n8")),
                _ => None,
            };
            if need.is_some_and(|(v, _)| best.is_none_or(|(b, _)| v > b)) {
                *best = need;
            }
            inst.op.regions().into_iter().for_each(|r| walk(r, best));
        }
    }
    let mut best = None;
    walk(&ir.body, &mut best);
    best
}

/// A kernel compiled to PTX ([`Kernel::compile_with`]): CuTe DSL's compiled function, without
/// the host side.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    name: String,
    arch: Arch,
    chip: &'static str,
    note: Option<String>,
    ptx_reason: Option<String>,
    ptx_version: u32,
    opt_level: u8,
    line_info: bool,
    options: CompileOptions,
    ptx: String,
    ir_text: String,
    mlir: String,
    llvm: Option<String>,
    cubin: Option<Vec<u8>>,
    /// The architecture the cubin's SASS is for: the kernel's with `ptxas`, the device's with
    /// the driver's linker.
    cubin_target: Option<Arch>,
    ptxas_log: Option<String>,
    ptxas_command: Option<String>,
    sass: Option<String>,
    notes: Vec<ArtifactNote>,
    diagnostics: Vec<crate::Diagnostic>,
    resource_usage: Option<crate::ResourceUsage>,
    signature: crate::KernelSignature,
}

impl CompiledKernel {
    /// The warnings and remarks the compile reported (CuTe DSL's compiler diagnostics),
    /// as [`CompileOptions::warnings`] and [`CompileOptions::remarks`] ask, after the
    /// filters.
    pub fn diagnostics(&self) -> &[crate::Diagnostic] {
        &self.diagnostics
    }

    /// The kernel's resources as `ptxas -v` reported them, when `ptxas` ran.
    pub fn resource_usage(&self) -> Option<&crate::ResourceUsage> {
        self.resource_usage.as_ref()
    }

    /// Read the resource usage from the ptxas log and gather the diagnostics the options
    /// ask for, at the kernel's `Kernel::build` line `build`.
    fn collect_diagnostics(&mut self, build: crate::ir::Loc, ptxas_ran: bool) -> Result<(), CompileError> {
        use crate::diagnostics::{Diagnostic, Severity, wants};
        self.resource_usage =
            self.ptxas_log.as_deref().and_then(|log| crate::ResourceUsage::from_ptxas_log(log, &self.name));
        let settings = self.options.diagnostic_settings();
        let at = Some(crate::SourceLoc::from(build));
        let mut found: Vec<Diagnostic> = Vec::new();
        let (warn, remark) = (settings.warnings, settings.remarks);
        if wants(remark, "target")
            && let Some(note) = &self.note
        {
            found.push(Diagnostic::new(Severity::Remark, "target", note.clone(), at.clone()));
        }
        if wants(warn, "ptx") || wants(remark, "ptx") {
            match &self.resource_usage {
                Some(u) => {
                    let (stores, loads) = (u.spill_store_bytes.unwrap_or(0), u.spill_load_bytes.unwrap_or(0));
                    if wants(warn, "ptx") && u.spills() {
                        found.push(Diagnostic::new(
                            Severity::Warning,
                            "ptx",
                            format!(
                                "kernel `{}` spills registers: {stores} bytes of spill stores and {loads} bytes of \
                                 spill loads, with {} registers",
                                self.name,
                                u.registers.unwrap_or(0)
                            ),
                            at.clone(),
                        ));
                    }
                    if wants(remark, "ptx") {
                        let stack = u.stack_bytes.unwrap_or(0);
                        if stack > 0 {
                            found.push(Diagnostic::new(
                                Severity::Remark,
                                "ptx",
                                format!("kernel `{}` uses {stack} bytes of local memory (stack frame)", self.name),
                                at.clone(),
                            ));
                        }
                        found.push(Diagnostic::new(
                            Severity::Remark,
                            "ptx",
                            format!(
                                "kernel `{}`: {} registers, {stack} bytes stack, {stores} bytes spill stores, {loads} \
                                 bytes spill loads, {} bytes shared, {} bytes constant memory",
                                self.name,
                                u.registers.unwrap_or(0),
                                u.shared_bytes.unwrap_or(0),
                                u.constant_bytes.unwrap_or(0)
                            ),
                            at.clone(),
                        ));
                    }
                }
                None if wants(remark, "ptx") => found.push(Diagnostic::new(
                    Severity::Remark,
                    "ptx",
                    if ptxas_ran {
                        "the ptxas log has no resource report for the kernel".to_string()
                    } else {
                        "ptxas did not run, so the resource usage comes from the driver when the kernel is loaded \
                         (CudaFunction::resource_usage, or ResourceUsage::from_jit_log of a verbose JIT log)"
                            .to_string()
                    },
                    at.clone(),
                )),
                None => {}
            }
        }
        if wants(remark, "loop") {
            found.push(Diagnostic::new(
                Severity::Remark,
                "loop",
                "loop remarks are unavailable: mlir-opt has no -pass-remarks option (only --lto-pass-remarks-*, \
                 which the NVPTX pipeline does not run)",
                at.clone(),
            ));
        }
        found.retain(|d| match d.severity {
            Severity::Warning => !settings.ignore && settings.warning_filter.is_none_or(|f| d.matches(f)),
            Severity::Remark => settings.remark_filter.is_none_or(|f| d.matches(f)),
            Severity::Error => true,
        });
        if let Some(path) = settings.remark_output {
            let yaml: String =
                found.iter().filter(|d| d.severity == Severity::Remark).map(Diagnostic::to_yaml).collect();
            std::fs::write(path, yaml)
                .map_err(|e| CompileError::Tool(format!("writing the remarks to {}: {e}", path.display())))?;
        }
        // Asked for, so shown, as CuTe DSL shows them.
        for d in &found {
            eprintln!("{d}");
        }
        if settings.as_errors && found.iter().any(|d| d.severity == Severity::Warning) {
            return Err(CompileError::Warnings(
                found.into_iter().filter(|d| d.severity == Severity::Warning).collect(),
            ));
        }
        self.diagnostics = found;
        Ok(())
    }

    /// The kernel's name, which is also its PTX entry point.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The kernel's launch signature, with the target and PTX version it was compiled for:
    /// what the CUDA launcher checks and packs arguments by.
    pub fn signature(&self) -> &crate::KernelSignature {
        &self.signature
    }

    /// The target architecture.
    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// The PTX ISA version, as `10 * major + minor` (70 for 7.0).
    pub fn ptx_version(&self) -> u32 {
        self.ptx_version
    }

    /// The instruction that raised the PTX ISA version above the target's
    /// own minimum ([`Arch::min_ptx`]), if one did, with its source line.
    pub fn ptx_reason(&self) -> Option<&str> {
        self.ptx_reason.as_deref()
    }

    /// The name of the target in the PTX's `.target`: the architecture's
    /// own, or for a Thor target that the toolchain only knows by its name
    /// before PTX ISA 9.0, that one (`sm_101a` for `sm_110a`), with a
    /// [`note`](Self::note).
    pub fn target_name(&self) -> &str {
        self.chip
    }

    /// Why the PTX names another target than [`arch`](Self::arch), if it
    /// does.
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    /// The optimization level it was compiled at.
    pub fn opt_level(&self) -> u32 {
        u32::from(self.opt_level)
    }

    /// Whether it carries line information.
    pub fn line_info(&self) -> bool {
        self.line_info
    }

    /// The options it was compiled with.
    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    /// The PTX.
    pub fn ptx(&self) -> &str {
        &self.ptx
    }

    /// The PTX, by value.
    pub fn into_ptx(self) -> String {
        self.ptx
    }

    /// The options the CUDA driver's JIT compiles the PTX with (`CudaContext::load`, with
    /// feature `cuda`, passes them to `cuModuleLoadDataEx`): the optimization level (3 becomes
    /// the JIT's default, 4, which matches `ptxas -O3`), with line information
    /// `CU_JIT_GENERATE_LINE_INFO`, and what the options' `ptxas` flags map onto
    /// ([`jit_options_for_ptxas_flags`]: `--maxrregcount`, `-O<n>`, `-lineinfo`, `-v`). A flag
    /// with no JIT equivalent is left out here; [`try_jit_options`](Self::try_jit_options)
    /// reports it.
    pub fn jit_options(&self) -> Vec<JitOption> {
        self.try_jit_options().unwrap_or_else(|_| self.base_jit_options())
    }

    /// [`jit_options`](Self::jit_options), failing, with a message that names `ptxas`, on a
    /// `ptxas` flag the driver's JIT has no option for.
    pub fn try_jit_options(&self) -> Result<Vec<JitOption>, OptionError> {
        let mut options = self.base_jit_options();
        let flags = self.options.get_ptxas_options().unwrap_or(&[]);
        for o in jit_options_for_ptxas_flags(flags)? {
            options.retain(|e| e.code() != o.code());
            options.push(o);
        }
        Ok(options)
    }

    fn base_jit_options(&self) -> Vec<JitOption> {
        let level = match self.opt_level {
            3 => 4,
            n => u32::from(n),
        };
        let mut options = vec![JitOption::OptimizationLevel(level)];
        if self.line_info {
            options.push(JitOption::GenerateLineInfo(true));
        }
        options
    }

    /// The cubin, if one was made: by `ptxas` during compilation, or from the driver's linker
    /// ([`set_cubin`](Self::set_cubin)).
    pub fn cubin(&self) -> Option<&[u8]> {
        self.cubin.as_deref()
    }

    /// The architecture the cubin was built for, if there is one: the kernel's target when
    /// `ptxas` made it, the device's when the driver's linker did. `CudaContext::load` uses the
    /// cubin only on a device that runs binaries built for it.
    pub fn cubin_target(&self) -> Option<Arch> {
        self.cubin_target
    }

    /// What `ptxas -v` printed (register, shared and local memory use), if it ran.
    pub fn ptxas_log(&self) -> Option<&str> {
        self.ptxas_log.as_deref()
    }

    /// The `ptxas` command line that made the cubin, if it ran.
    pub fn ptxas_command(&self) -> Option<&str> {
        self.ptxas_command.as_deref()
    }

    /// The SASS, if it was asked for (`keep_sass`) and could be made.
    pub fn sass(&self) -> Option<&str> {
        self.sass.as_deref()
    }

    /// The MLIR the kernel was compiled from; with the MLIR kept
    /// ([`CompileOptions::keep`]) it carries source locations.
    pub fn mlir(&self) -> &str {
        &self.mlir
    }

    /// [`Kernel::dump`](crate::Kernel::dump): the readable IR listing.
    pub fn ir(&self) -> &str {
        &self.ir_text
    }

    /// The module after the first lowering stage (LLVM dialect), if it was asked for
    /// (`keep(Artifact::Llvm, true)`).
    pub fn llvm(&self) -> Option<&str> {
        self.llvm.as_deref()
    }

    /// What happened to each artifact that was asked for: where it was written, or why it
    /// could not be made or written. (Each unavailable one is also reported on stderr.)
    pub fn artifacts(&self) -> &[ArtifactNote] {
        &self.notes
    }

    /// The SASS of the cubin, made now: `nvdisasm` with the options' flags (`-g -c` by
    /// default), else `cuobjdump -sass`. With line information the SASS carries the Rust source
    /// lines.
    pub fn disassemble(&self, toolchain: &Toolchain) -> Result<String, String> {
        let cubin = self.cubin.as_deref().ok_or_else(|| self.no_cubin_reason())?;
        cudatools::disassemble(toolchain.cuda_tools(), cubin, &self.options.effective_nvdisasm())
    }

    /// Supply the cubin, built for `target`, for when `ptxas` was not available at compile time
    /// (the driver's linker made it for its device: `CudaContext::link_cubin`). Makes the SASS
    /// and writes the artifacts again if they are kept.
    pub fn set_cubin(&mut self, cubin: Vec<u8>, target: Arch, toolchain: &Toolchain) {
        self.cubin = Some(cubin);
        self.cubin_target = Some(target);
        self.refresh_artifacts(toolchain);
    }

    fn no_cubin_reason(&self) -> String {
        "no cubin: ptxas was not found or was turned off (it is looked up in $CUTE_RS_PTXAS, $CUDA_HOME/bin, \
         $CUDA_PATH/bin, PATH and /usr/local/cuda/bin), and on a GPU machine CudaContext::link_cubin makes one \
         with the driver's linker"
            .to_string()
    }

    /// Make the SASS if asked for, then write what is kept and record the outcome.
    fn refresh_artifacts(&mut self, toolchain: &Toolchain) {
        let mut sass_error = None;
        if self.options.keeps(Artifact::Sass) {
            self.sass = None;
            if self.cubin.is_some() {
                let t = Instant::now();
                match self.disassemble(toolchain) {
                    Ok(text) => {
                        crate::cute_log!(
                            Level::Info,
                            "kernel `{}`: SASS made in {:.1} ms",
                            self.name,
                            t.elapsed().as_secs_f64() * 1e3
                        );
                        self.sass = Some(text);
                    }
                    Err(e) => sass_error = Some(e),
                }
            }
        }
        self.notes.clear();
        let dir: PathBuf = self.options.get_dump_dir().map_or_else(|| PathBuf::from("."), PathBuf::from);
        // The PTX and what turns it into a cubin and SASS: one PTX assembled with different
        // `ptxas` flags (`--maxrregcount`) or disassembled with other `nvdisasm` flags is
        // another variant, and gets its own files.
        let mut variant = self.ptx.clone().into_bytes();
        variant.extend(format!("\0O{}\0{}", self.opt_level, self.line_info).bytes());
        for flag in self.options.get_ptxas_options().unwrap_or(&[]) {
            variant.extend(format!("\0ptxas {flag}").bytes());
        }
        if self.options.keeps(Artifact::Sass) {
            for flag in self.options.effective_nvdisasm() {
                variant.extend(format!("\0nvdisasm {flag}").bytes());
            }
        }
        let hash = artifacts::short_hash(&variant);
        let mut made_dir = false;
        for artifact in Artifact::ALL.into_iter().filter(|a| self.options.keeps(*a)) {
            let bytes: Result<Vec<u8>, String> = match artifact {
                Artifact::Ir => Ok(self.ir_text.clone().into_bytes()),
                Artifact::Mlir => Ok(self.mlir.clone().into_bytes()),
                Artifact::Llvm => self.llvm.clone().map(String::into_bytes).ok_or_else(|| "not captured".to_string()),
                Artifact::Ptx => Ok(self.ptx.clone().into_bytes()),
                Artifact::Cubin => self.cubin.clone().ok_or_else(|| self.no_cubin_reason()),
                Artifact::Sass => match (&self.sass, &sass_error) {
                    (Some(text), _) => Ok(text.clone().into_bytes()),
                    (None, Some(e)) => Err(e.clone()),
                    (None, None) => Err(self.no_cubin_reason()),
                },
            };
            let note = match bytes {
                Err(why) => ArtifactNote { artifact, path: None, unavailable: Some(why) },
                Ok(bytes) => {
                    if !made_dir {
                        made_dir = true;
                        if let Err(e) = std::fs::create_dir_all(&dir) {
                            crate::cute_log!(Level::Warn, "cannot create {}: {e}", dir.display());
                        }
                    }
                    let path = artifacts::artifact_path(&dir, &self.name, self.chip, &hash, artifact);
                    match std::fs::write(&path, bytes) {
                        Ok(()) => {
                            crate::cute_log!(Level::Info, "kernel `{}`: wrote {}", self.name, path.display());
                            ArtifactNote { artifact, path: Some(path), unavailable: None }
                        }
                        Err(e) => ArtifactNote {
                            artifact,
                            path: None,
                            unavailable: Some(format!("writing {}: {e}", path.display())),
                        },
                    }
                }
            };
            if let Some(why) = &note.unavailable {
                eprintln!("cute-rs: warning: kernel `{}`: the {} artifact is unavailable: {why}", self.name, artifact);
                crate::cute_log!(Level::Warn, "kernel `{}`: the {artifact} artifact is unavailable: {why}", self.name);
            }
            self.notes.push(note);
        }
    }
}

/// A human-readable listing of the IR.
pub fn dump(ir: &KernelIr) -> String {
    let mut out = String::new();
    let params: Vec<String> = ir
        .args
        .iter()
        .map(|p| match p {
            Param::Buffer(b) => {
                let d = ir.buffer(*b);
                let len = d.len.map_or("?".to_string(), |n| n.to_string());
                format!("{}: gmem<{len} x {}>", d.name, d.elem)
            }
            Param::Scalar { name, value, .. } => format!("{name} = {value}: {}", ir.dtype(*value)),
            Param::ByValue { buffer, grid_constant } => {
                let d = ir.buffer(*buffer);
                let gc = if *grid_constant { " grid_constant" } else { "" };
                format!("{}: byval<{} bytes, align {}>{gc}", d.name, d.len.unwrap_or(0), d.align)
            }
        })
        .collect();
    let _ = write!(out, "kernel {}({})", ir.name, params.join(", "));
    if let Some([x, y, z]) = ir.block_dim {
        let _ = write!(out, " block=({x},{y},{z})");
    }
    out.push('\n');
    for b in ir.buffers.iter().filter(|b| b.space == MemSpace::Shared) {
        // The dynamic region is sized at launch.
        let len = b.len.map_or("?".to_string(), |n| n.to_string());
        let _ = writeln!(out, "  shared {}: {len} x {} (align {})", b.name, b.elem, b.align);
        if let Some((name, fields)) = b.fields.split_first() {
            let _ = writeln!(out, "    struct {name} {{ {} }}", fields.join(", "));
        }
    }
    dump_region(ir, &ir.body, 1, &mut out);
    out
}

/// A TMA instruction's access mode in a dump: nothing for `.tile` (m7-tma-im2col and
/// m7-tma-sm100-modes).
fn tma_mode_text(mode: &crate::ir::TmaMode) -> String {
    match mode {
        crate::ir::TmaMode::Tile => String::new(),
        crate::ir::TmaMode::Im2col { offsets } => format!(" im2col offsets {}", list(offsets)),
        m => format!(" {}", &m.ptx()[1..]),
    }
}

/// A TMA instruction's `.override::*` operands in a dump.
fn tma_override_text(o: &Option<crate::ir::TmaOverride>) -> String {
    match o {
        None => String::new(),
        Some(o) if o.dims.is_empty() => format!(" override address {}", o.global_address),
        Some(o) => format!(
            " override address {} dims {} strides {} upper {}",
            o.global_address,
            list(&o.dims),
            list(&o.lower_strides),
            o.upper_strides.map(|v| v.to_string()).unwrap_or_default()
        ),
    }
}

fn list(vs: &[crate::ir::ValueId]) -> String {
    vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
}

fn dump_region(ir: &KernelIr, r: &Region, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    for Inst { op, scope, .. } in &r.insts {
        let results = op.results();
        let lhs = if results.is_empty() { String::new() } else { format!("{} = ", list(&results)) };
        let buf = |b: &crate::ir::BufferId| ir.buffer(*b).name.clone();
        let asm_inputs = |inputs: &[crate::ir::AsmInput]| {
            let inputs: Vec<String> = inputs
                .iter()
                .map(|i| match i {
                    crate::ir::AsmInput::Value(v) => v.to_string(),
                    crate::ir::AsmInput::Address { buffer, index, .. } => format!("&{}[{index}]", buf(buffer)),
                })
                .collect();
            inputs.join(", ")
        };
        let body = match op {
            Op::Const { value, .. } => format!("const {value:?}"),
            Op::Unary { op, x, .. } => format!("{op:?} {x}"),
            Op::Binary { op, lhs, rhs, .. } => format!("{op:?} {lhs}, {rhs}"),
            Op::Fma { a, b, c, .. } => format!("fma {a}, {b}, {c}"),
            Op::FloatInst { func, args, mode, ftz, nan, xorsign_abs, .. } => {
                let name = crate::ir::float_inst_name(*func, *mode, *ftz, *nan, *xorsign_abs, ir.dtype(args[0]));
                format!("{name} {}", list(args))
            }
            Op::IntInst { func, args, .. } => {
                let name = crate::ir::int_inst_name(*func, ir.dtype(args[0]));
                match func {
                    crate::ir::IntFn::Lop3 { lut } => format!("{name} {}, {lut:#04x}", list(args)),
                    _ => format!("{name} {}", list(args)),
                }
            }
            Op::Cmp { pred, lhs, rhs, .. } => format!("cmp {pred:?} {lhs}, {rhs}"),
            Op::Select { cond, on_true, on_false, .. } => format!("select {cond}, {on_true}, {on_false}"),
            Op::Cast { x, .. } => format!("cast {x}"),
            Op::Bitcast { x, .. } => format!("bitcast {x}"),
            Op::Special { reg, .. } => format!("{reg:?}"),
            Op::Load { buffer, index, attrs, pred, pass_thru, .. } => {
                let policy = attrs.cache_policy.map(|p| format!(", policy {p}")).unwrap_or_default();
                let p = pred.map(|p| format!(" if {p}")).unwrap_or_default();
                let pt = if pass_thru.is_empty() { String::new() } else { format!(" else {}", list(pass_thru)) };
                let cta = attrs.peer.map(|p| format!(" in CTA {p}")).unwrap_or_default();
                format!("load{} {}[{index}]{cta}{policy}{p}{pt}", attrs.qualifiers(), buf(buffer))
            }
            Op::Store { values, buffer, index, attrs, pred } => {
                let policy = attrs.cache_policy.map(|p| format!(", policy {p}")).unwrap_or_default();
                let p = pred.map(|p| format!(" if {p}")).unwrap_or_default();
                let cta = attrs.peer.map(|p| format!(" in CTA {p}")).unwrap_or_default();
                format!("store{} {} -> {}[{index}]{cta}{policy}{p}", attrs.qualifiers(), list(values), buf(buffer))
            }
            Op::CpAsync { src, src_index, dst, dst_index, elems, pred, .. } => {
                let p = pred.map(|p| format!(" if {p}")).unwrap_or_default();
                format!("cp.async {}[{src_index}] -> {}[{dst_index}] x{elems}{p}", buf(src), buf(dst))
            }
            Op::CpAsyncCommitGroup => "cp.async.commit_group".into(),
            Op::CpAsyncWaitGroup { pending } => format!("cp.async.wait_group {pending}"),
            Op::StMatrix { values, buffer, index, num, trans, form } => {
                format!("{} {}[{index}], {}", form.ptx(*num, *trans), buf(buffer), list(values))
            }
            Op::SetMaxNReg { increase, count } => {
                format!("setmaxnreg.{}.sync.aligned.u32 {count}", if *increase { "inc" } else { "dec" })
            }
            Op::LdMatrix { buffer, index, num, trans, form: LdMatrixForm::M8n8B16, .. } => {
                format!("ldmatrix.x{num}{} {}[{index}]", if *trans { ".trans" } else { "" }, buf(buffer))
            }
            Op::LdMatrix { buffer, index, num, trans, form, .. } => {
                format!(
                    "ldmatrix.{}.x{num}{}.{} {}[{index}]",
                    form.shape(),
                    if *trans { ".trans" } else { "" },
                    form.format(),
                    buf(buffer)
                )
            }
            Op::Mma { kind, a, b, c, e: None, .. } => {
                format!("mma {kind:?} a=[{}] b=[{}] c=[{}]", list(a), list(b), list(c))
            }
            Op::Mma { kind, a, b, c, e: Some(e), .. } => {
                format!("mma {kind:?} a=[{}] b=[{}] c=[{}] e=%{}", list(a), list(b), list(c), e.0)
            }
            Op::MmaSpBlockScaled { kind, a, b, c, e, sfa, sfb, .. } => format!(
                "{} a=[{}] b=[{}] c=[{}] e=%{} sfa=%{} sfb=%{}",
                kind.ptx(),
                list(a),
                list(b),
                list(c),
                e.0,
                sfa.0,
                sfb.0
            ),
            Op::MmaBlockScaled { kind, a, b, c, sfa, sfb, .. } => {
                format!("{} a=[{}] b=[{}] c=[{}] sfa=%{} sfb=%{}", kind.ptx(), list(a), list(b), list(c), sfa.0, sfb.0)
            }
            Op::Barrier => "barrier".into(),
            Op::NamedBarrier { id, threads, .. } => {
                let n = threads.map(|n| format!(", {n}")).unwrap_or_default();
                format!("{} {id}{n}", op.name())
            }
            Op::WarpBarrier { mask } => format!("warp barrier {mask}"),
            Op::Shuffle { kind, value, offset, mask, clamp, .. } => {
                format!("shuffle {kind:?} {value}, {offset}, {clamp}, {mask}")
            }
            Op::Vote { kind, pred, mask, .. } => format!("vote {kind:?} {pred}, {mask}"),
            Op::Match { kind, value, mask, .. } => format!("match {kind:?} {value}, {mask}"),
            Op::ElectOne { mask, .. } => format!("elect {mask}"),
            Op::ActiveMask { .. } => "activemask".into(),
            Op::WarpUniform { value, .. } => format!("make_warp_uniform {value}"),
            Op::Atomic { kind, ty, buffer, index, value, compare, sem, scope, peer, .. } => {
                let c = compare.map(|c| format!(" if {c}")).unwrap_or_default();
                let p = peer.map(|p| format!(" in CTA {p}")).unwrap_or_default();
                format!(
                    "atomic.{}.{}.{} {}[{index}]{p}, {value}{c}",
                    sem.name(),
                    scope.ptx(),
                    ty.op_suffix(*kind),
                    buf(buffer)
                )
            }
            Op::Red { kind, ty, buffer, index, value, sem, scope, peer } => {
                let p = peer.map(|p| format!(" in CTA {p}")).unwrap_or_default();
                format!(
                    "red.{}.{}.{} {}[{index}]{p}, {value}",
                    sem.name(),
                    scope.ptx(),
                    ty.op_suffix(*kind),
                    buf(buffer)
                )
            }
            Op::MbarrierInit { buffer, index, count } => format!("mbarrier.init {}[{index}], {count}", buf(buffer)),
            Op::MbarrierArrive { buffer, index, count, expect_tx, peer, relaxed, scope } => {
                let mut text = format!("mbarrier.arrive {}[{index}]", buf(buffer));
                if let Some(c) = count {
                    let _ = write!(text, " x{c}");
                }
                if let Some(t) = expect_tx {
                    let _ = write!(text, " expect {t}");
                }
                if let Some(p) = peer {
                    let _ = write!(text, " on cta {p}");
                }
                let _ = write!(text, " {}{scope:?}", if *relaxed { "relaxed " } else { "" });
                text
            }
            Op::MbarrierExpectTx { buffer, index, bytes, peer, scope } => {
                let p = peer.map(|p| format!(" on cta {p}")).unwrap_or_default();
                format!("mbarrier.expect_tx {}[{index}], {bytes}{p} {scope:?}", buf(buffer))
            }
            Op::MbarrierWait { buffer, index, parity, state, time_limit, .. } => {
                let state = if *state { " state" } else { "" };
                let limit = time_limit.map(|t| format!(", limit {t}")).unwrap_or_default();
                format!("{}{state} {}[{index}], {parity}{limit}", op.name(), buf(buffer))
            }
            Op::BarrierRed { kind, pred, id, threads, .. } => {
                let n = threads.map(|n| format!(", {n}")).unwrap_or_default();
                format!("barrier.cta.red.{} {id}{n}, {pred}", kind.ptx())
            }
            Op::MbarrierExt { kind, buffer, index, value, .. } => {
                let v = value.map(|v| format!(", {v}")).unwrap_or_default();
                format!("{} {}[{index}]{v}", kind.ptx(), buf(buffer))
            }
            Op::PrimMarker { marker } => format!("{marker:?}"),
            Op::Exit => "exit".into(),
            Op::StBulk { buffer, index, elem, size } => format!("st.bulk {}[{index}] ({elem}), {size}", buf(buffer)),
            Op::Tcgen05Shift { taddr, two_cta } => format!("tcgen05.shift.cta_group::{} {taddr}", 1 + *two_cta as u8),
            Op::FenceMbarrierInit => "fence.mbarrier_init".into(),
            Op::GridDepControl { .. } => op.name().into(),
            Op::ClusterArrive { aligned, .. } | Op::ClusterWait { aligned } => {
                format!("{}{}", op.name(), if *aligned { ".aligned" } else { "" })
            }
            Op::CpAsyncBulk {
                src,
                src_index,
                dst,
                dst_index,
                bytes,
                completion,
                cache_policy,
                mcast_mask,
                peer,
                byte_mask,
                ..
            } => {
                let done = match completion {
                    crate::ir::BulkCompletion::Mbarrier { buffer, index } => {
                        format!("mbarrier {}[{index}]", buf(buffer))
                    }
                    crate::ir::BulkCompletion::BulkGroup => "bulk_group".to_string(),
                };
                let mut extra = String::new();
                if let Some(p) = peer {
                    let _ = write!(extra, " on cta {p}");
                }
                if let Some(m) = mcast_mask {
                    let _ = write!(extra, " multicast {m}");
                }
                if let Some(p) = cache_policy {
                    let _ = write!(extra, " cache_hint {p}");
                }
                if let Some(m) = byte_mask {
                    let _ = write!(extra, " cp_mask {m}");
                }
                format!("cp.async.bulk {}[{src_index}] -> {}[{dst_index}] x{bytes} {done}{extra}", buf(src), buf(dst))
            }
            Op::TmaLoad {
                dst,
                dst_index,
                desc,
                coords,
                mbar,
                mbar_index,
                cache_policy,
                box_bytes,
                mcast_mask,
                mode,
                overrides,
                two_cta,
                ..
            } => {
                let coords: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
                let hint = cache_policy.map(|p| format!(" cache_hint {p}")).unwrap_or_default();
                let mc = mcast_mask.map(|m| format!(" multicast {m}")).unwrap_or_default();
                let pair = if *two_cta { " cta_group::2" } else { "" };
                format!(
                    "cp.async.bulk.tensor.{}d [{desc}, {{{}}}] -> {}[{dst_index}] x{box_bytes} mbarrier {}[{mbar_index}]{hint}{mc}{pair}{}{}",
                    coords.len(),
                    coords.join(", "),
                    buf(dst),
                    buf(mbar),
                    tma_mode_text(mode),
                    tma_override_text(overrides)
                )
            }
            Op::TmaStore { src, src_index, desc, coords, reduce, cache_policy, box_bytes, mode, overrides, .. } => {
                let coords: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
                let hint = cache_policy.map(|p| format!(" cache_hint {p}")).unwrap_or_default();
                let op = reduce.map(|k| format!(" reduce.{}", k.ptx_name())).unwrap_or_default();
                format!(
                    "cp.async.bulk.tensor.{}d {}[{src_index}] -> [{desc}, {{{}}}] x{box_bytes}{op} bulk_group{hint}{}{}",
                    coords.len(),
                    buf(src),
                    coords.join(", "),
                    tma_mode_text(mode),
                    tma_override_text(overrides)
                )
            }
            Op::TmaPrefetch { desc, coords, mode } => {
                let coords: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
                format!(
                    "cp.async.bulk.prefetch.tensor.{}d [{desc}, {{{}}}]{}",
                    coords.len(),
                    coords.join(", "),
                    tma_mode_text(mode)
                )
            }
            Op::PrefetchTensormap { desc } => format!("prefetch.tensormap [{desc}]"),
            Op::TensormapReplace { addr, space, field, value } => {
                format!("tensormap.replace.{field} {space} [{addr}] <- {value}")
            }
            Op::FenceTensormapRelease => op.name().into(),
            Op::FenceTensormapAcquire { addr } => format!("{} [{addr}]", op.name()),
            Op::TensormapCpFenceproxy { dst, src } => format!("tensormap.cp_fenceproxy [{dst}] <- [{src}]"),
            Op::BulkPrefetch { buffer, index, bytes, .. } => {
                format!("cp.async.bulk.prefetch {}[{index}] x{bytes}", buf(buffer))
            }
            Op::ClcTryCancel { buffer, index, mbar, mbar_index, multicast, .. } => format!(
                "clusterlaunchcontrol.try_cancel{} {}[{index}] mbarrier {}[{mbar_index}]",
                if *multicast { ".multicast" } else { "" },
                buf(buffer),
                buf(mbar)
            ),
            Op::ClcQueryCancel { buffer, index, results, .. } => {
                format!("{} = clusterlaunchcontrol.query_cancel {}[{index}]", list(results), buf(buffer))
            }
            Op::StAsync { buffer, index, values, mbar, mbar_index, peer, .. } => {
                format!(
                    "st.async {}[{index}] <- {} mbarrier {}[{mbar_index}] on cta {peer}",
                    buf(buffer),
                    list(values),
                    buf(mbar)
                )
            }
            Op::CpAsyncBulkCommitGroup => "cp.async.bulk.commit_group".into(),
            Op::CpAsyncBulkWaitGroup { pending, .. } => format!("{} {pending}", op.name()),
            Op::FenceProxy { kind } => kind.ptx().into(),
            Op::Fence { scope, sc: false } => format!("fence {scope:?}"),
            Op::Fence { scope, sc: true } => format!("fence.sc {scope:?}"),
            Op::Printf { format, args } => format!("printf {format:?} {}", list(args)),
            Op::Assert { cond, message } => format!("assert {cond} {message:?}"),
            Op::InlinePtx { template, inputs, constraints, .. } => {
                format!("asm {template:?} {constraints:?} {}", asm_inputs(inputs))
            }
            Op::Call { callee, args, body, .. } => {
                let symbol = &ir.extern_decl(*callee).symbol;
                match body {
                    None => format!("call @{symbol}({})", asm_inputs(args)),
                    Some(r) => {
                        let _ = writeln!(out, "{pad}{lhs}call @{symbol}({}) simulated as {{", asm_inputs(args));
                        dump_region(ir, r, depth + 1, out);
                        let _ = writeln!(out, "{pad}}} yield {}", list(&r.yields));
                        continue;
                    }
                }
            }
            Op::Multimem { op, buffer, index, values, .. } => {
                let stored = if values.is_empty() { String::new() } else { format!(", {}", list(values)) };
                format!("{} &{}[{index}]{stored}", op.ptx(), buf(buffer))
            }
            Op::If { cond, then_region, else_region, .. } => {
                let _ = writeln!(out, "{pad}{lhs}if {cond} {{");
                dump_region(ir, then_region, depth + 1, out);
                let _ = writeln!(out, "{pad}}} else {{");
                dump_region(ir, else_region, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
            Op::For { iv, lower, upper, step, init, args, body, opts, .. } => {
                let iters: Vec<String> = args.iter().zip(init).map(|(a, i)| format!("{a} = {i}")).collect();
                let mut hints = String::new();
                if let Some(count) = opts.unroll {
                    let _ = write!(hints, " unroll({count})");
                }
                for (on, name) in [
                    (opts.unroll_full, "unroll_full"),
                    (opts.vectorize, "vectorize"),
                    (opts.at_least_once, "at_least_once"),
                ] {
                    if on {
                        let _ = write!(hints, " {name}");
                    }
                }
                let _ = writeln!(
                    out,
                    "{pad}{lhs}for {iv} in {lower}..{upper} step {step} iter({}){hints} {{",
                    iters.join(", ")
                );
                dump_region(ir, body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
            Op::While { init, cond_args, cond, body_args, body, .. } => {
                let iters: Vec<String> = cond_args.iter().zip(init).map(|(a, i)| format!("{a} = {i}")).collect();
                let _ = writeln!(out, "{pad}{lhs}while ({}) {{", iters.join(", "));
                dump_region(ir, cond, depth + 1, out);
                let _ = writeln!(out, "{pad}}} do ({}) {{", list(body_args));
                dump_region(ir, body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
                continue;
            }
            Op::Redux { kind, value, mask, abs, nan, .. } => {
                format!("{} {value}, {mask}", crate::ir::redux_inst_name(*kind, *abs, *nan, ir.dtype(*value)))
            }
            Op::CarryInst { func, args, .. } => {
                format!("{} {}", crate::ir::carry_inst_name(*func, ir.dtype(args[0])), list(args))
            }
            Op::MlirOp { text, operands, .. } => format!("mlir {text:?} {}", asm_inputs(operands)),
            Op::CpAsyncMbarrierArrive { buffer, index, .. } => format!("{} {}[{index}]", op.name(), buf(buffer)),
            Op::Prefetch { buffer, index, level, evict, uniform, pred, .. } => {
                let p = pred.map(|p| format!(" if {p}")).unwrap_or_default();
                let what = match (uniform, evict) {
                    (true, _) => "prefetchu.L1".to_string(),
                    (false, Some(e)) => format!("prefetch.L2::{}", e.name()),
                    (false, None) => format!("prefetch.{}", level.ptx()),
                };
                format!("{what} {}[{index}]{p}", buf(buffer))
            }
            Op::Pack { parts, .. } => format!("pack {}", list(parts)),
            Op::Unpack { x, .. } => format!("unpack {x}"),
            Op::HalfInst { func, ty, args, mods, predicate, .. } => {
                let guard = predicate.map(|p| format!("@{p} ")).unwrap_or_default();
                format!("{guard}{} {}", crate::ir::half_inst_name(*func, *ty, *mods), list(args))
            }
            Op::Cvt { func, args, .. } => format!("{} {}", func.ptx(), list(args)),
            Op::Assume { x, divby, pow2, .. } => format!("assume {x} divby={divby} pow2={pow2}"),
            Op::PtrToInt { buffer, index, elem, .. } => format!("ptrtoint &{}[{index}] ({elem})", buf(buffer)),
            Op::CvtAddr { x, from, to, .. } => format!("cvta {x} {from} -> {to}"),
            Op::Wgmma { kind, c, a, desc_b, scale_d, .. } => {
                let t = |b: bool| b as u8;
                let a = if kind.a_regs { format!("[{}]", list(a)) } else { list(a) };
                format!(
                    "{} c=[{}] a={a} b={desc_b} scale_d={scale_d} trans=({}, {})",
                    kind.ptx(),
                    list(c),
                    t(kind.trans_a),
                    t(kind.trans_b)
                )
            }
            Op::WgmmaFence { inputs, .. } => format!("wgmma.fence [{}]", list(inputs)),
            Op::WgmmaCommitGroup => "wgmma.commit_group".into(),
            Op::WgmmaWait { pending, inputs, .. } => format!("wgmma.wait_group {pending} [{}]", list(inputs)),
            Op::Tcgen05 { inst, two_cta, exclusive, buffer, args } => {
                let ex = if *exclusive { ".exclusive" } else { "" };
                let dst = buffer.map(|b| format!(" {}", buf(&b))).unwrap_or_default();
                format!("{}{ex}.cta_group::{}{dst} {}", inst.name(), if *two_cta { 2 } else { 1 }, list(args))
            }
            Op::Tcgen05Ld { taddr, shape, num, pack, red, sp, .. } => {
                let red = red.map(|r| format!(".red{}{}", r.op.ptx(), if r.nan { ".NaN" } else { "" }));
                format!(
                    "tcgen05.ld.{}.x{num}{}{}{} [{taddr}]",
                    shape.ptx(),
                    if *pack { ".pack::16b" } else { "" },
                    red.unwrap_or_default(),
                    if sp.is_some() { ".spcompress" } else { "" }
                )
            }
            Op::Tcgen05St { values, taddr, shape, num, unpack } => format!(
                "tcgen05.st.{}.x{num}{} [{taddr}], {}",
                shape.ptx(),
                if *unpack { ".unpack::16b" } else { "" },
                list(values)
            ),
            Op::Tcgen05Mma {
                kind,
                taddr,
                desc_a,
                desc_b,
                enable_input_d,
                elect,
                idesc,
                disable_lane,
                sf,
                meta,
                zero_col_mask,
            } => {
                format!(
                    "{} [{taddr}], {}, {desc_b}, {}{}{}{}, {enable_input_d}{}{} (elect {elect})",
                    kind.ptx(),
                    if kind.a_tmem { format!("[{desc_a}]") } else { desc_a.to_string() },
                    match meta {
                        Some(e) => format!("[{e}], "),
                        None => String::new(),
                    },
                    match idesc {
                        Some(v) => format!("{:#010x} | {v}", kind.idesc()),
                        None => format!("{:#010x}", kind.idesc()),
                    },
                    match sf {
                        Some((a, b)) => format!(", [{a}], [{b}]"),
                        None => String::new(),
                    },
                    if disable_lane.is_empty() { String::new() } else { format!(", {{{}}}", list(disable_lane)) },
                    kind.prims.scale_input_d.map(|s| format!(", {s}")).unwrap_or_default(),
                    zero_col_mask.map(|z| format!(", {z}")).unwrap_or_default()
                )
            }
            Op::Tcgen05Cp { shape, two_cta, taddr, desc, elect } => format!(
                "tcgen05.cp.cta_group::{}.{} [{taddr}], {desc} (elect {elect})",
                if *two_cta { 2 } else { 1 },
                shape.ptx()
            ),
            Op::Tcgen05Commit { buffer, index, mask, two_cta } => format!(
                "tcgen05.commit.cta_group::{} {}[{index}]{}",
                if *two_cta { 2 } else { 1 },
                buf(buffer),
                mask.map(|m| format!(", mask {m}")).unwrap_or_default()
            ),
        };
        match scope {
            Some(id) => {
                let _ = writeln!(out, "{pad}{lhs}{body}  // scope {}", ir.scope_path(*id));
            }
            None => {
                let _ = writeln!(out, "{pad}{lhs}{body}");
            }
        }
    }
    if !r.yields.is_empty() {
        let _ = writeln!(out, "{pad}yield {}", list(&r.yields));
    }
}

/// The stage hook that marks the kernel's by-value parameters: `llvm.byval` with the
/// aggregate's type and `nvvm.grid_constant` when asked for (the bare-pointer convention
/// keeps one `llvm.func` argument per parameter).
fn byval_hook(ir: &crate::ir::KernelIr) -> Option<options::StageHook> {
    let marks: Vec<(usize, usize, bool)> = ir
        .args
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            Param::ByValue { buffer, grid_constant } => Some((i, ir.buffer(*buffer).len.unwrap_or(0), *grid_constant)),
            _ => None,
        })
        .collect();
    if marks.is_empty() {
        return None;
    }
    let name = ir.name.clone();
    Some(std::sync::Arc::new(move |m: &mut LoweredModule| {
        for &(i, size, grid_constant) in &marks {
            m.set_argument_attribute(&name, i, "llvm.byval", &format!("!llvm.array<{size} x i8>"))?;
            if grid_constant {
                m.set_argument_attribute(&name, i, "nvvm.grid_constant", "unit")?;
            }
        }
        Ok(())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiled(opt_level: u8, line_info: bool) -> CompiledKernel {
        CompiledKernel {
            name: "k".to_string(),
            arch: Arch::Sm80,
            chip: "sm_80",
            note: None,
            ptx_reason: None,
            ptx_version: 70,
            opt_level,
            line_info,
            options: CompileOptions::new(),
            ptx: String::new(),
            ir_text: String::new(),
            mlir: String::new(),
            llvm: None,
            cubin: None,
            cubin_target: None,
            ptxas_log: None,
            ptxas_command: None,
            sass: None,
            notes: Vec::new(),
            diagnostics: Vec::new(),
            resource_usage: None,
            signature: crate::Kernel::build("k", || {}).signature(),
        }
    }

    #[test]
    fn jit_options_follow_the_compile_options() {
        let level = |o: u8| compiled(o, false).jit_options();
        assert_eq!(level(0), [JitOption::OptimizationLevel(0)]);
        assert_eq!(level(1), [JitOption::OptimizationLevel(1)]);
        assert_eq!(level(2), [JitOption::OptimizationLevel(2)]);
        assert_eq!(level(3), [JitOption::OptimizationLevel(4)]);
        assert_eq!(
            compiled(3, true).jit_options(),
            [JitOption::OptimizationLevel(4), JitOption::GenerateLineInfo(true)]
        );
        let codes: Vec<(u32, u32)> = compiled(0, true).jit_options().iter().map(|o| (o.code(), o.value())).collect();
        assert_eq!(codes, [(7, 0), (13, 1)]);
    }
}

#[cfg(test)]
mod unknown_intrinsic_tests {
    #[test]
    fn extern_intrinsic_declarations_are_reported() {
        let ptx = ".version 8.8\n.extern .func llvm.nvvm.tcgen05.mma.sp.shared.mxf4nvf4\n(\n\t.param .b64 p0\n);\n";
        let e = super::unknown_intrinsic(ptx).unwrap();
        assert!(e.contains("`llvm.nvvm.tcgen05.mma.sp.shared.mxf4nvf4`"), "{e}");
        assert!(super::unknown_intrinsic(".extern .func (.param .b32 r) vprintf\n").is_none());
        assert!(super::unknown_intrinsic(".visible .entry k()\n{\n}\n").is_none());
    }
}

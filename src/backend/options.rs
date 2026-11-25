//! Compile options: CuTe DSL's `CompileOptions`.
//!
//! CuTe DSL takes options from three sources, and so does cute-rs:
//!
//! * typed objects, `cute.compile[cute.OptLevel(2), cute.GenerateLineInfo(True)]`, which are
//!   the builder methods here ([`CompileOptions::opt_level`],
//!   [`CompileOptions::generate_line_info`], [`CompileOptions::gpu_arch`]);
//! * an option string, `cute.compile(fn, options="--opt-level 2 --gpu-arch sm_90a")`, which
//!   is [`CompileOptions::parse`]. It accepts both of CuTe's syntaxes: argparse-style flags
//!   (`--opt-level 2`, `--opt-level=2`, unambiguous prefixes such as `--opt 2`) and the compact
//!   tokens of `CUTE_DSL_COMPILER_OPT` (`opt-level=2`, `preserve-line-info`, `cubin-chip=sm_80`,
//!   `warnings{nvvm}`, `iket`), with shell-style quoting;
//! * environment variables, named `CUTE_RS_` plus CuTe's suffix so that a machine with CuTe
//!   DSL installed is unaffected: `CUTE_RS_COMPILER_OPT` (an option string), `CUTE_RS_ARCH`
//!   and `CUTE_RS_LINEINFO`. [`CompileOptions::with_env`] fills in what the explicit options
//!   leave unset, so explicit options override the environment. That is a deliberate
//!   difference from CuTe DSL, which applies its environment on top of the explicit options;
//!   among the variables, CuTe's order is kept (see [`CompileOptions::from_env_with`]).
//!
//! Every CuTe DSL 4.8.1 spelling is recognised. Options whose feature has not been built yet
//! are rejected with a message naming the part of the roadmap that adds them, rather than
//! ignored; options that select one of NVIDIA's other Python front ends are accepted and do
//! nothing (cute-rs has one front end and one pipeline). The README's "Compile options"
//! section has the full table.

use std::fmt;
use std::sync::Arc;

use super::Arch;
use std::path::{Path, PathBuf};

use super::artifacts::{Artifact, Keep};
use super::stage::LoweredModule;

/// The diagnostic settings of a compile.
pub(crate) struct DiagnosticSettings<'a> {
    pub warnings: &'a Option<Vec<String>>,
    pub remarks: &'a Option<Vec<String>>,
    pub warning_filter: Option<&'a str>,
    pub remark_filter: Option<&'a str>,
    pub remark_output: Option<&'a Path>,
    pub as_errors: bool,
    pub ignore: bool,
}

/// The optimization level used when none is given (CuTe DSL's default).
pub const DEFAULT_OPT_LEVEL: u32 = 3;

/// The option string read from the environment (CuTe DSL's `CUTE_DSL_COMPILER_OPT`).
pub const ENV_COMPILER_OPT: &str = "CUTE_RS_COMPILER_OPT";
/// The target architecture read from the environment (`CUTE_DSL_ARCH`), e.g. `sm_80`, for
/// when `CUTE_RS_COMPILER_OPT` names none.
pub const ENV_ARCH: &str = "CUTE_RS_ARCH";
/// Switches line information on (`CUTE_DSL_LINEINFO`) when `1`, `true`, `yes` or `on`; `0`,
/// `false`, `no` or `off` leave it as the other options set it.
pub const ENV_LINEINFO: &str = "CUTE_RS_LINEINFO";

/// A function that edits the kernel between the two lowering stages.
pub(crate) type StageHook = Arc<dyn Fn(&mut LoweredModule) -> Result<(), String> + Send + Sync>;

/// An option that could not be parsed or is not supported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionError {
    message: String,
}

impl OptionError {
    pub(crate) fn new(message: impl Into<String>) -> OptionError {
        OptionError { message: message.into() }
    }

    /// The message, which names the offending option.
    pub fn message(&self) -> &str {
        &self.message
    }

    fn within(self, source: &str) -> OptionError {
        OptionError { message: format!("{source}: {}", self.message) }
    }
}

impl fmt::Display for OptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OptionError {}

/// How a kernel is compiled: CuTe DSL's `CompileOptions`.
///
/// Every setting is optional; an unset one takes its default when the kernel is compiled
/// (optimization level 3, no line information). [`Kernel::compile_with`](crate::Kernel::compile_with)
/// uses exactly the options it is given; [`with_env`](Self::with_env) adds the environment,
/// as `Kernel::run_cuda` (feature `cuda`) does.
///
/// ```
/// use cute::{Arch, CompileOptions};
///
/// let typed = CompileOptions::new().gpu_arch(Arch::Sm80).opt_level(2).generate_line_info(true);
/// let parsed = CompileOptions::parse("--gpu-arch sm_80 --opt-level 2 --generate-line-info").unwrap();
/// assert_eq!(typed.get_opt_level(), parsed.get_opt_level());
/// assert_eq!(parsed.get_gpu_arch(), Some(Arch::Sm80));
///
/// // Explicit options override the environment.
/// let env = |name: &str| (name == "CUTE_RS_COMPILER_OPT").then(|| "opt-level=0 preserve-line-info".to_string());
/// let merged = CompileOptions::new().opt_level(1).with_env_from(env).unwrap();
/// assert_eq!(merged.get_opt_level(), Some(1));
/// assert_eq!(merged.get_generate_line_info(), Some(true));
/// ```
#[derive(Clone, Default)]
pub struct CompileOptions {
    gpu_arch: Option<Arch>,
    opt_level: Option<u8>,
    line_info: Option<bool>,
    hooks: Vec<StageHook>,
    mma_inline_ptx: Option<bool>,
    ptxas_flags: Option<Vec<String>>,
    use_ptxas: Option<bool>,
    keep: Keep,
    dump_dir: Option<PathBuf>,
    nvdisasm_flags: Option<Vec<String>>,
    print_ir: Option<bool>,
    enable_assertions: Option<bool>,
    fp_contract: Option<bool>,
    enable_tvm_ffi: Option<bool>,
    warnings: Option<Vec<String>>,
    remarks: Option<Vec<String>>,
    warning_filter: Option<String>,
    remark_filter: Option<String>,
    remark_output: Option<PathBuf>,
    warnings_as_errors: Option<bool>,
    warnings_ignore: Option<bool>,
    no_cache: Option<bool>,
    /// LLVM bitcode linked into every module compiled with these options (CuTe DSL's
    /// `cute.LinkLibraries`).
    link_libraries: Vec<PathBuf>,
}

impl fmt::Debug for CompileOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompileOptions")
            .field("gpu_arch", &self.gpu_arch)
            .field("opt_level", &self.opt_level)
            .field("generate_line_info", &self.line_info)
            .field("stage_hooks", &self.hooks.len())
            .field("mma_inline_ptx", &self.mma_inline_ptx)
            .field("ptxas_options", &self.ptxas_flags)
            .field("use_ptxas", &self.use_ptxas)
            .field("keep", &self.keep)
            .field("dump_dir", &self.dump_dir)
            .field("nvdisasm_options", &self.nvdisasm_flags)
            .field("print_ir", &self.print_ir)
            .field("enable_assertions", &self.enable_assertions)
            .field("fp_contract", &self.fp_contract)
            .field("enable_tvm_ffi", &self.enable_tvm_ffi)
            .field("warnings", &self.warnings)
            .field("remarks", &self.remarks)
            .field("warning_filter", &self.warning_filter)
            .field("remark_filter", &self.remark_filter)
            .field("remark_output", &self.remark_output)
            .field("warnings_as_errors", &self.warnings_as_errors)
            .field("warnings_ignore", &self.warnings_ignore)
            .field("no_cache", &self.no_cache)
            .field("link_libraries", &self.link_libraries)
            .finish()
    }
}

impl CompileOptions {
    /// No options set: everything takes its default.
    pub fn new() -> CompileOptions {
        CompileOptions::default()
    }

    /// The target architecture (CuTe DSL's `cute.GPUArch`, `--gpu-arch sm_XX`).
    pub fn gpu_arch(mut self, arch: Arch) -> Self {
        self.gpu_arch = Some(arch);
        self
    }

    /// The optimization level, 0 to 3 (CuTe DSL's `cute.OptLevel(n)`, `--opt-level n`). It sets
    /// both LLVM's level (`nvvm-attach-target O=`) and the driver JIT's
    /// (`CU_JIT_OPTIMIZATION_LEVEL`: 0, 1 and 2 map to themselves and 3 to the JIT's default,
    /// 4, which corresponds to `ptxas -O3`). Panics for any other level, as CuTe DSL raises
    /// for `OptLevel(4)`.
    #[track_caller]
    pub fn opt_level(mut self, level: u32) -> Self {
        match u8::try_from(level).ok().filter(|&l| l <= 3) {
            Some(l) => self.opt_level = Some(l),
            None => panic!("cute: {}", opt_level_range(&level.to_string())),
        }
        self
    }

    /// Emit line information, so that PTX and SASS map back to the Rust source for Nsight
    /// Compute, cuda-gdb and compute-sanitizer (CuTe DSL's `cute.GenerateLineInfo(True)`,
    /// `--generate-line-info`, `CUTE_DSL_LINEINFO=1`). The PTX gets `.file` and `.loc`
    /// directives, the driver JIT keeps them (`CU_JIT_GENERATE_LINE_INFO`), and the
    /// instructions are the same as without it.
    pub fn generate_line_info(mut self, on: bool) -> Self {
        self.line_info = Some(on);
        self
    }

    /// Run `hook` on the kernel after the first lowering stage, when it is LLVM dialect inside
    /// the `gpu.module`, before it is serialised to PTX. The hook edits the text, usually
    /// through the helpers of [`LoweredModule`]; an error fails the compilation. Hooks run in
    /// the order they were added.
    pub fn stage_hook(
        mut self,
        hook: impl Fn(&mut LoweredModule) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Emit the warp MMAs that need a recent NVVM dialect as inline PTX (a cute-rs option, off
    /// by default): FP8 `mma.sync` ([`MmaFP8Op`](crate::MmaFP8Op)), whose `nvvm.mma.sync` form
    /// needs the `mlir-opt` of LLVM 23, and `mma.sp` ([`MmaF16BF16SparseOp`](crate::MmaF16BF16SparseOp)),
    /// whose `nvvm.mma.sp.sync` needs LLVM 22, and with them the `.m16n16` and `.m8n16`
    /// forms of `ldmatrix` ([`CopyOp::LdMatrix16x16x8b`](crate::CopyOp::LdMatrix16x16x8b) and
    /// the unpacking atoms), whose `nvvm.ldmatrix` attributes need LLVM 22. Without it,
    /// compiling such a kernel with an older `mlir-opt` fails with
    /// [`CompileError::Unsupported`](crate::CompileError). Neither MMA form is marked
    /// convergent (LLVM's `llvm.nvvm.mma.*` intrinsics are only
    /// `memory(none)`); the inline assembly has side effects, so LLVM does not delete,
    /// speculate or sink it, but it may still duplicate it into both sides of a branch. These
    /// are warp-collective `.aligned` instructions, which every lane must reach together, so
    /// keep them out of code whose control flow LLVM could make divergent. The same holds for
    /// the inline `ldmatrix` (`nvvm.ldmatrix` is convergent).
    pub fn mma_inline_ptx(mut self, on: bool) -> Self {
        self.mma_inline_ptx = Some(on);
        self
    }

    /// Whether [`mma_inline_ptx`](Self::mma_inline_ptx) is on, if set.
    pub fn get_mma_inline_ptx(&self) -> Option<bool> {
        self.mma_inline_ptx
    }

    /// CuTe DSL's `cute.FrontendNext`, which selects NVIDIA's PyIR front end. It has no effect:
    /// cute-rs has a single front end (Rust closures recorded by the builder).
    pub fn frontend_next(self, on: bool) -> Self {
        if on {
            debug_log(|| "FrontendNext has no effect: cute-rs has a single front end".to_string());
        }
        self
    }

    /// CuTe DSL's `cute.EnablePYIR` (`--enable-pyir`), which enables its PyIR-to-SCF conversion.
    /// It has no effect: cute-rs records structured control flow directly.
    pub fn enable_pyir(self, on: bool) -> Self {
        if on {
            debug_log(|| NO_PYIR.to_string());
        }
        self
    }

    /// CuTe DSL's `cute.DisableCuteExtCompile`, which skips its `cute_ext` compiler. It has no
    /// effect: cute-rs has one compiler pipeline.
    pub fn disable_cute_ext_compile(self, on: bool) -> Self {
        if on {
            debug_log(|| "DisableCuteExtCompile has no effect: cute-rs has one compiler pipeline".to_string());
        }
        self
    }

    /// Raw flags for the PTX assembler (CuTe DSL's `cute.PtxasOptions('--maxrregcount=128')`,
    /// `--ptxas-options`, `ptx-options=`): a string split into words as a shell does. With a
    /// CUDA toolkit they are passed to `ptxas` after cute-rs's own (`-arch`, `-O<n>`,
    /// `-lineinfo`, `-v`); an `-O<n>` among them replaces cute-rs's. Without one, the ones
    /// the driver's JIT has an option for are mapped onto it (`--maxrregcount=N`, `-O<n>`,
    /// `-lineinfo`, `-v`; see [`jit_options_for_ptxas_flags`](crate::jit_options_for_ptxas_flags))
    /// and any other is an error that names `ptxas`.
    ///
    /// Panics if the string has an unbalanced quote.
    #[track_caller]
    pub fn ptxas_options(mut self, flags: &str) -> Self {
        match split_words(flags) {
            Ok(words) => self.ptxas_flags = Some(words),
            Err(e) => panic!("cute: ptxas options {flags:?}: {e}"),
        }
        self
    }

    /// Whether to assemble the PTX with a CUDA toolkit's `ptxas` (when one is found). By
    /// default it runs only when something needs it: `ptxas` flags
    /// ([`ptxas_options`](Self::ptxas_options)) or a kept cubin or SASS; otherwise the PTX is
    /// left to the driver's JIT, which may be a newer assembler than the toolkit's. `true`
    /// always runs it; `false` never does, and then no cubin is made (unless
    /// [`CudaContext`](crate::cuda) links one).
    pub fn use_ptxas(mut self, on: bool) -> Self {
        self.use_ptxas = Some(on);
        self
    }

    /// Bypass the in-memory and on-disk compile caches and the per-context module cache
    /// (CuTe DSL's `no_cache=True`; also `CUTE_RS_NO_CACHE`). Keeping an artifact bypasses
    /// them too.
    pub fn no_cache(mut self, on: bool) -> Self {
        self.no_cache = Some(on);
        self
    }

    /// Link these LLVM bitcode files into the module (CuTe DSL's `cute.LinkLibraries`), in
    /// addition to the ones the kernel's external functions name
    /// ([`ExternFn::source`](crate::ffi::ExternFn::source)). They are added to any already set.
    pub fn link_libraries<P: Into<PathBuf>>(mut self, paths: impl IntoIterator<Item = P>) -> Self {
        self.link_libraries.extend(paths.into_iter().map(Into::into));
        self
    }

    /// The bitcode files [`link_libraries`](Self::link_libraries) set.
    pub fn get_link_libraries(&self) -> &[PathBuf] {
        &self.link_libraries
    }

    /// Keep an artifact (CuTe DSL's `CUTE_DSL_KEEP` tokens, `cute.KeepPTX(True)` and the like):
    /// it is written to the dump directory and exposed on the
    /// [`CompiledKernel`](crate::CompiledKernel).
    pub fn keep(mut self, artifact: Artifact, on: bool) -> Self {
        self.keep.set(artifact, on);
        self
    }

    /// Keep the PTX (`cute.KeepPTX`, `--keep-ptx`).
    pub fn keep_ptx(self, on: bool) -> Self {
        self.keep(Artifact::Ptx, on)
    }

    /// Keep the cubin (`cute.KeepCUBIN`, `--keep-cubin`).
    pub fn keep_cubin(self, on: bool) -> Self {
        self.keep(Artifact::Cubin, on)
    }

    /// Keep the SASS (`cute.KeepSASS`, `--keep-sass`).
    pub fn keep_sass(self, on: bool) -> Self {
        self.keep(Artifact::Sass, on)
    }

    /// Keep every artifact, or none of them (`CUTE_RS_KEEP=all`).
    pub fn keep_all(mut self, on: bool) -> Self {
        Artifact::ALL.iter().for_each(|a| self.keep.set(*a, on));
        self
    }

    /// Flags for `nvdisasm` when it makes the SASS (CuTe DSL's `cute.NvdisasmOptions`,
    /// `--nvdisasm-options`), as a shell-style string. The default is `-g -c`: line
    /// information, and only the code.
    ///
    /// Panics if the string has an unbalanced quote.
    #[track_caller]
    pub fn nvdisasm_options(mut self, flags: &str) -> Self {
        match split_words(flags) {
            Ok(words) => self.nvdisasm_flags = Some(words),
            Err(e) => panic!("cute: nvdisasm options {flags:?}: {e}"),
        }
        self
    }

    /// The directory kept artifacts go to (CuTe DSL's `cute.DumpDir`, `--dump-dir`,
    /// `CUTE_DSL_DUMP_DIR`); it is created if needed. The default is the working directory.
    pub fn dump_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dump_dir = Some(dir.into());
        self
    }

    /// Print the emitted MLIR to standard output when compiling (`CUTE_DSL_PRINT_IR`).
    pub fn print_ir(mut self, on: bool) -> Self {
        self.print_ir = Some(on);
        self
    }

    /// Debug mode (`CUTE_RS_DEBUG`): line information on, so the MLIR carries locations, and
    /// every artifact kept. Validation of the IR and of launches runs every time anyway, and
    /// `RUST_BACKTRACE` gives full backtraces.
    pub fn debug(self, on: bool) -> Self {
        if on { self.generate_line_info(true).keep_all(true) } else { self }
    }

    /// The raw `ptxas` flags, if set.
    pub fn get_ptxas_options(&self) -> Option<&[String]> {
        self.ptxas_flags.as_deref()
    }

    /// Whether `ptxas` is used when found, if set.
    pub fn get_use_ptxas(&self) -> Option<bool> {
        self.use_ptxas
    }

    /// Whether the compile caches are bypassed ([`no_cache`](Self::no_cache)), if set.
    pub fn get_no_cache(&self) -> Option<bool> {
        self.no_cache
    }

    /// The key of the options that change what a compile produces (CuTe DSL keys its JIT
    /// cache on them): the target, optimization level, line information, inline-PTX MMAs,
    /// `ptxas` use and flags, assertions and the diagnostics asked for. Stage hooks, which
    /// cannot be compared, make a compile bypass the caches instead.
    pub fn cache_key(&self) -> String {
        format!(
            "arch={:?} O={:?} lineinfo={:?} mma_inline_ptx={:?} ptxas={:?} ptxas_flags={:?} assertions={:?} \
             warnings={:?} remarks={:?} warning_filter={:?} remark_filter={:?} werror={:?} wignore={:?} hooks={}{}",
            self.gpu_arch,
            self.opt_level,
            self.line_info,
            self.mma_inline_ptx,
            self.use_ptxas,
            self.ptxas_flags,
            self.enable_assertions,
            self.warnings,
            self.remarks,
            self.warning_filter,
            self.remark_filter,
            self.warnings_as_errors,
            self.warnings_ignore,
            self.hooks.len(),
            // Only when set, so that other options keep their keys.
            if self.link_libraries.is_empty() { String::new() } else { format!(" link={:?}", self.link_libraries) }
        ) + &self.fp_contract.map_or_else(String::new, |on| format!(" fp_contract={on}"))
    }

    /// Whether `artifact` is kept, if set.
    pub fn get_keep(&self, artifact: Artifact) -> Option<bool> {
        self.keep.get(artifact)
    }

    /// The directory for kept artifacts, if set.
    pub fn get_dump_dir(&self) -> Option<&Path> {
        self.dump_dir.as_deref()
    }

    /// The `nvdisasm` flags, if set.
    pub fn get_nvdisasm_options(&self) -> Option<&[String]> {
        self.nvdisasm_flags.as_deref()
    }

    /// Whether the MLIR is printed, if set.
    pub fn get_print_ir(&self) -> Option<bool> {
        self.print_ir
    }

    /// Compile device assertions ([`testing::assert_`](crate::testing::assert_)) in, as
    /// `cf.assert` (CuTe DSL's `cute.EnableAssertions`, `--enable-assertions`,
    /// `CUTE_RS_ENABLE_ASSERTIONS`). Off by default: the assertions are left out.
    pub fn enable_assertions(mut self, on: bool) -> Self {
        self.enable_assertions = Some(on);
        self
    }

    /// Whether assertions are compiled in, if set.
    pub fn get_enable_assertions(&self) -> Option<bool> {
        self.enable_assertions
    }

    pub(crate) fn effective_enable_assertions(&self) -> bool {
        self.enable_assertions.unwrap_or(false)
    }

    /// Whether the compiler may contract a kernel's Float16, BFloat16, Float32 and Float64
    /// `+`, `-` and `*` (a multiply feeding an add becomes one `fma`). On by default, as in
    /// nvcc (`--fmad=true`): the operations are emitted with `fastmath<contract>`, LLVM fuses
    /// what it can, and the rest lose the `.rn` that stops `ptxas` fusing them.
    /// Arithmetic-heavy kernels gain most: the main loop of
    /// `examples/flashinfer_batch_decode.rs` runs about a fifth fewer instructions.
    ///
    /// A fused multiply-add rounds once where the separate operations round twice, and the
    /// simulator never contracts, so with contraction the GPU's results can differ from the
    /// simulator's in the last bits. Turn it off (`fp_contract(false)`) for bit-for-bit
    /// agreement. Never contracted either way: explicit [`fma`](crate::fma), division,
    /// min/max, the `_with` exact-rounding functions, and everything the
    /// [`math`](crate::math) functions build (except `add`, `sub` and `mul` with
    /// `fastmath = true`).
    pub fn fp_contract(mut self, on: bool) -> Self {
        self.fp_contract = Some(on);
        self
    }

    /// Whether floating-point contraction is allowed, if set.
    pub fn get_fp_contract(&self) -> Option<bool> {
        self.fp_contract
    }

    pub(crate) fn effective_fp_contract(&self) -> bool {
        self.fp_contract.unwrap_or(true)
    }

    /// Mark the compile as one for a TVM-FFI callable (CuTe DSL's `cute.EnableTVMFFI`,
    /// `--enable-tvm-ffi`, `CUTE_RS_ENABLE_TVM_FFI`). The PTX is the same either way: the
    /// callable is the Rust wrapper [`TvmFfiFunction`](crate::tvm_ffi::TvmFfiFunction), which
    /// [`TvmFfiFunction::compile`](crate::tvm_ffi::TvmFfiFunction::compile) builds with this
    /// option set and [`tvm_ffi_export!`](crate::tvm_ffi_export) exports from a `cdylib`.
    pub fn enable_tvm_ffi(mut self, on: bool) -> Self {
        self.enable_tvm_ffi = Some(on);
        self
    }

    /// Whether the compile is marked as one for a TVM-FFI callable, if set.
    pub fn get_enable_tvm_ffi(&self) -> Option<bool> {
        self.enable_tvm_ffi
    }

    /// Report warnings of the given categories ([`CATEGORIES`](crate::diagnostics::CATEGORIES);
    /// none for every category): CuTe DSL's `warnings` and `warnings{ptx}`. The ptx category
    /// warns of register spills, at the kernel's `Kernel::build` line.
    pub fn warnings(mut self, categories: &[&str]) -> Self {
        self.warnings = Some(categories.iter().map(|c| c.to_string()).collect());
        self
    }

    /// Report remarks of the given categories (none for every category): CuTe DSL's
    /// `remarks` and `remarks{ptx}`, `remarks{loop}`. The ptx category gives the resource
    /// usage and local memory, target the PTX ISA raises and target fallbacks.
    pub fn remarks(mut self, categories: &[&str]) -> Self {
        self.remarks = Some(categories.iter().map(|c| c.to_string()).collect());
        self
    }

    /// Show only the warnings `filter` selects (CuTe DSL's `WarningsFilter`): `|`-separated
    /// categories or message substrings ([`Diagnostic::matches`](crate::Diagnostic::matches)).
    pub fn warning_filter(mut self, filter: &str) -> Self {
        self.warning_filter = Some(filter.to_string());
        self
    }

    /// Show only the remarks `filter` selects (CuTe DSL's `RemarkFilter`,
    /// `--remark-filter`).
    pub fn remark_filter(mut self, filter: &str) -> Self {
        self.remark_filter = Some(filter.to_string());
        self
    }

    /// Write the remarks to `path` as YAML (CuTe DSL's `RemarkOutput`, `--remark-output`).
    pub fn remark_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.remark_output = Some(path.into());
        self
    }

    /// Fail the compile when a warning is reported (`CUTE_RS_WARNINGS_AS_ERRORS`, CuTe DSL's
    /// `CUTE_DSL_WARNINGS_AS_ERRORS`).
    pub fn warnings_as_errors(mut self, on: bool) -> Self {
        self.warnings_as_errors = Some(on);
        self
    }

    /// Drop warnings (`CUTE_RS_WARNINGS_IGNORE`, CuTe DSL's `CUTE_DSL_WARNINGS_IGNORE`).
    pub fn warnings_ignore(mut self, on: bool) -> Self {
        self.warnings_ignore = Some(on);
        self
    }

    /// The warning categories asked for (`Some(empty)` for all), if any.
    pub fn get_warnings(&self) -> Option<&[String]> {
        self.warnings.as_deref()
    }

    /// The remark categories asked for (`Some(empty)` for all), if any.
    pub fn get_remarks(&self) -> Option<&[String]> {
        self.remarks.as_deref()
    }

    pub(crate) fn diagnostic_settings(&self) -> DiagnosticSettings<'_> {
        DiagnosticSettings {
            warnings: &self.warnings,
            remarks: &self.remarks,
            warning_filter: self.warning_filter.as_deref(),
            remark_filter: self.remark_filter.as_deref(),
            remark_output: self.remark_output.as_deref(),
            as_errors: self.warnings_as_errors.unwrap_or(false),
            ignore: self.warnings_ignore.unwrap_or(false),
        }
    }

    pub(crate) fn keeps(&self, a: Artifact) -> bool {
        self.keep.wants(a)
    }

    pub(crate) fn effective_nvdisasm(&self) -> Vec<String> {
        self.nvdisasm_flags.clone().unwrap_or_else(|| vec!["-g".to_string(), "-c".to_string()])
    }

    /// The target architecture, if set.
    pub fn get_gpu_arch(&self) -> Option<Arch> {
        self.gpu_arch
    }

    /// The optimization level, if set.
    pub fn get_opt_level(&self) -> Option<u32> {
        self.opt_level.map(u32::from)
    }

    /// Whether line information is on, if set.
    pub fn get_generate_line_info(&self) -> Option<bool> {
        self.line_info
    }

    /// The number of stage hooks.
    pub fn stage_hook_count(&self) -> usize {
        self.hooks.len()
    }

    pub(crate) fn effective_opt_level(&self) -> u8 {
        self.opt_level.unwrap_or(DEFAULT_OPT_LEVEL as u8)
    }

    pub(crate) fn effective_line_info(&self) -> bool {
        self.line_info.unwrap_or(false)
    }

    pub(crate) fn hooks(&self) -> &[StageHook] {
        &self.hooks
    }

    /// Fill every setting `self` leaves unset from `defaults`. Stage hooks from both run,
    /// `self`'s first.
    pub fn or(mut self, defaults: CompileOptions) -> CompileOptions {
        self.gpu_arch = self.gpu_arch.or(defaults.gpu_arch);
        self.opt_level = self.opt_level.or(defaults.opt_level);
        self.line_info = self.line_info.or(defaults.line_info);
        self.mma_inline_ptx = self.mma_inline_ptx.or(defaults.mma_inline_ptx);
        self.ptxas_flags = self.ptxas_flags.or(defaults.ptxas_flags);
        self.use_ptxas = self.use_ptxas.or(defaults.use_ptxas);
        self.keep = self.keep.or(defaults.keep);
        self.dump_dir = self.dump_dir.or(defaults.dump_dir);
        self.nvdisasm_flags = self.nvdisasm_flags.or(defaults.nvdisasm_flags);
        self.print_ir = self.print_ir.or(defaults.print_ir);
        self.enable_assertions = self.enable_assertions.or(defaults.enable_assertions);
        self.fp_contract = self.fp_contract.or(defaults.fp_contract);
        self.enable_tvm_ffi = self.enable_tvm_ffi.or(defaults.enable_tvm_ffi);
        self.warnings = self.warnings.or(defaults.warnings);
        self.remarks = self.remarks.or(defaults.remarks);
        self.warning_filter = self.warning_filter.or(defaults.warning_filter);
        self.remark_filter = self.remark_filter.or(defaults.remark_filter);
        self.remark_output = self.remark_output.or(defaults.remark_output);
        self.warnings_as_errors = self.warnings_as_errors.or(defaults.warnings_as_errors);
        self.warnings_ignore = self.warnings_ignore.or(defaults.warnings_ignore);
        self.no_cache = self.no_cache.or(defaults.no_cache);
        self.link_libraries.extend(defaults.link_libraries);
        self.hooks.extend(defaults.hooks);
        self
    }

    /// Parse an option string (CuTe DSL's `cute.compile(..., options="...")` and
    /// `CUTE_DSL_COMPILER_OPT`). Later options override earlier ones.
    ///
    /// ```
    /// use cute::{Arch, CompileOptions};
    ///
    /// let o = CompileOptions::parse("--opt-level=1 cubin-chip=sm_86 preserve-line-info").unwrap();
    /// assert_eq!((o.get_opt_level(), o.get_gpu_arch(), o.get_generate_line_info()), (Some(1), Some(Arch::Sm86), Some(true)));
    /// let e = CompileOptions::parse("--opt-level 7").unwrap_err();
    /// assert!(e.message().contains("0 to 3"), "{e}");
    /// let e = CompileOptions::parse("dump-ptx-path=maybe").unwrap_err();
    /// assert!(e.message().contains("takes 1, true"), "{e}");
    /// let o = CompileOptions::parse("--ptxas-options '--maxrregcount=32 -v' --keep-ptx").unwrap();
    /// assert_eq!(o.get_ptxas_options().unwrap(), ["--maxrregcount=32", "-v"]);
    /// ```
    pub fn parse(text: &str) -> Result<CompileOptions, OptionError> {
        let mut options = CompileOptions::new();
        options.apply(text)?;
        Ok(options)
    }

    /// The options set by the process environment: `CUTE_RS_COMPILER_OPT`, `CUTE_RS_ARCH`
    /// and `CUTE_RS_LINEINFO`. See [`from_env_with`](Self::from_env_with).
    pub fn from_env() -> Result<CompileOptions, OptionError> {
        CompileOptions::from_env_with(|name| std::env::var(name).ok())
    }

    /// The options set by the environment variables that `lookup` returns:
    ///
    /// * `CUTE_RS_COMPILER_OPT`: an option string, as [`parse`](Self::parse) takes;
    /// * `CUTE_RS_ARCH`: the target, such as `sm_90a`;
    /// * `CUTE_RS_LINEINFO`: `1` (also `true`, `yes`, `on`) switches line information on; `0`
    ///   (`false`, `no`, `off`) asks for nothing.
    ///
    /// The order is CuTe DSL's: a target in `CUTE_RS_COMPILER_OPT` (`cubin-chip=` or
    /// `--gpu-arch`) beats `CUTE_RS_ARCH`, and `CUTE_RS_LINEINFO` can only switch line
    /// information on, so `CUTE_RS_LINEINFO=1` overrides `preserve-line-info=false` while
    /// `CUTE_RS_LINEINFO=0` leaves what `CUTE_RS_COMPILER_OPT` sets. A variable that is set but
    /// invalid is an error even when another one takes precedence. Empty variables are ignored.
    ///
    /// The artifact variables: `CUTE_RS_KEEP` (a comma-separated list of `ir`, `mlir`, `llvm`,
    /// `ptx`, `cubin`, `sass`, `all`), `CUTE_RS_DUMP_DIR`, `CUTE_RS_PRINT_IR`, CuTe's deprecated
    /// per-artifact switches `CUTE_RS_KEEP_IR` (the MLIR), `CUTE_RS_KEEP_PTX`,
    /// `CUTE_RS_KEEP_CUBIN` and `CUTE_RS_KEEP_SASS`, and `CUTE_RS_DEBUG` (line information and
    /// every artifact). Like `CUTE_RS_LINEINFO`, they can only switch things on.
    ///
    /// CuTe DSL's other compile-option variables are recognised too: `CUTE_RS_ENABLE_PYIR` has
    /// no effect, and `CUTE_RS_ENABLE_ASSERTIONS` and `CUTE_RS_ENABLE_TVM_FFI` set
    /// [`enable_assertions`](Self::enable_assertions) and [`enable_tvm_ffi`](Self::enable_tvm_ffi).
    pub fn from_env_with(lookup: impl Fn(&str) -> Option<String>) -> Result<CompileOptions, OptionError> {
        let var = |name: &str| lookup(name).filter(|v| !v.trim().is_empty());
        let flag = |name: &str| -> Result<Option<bool>, OptionError> {
            var(name)
                .map(|v| {
                    parse_bool(&v).ok_or_else(|| {
                        OptionError::new(format!(
                            "{name}={v:?} is not a boolean: use 1, true, yes or on, or 0, false, no or off"
                        ))
                    })
                })
                .transpose()
        };
        let mut options = CompileOptions::new();
        if let Some(text) = var(ENV_COMPILER_OPT) {
            options.apply(&text).map_err(|e| e.within(ENV_COMPILER_OPT))?;
        }
        // CuTe DSL compiles for `compile_options.gpu_arch or envar.arch`, after applying
        // CUTE_DSL_COMPILER_OPT to the compile options.
        if let Some(arch) = var(ENV_ARCH) {
            let arch = parse_arch(&arch).map_err(|e| e.within(ENV_ARCH))?;
            options.gpu_arch = options.gpu_arch.or(Some(arch));
        }
        // CuTe DSL sets GenerateLineInfo when CUTE_DSL_LINEINFO is true, and never clears it.
        if flag(ENV_LINEINFO)? == Some(true) {
            options.line_info = Some(true);
        }
        if flag("CUTE_RS_ENABLE_PYIR")? == Some(true) {
            debug_log(|| format!("CUTE_RS_ENABLE_PYIR: {NO_PYIR}"));
        }
        if let Some(text) = var("CUTE_RS_KEEP") {
            let keep = Keep::parse(&text).map_err(|e| e.within("CUTE_RS_KEEP"))?;
            options.keep = keep.or(options.keep);
        }
        if let Some(dir) = var("CUTE_RS_DUMP_DIR") {
            options.dump_dir = options.dump_dir.or(Some(PathBuf::from(dir.trim())));
        }
        if flag("CUTE_RS_PRINT_IR")? == Some(true) {
            options.print_ir = Some(true);
        }
        // CuTe DSL 4.8.1 still honours its per-artifact switches, with a deprecation warning,
        // as the CUTE_DSL_KEEP tokens they stand for.
        for (name, artifact) in [
            ("CUTE_RS_KEEP_IR", Artifact::Mlir),
            ("CUTE_RS_KEEP_PTX", Artifact::Ptx),
            ("CUTE_RS_KEEP_CUBIN", Artifact::Cubin),
            ("CUTE_RS_KEEP_SASS", Artifact::Sass),
        ] {
            if flag(name)? == Some(true) {
                options.keep.set(artifact, true);
            }
        }
        if let Some(on) = flag("CUTE_RS_ENABLE_ASSERTIONS")? {
            options.enable_assertions = Some(on);
        }
        if let Some(on) = flag("CUTE_RS_WARNINGS_AS_ERRORS")? {
            options.warnings_as_errors = Some(on);
        }
        if let Some(on) = flag("CUTE_RS_WARNINGS_IGNORE")? {
            options.warnings_ignore = Some(on);
        }
        if let Some(on) = flag("CUTE_RS_ENABLE_TVM_FFI")? {
            options.enable_tvm_ffi = Some(on);
        }
        if flag("CUTE_RS_DEBUG")? == Some(true) {
            options.line_info = Some(true);
            Artifact::ALL.iter().for_each(|a| options.keep.set(*a, true));
        }
        Ok(options)
    }

    /// These options, with what they leave unset taken from the process environment.
    ///
    /// Explicit options win, which is a deliberate difference from CuTe DSL: there,
    /// `CUTE_DSL_COMPILER_OPT` is applied on top of the explicit options and
    /// `CUTE_DSL_LINEINFO=1` switches line information on even over an explicit
    /// `GenerateLineInfo(False)`. In cute-rs, options a program sets cannot be changed from
    /// outside it.
    pub fn with_env(self) -> Result<CompileOptions, OptionError> {
        Ok(self.or(CompileOptions::from_env()?))
    }

    /// These options, with what they leave unset taken from the environment variables that
    /// `lookup` returns (see [`from_env_with`](Self::from_env_with)).
    pub fn with_env_from(self, lookup: impl Fn(&str) -> Option<String>) -> Result<CompileOptions, OptionError> {
        Ok(self.or(CompileOptions::from_env_with(lookup)?))
    }

    /// Apply an option string on top of these options.
    fn apply(&mut self, text: &str) -> Result<(), OptionError> {
        let words = split_words(text)?;
        let mut i = 0;
        while i < words.len() {
            let word = &words[i];
            i += 1;
            if let Some(body) = word.strip_prefix("--") {
                let end = body.find(['=', '{']).unwrap_or(body.len());
                let (name, rest) = body.split_at(end);
                let spec = SPECS.iter().find(|s| s.flag == name);
                let compact = spec.is_none() && is_compact_name(name);
                let spec = match (spec, compact) {
                    (Some(spec), _) => Some(spec),
                    (None, true) => None,
                    (None, false) => Some(abbreviation(name, word)?),
                };
                if let Some(spec) = spec {
                    let spelled = format!("--{}", spec.flag);
                    if rest.starts_with('{') {
                        return Err(OptionError::new(format!(
                            "`{word}`: {spelled} has no selectors, so it takes no braces"
                        )));
                    }
                    let inline = rest.strip_prefix('=').map(str::to_string);
                    let joiner = if inline.is_some() { "=" } else { " " };
                    let value = match (spec.kind, inline) {
                        (Kind::Bool, Some(_)) => {
                            return Err(OptionError::new(format!(
                                "`{word}`: {spelled} takes no value (write `{}=false` to turn it off)",
                                spec.compact
                            )));
                        }
                        (Kind::Bool, None) => None,
                        (_, Some(v)) => Some(v),
                        (_, None) => match words.get(i) {
                            // argparse takes the next word as the value unless it looks like
                            // an option. CuTe DSL passes the word after a literal
                            // `--ptxas-options` through whatever it is, since ptxas's own
                            // options start with '-'; an abbreviation gets no such treatment.
                            Some(next) if word == "--ptxas-options" || !looks_like_flag(next) => {
                                i += 1;
                                Some(next.clone())
                            }
                            _ => None,
                        },
                    };
                    let shown = match &value {
                        Some(v) => format!("{spelled}{joiner}{v}"),
                        None => spelled.clone(),
                    };
                    self.set(spec, &spelled, &shown, value)?;
                    continue;
                }
                self.apply_compact(body, word)?;
            } else if word.starts_with('-') {
                return Err(OptionError::new(format!(
                    "unexpected `{word}`: options are spelled `--name`, `--name value` or `name=value`"
                )));
            } else if self.apply_compact(word, word)? == 0 && !is_separator(word) {
                // argparse rejects a stray word; the compact syntax only allows commas between
                // its options.
                let shown = if word.is_empty() { "empty word".to_string() } else { format!("`{word}`") };
                return Err(OptionError::new(format!(
                    "unexpected {shown}, which holds no option: options are spelled `--name`, `--name value` or \
                     `name=value`, separated by spaces or commas"
                )));
            }
        }
        Ok(())
    }

    /// Apply compact tokens (`name`, `name=value`, `name{a,b}`, separated by commas or
    /// spaces), as CuTe DSL tokenises `CUTE_DSL_COMPILER_OPT`, returning how many there were.
    fn apply_compact(&mut self, text: &str, word: &str) -> Result<usize, OptionError> {
        let tokens = compact_tokens(text, word)?;
        let count = tokens.len();
        for token in tokens {
            let CompactToken { name, braces, value } = token;
            match name.as_str() {
                "warnings" | "remarks" => {
                    let shown = match &braces {
                        Some(b) => format!("{name}{{{b}}}"),
                        None => name.clone(),
                    };
                    if value.is_some() {
                        return Err(OptionError::new(format!(
                            "`{word}`: {name} takes a braced category list ({name}{{ptx,loop}}), not a value"
                        )));
                    }
                    let cats = crate::diagnostics::parse_categories(braces.as_deref(), &shown)?;
                    let slot = if name == "warnings" { &mut self.warnings } else { &mut self.remarks };
                    // Each spelling adds its categories; a bare one asks for all.
                    *slot = Some(match slot.take() {
                        None => cats,
                        Some(have) if have.is_empty() || cats.is_empty() => Vec::new(),
                        Some(mut have) => {
                            have.extend(cats);
                            have
                        }
                    });
                }
                "iket" => {
                    if braces.is_some() || value.is_some() {
                        return Err(OptionError::new(format!("`{word}`: iket takes no value or selectors")));
                    }
                    return Err(pending("iket", "IKET in-kernel event tracing"));
                }
                "debug" => {
                    let Some(selectors) = braces.filter(|_| value.is_none()) else {
                        return Err(OptionError::new(format!(
                            "`{word}`: debug must be followed by a braced selector list, such as debug{{launch-check}}"
                        )));
                    };
                    for s in selectors.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        if s != "launch-check" {
                            return Err(OptionError::new(format!(
                                "`{word}`: `{s}` is not a debug selector (launch-check is the only one)"
                            )));
                        }
                    }
                    debug_log(|| {
                        "debug{launch-check} has no effect: CudaContext::launch and Kernel::run_cuda check each \
                         launch's configuration and arguments already, launch_on_device and launch_async check the \
                         argument count and kinds and the shared memory size, and the driver rejects configurations \
                         beyond the device's limits"
                            .to_string()
                    });
                }
                _ => {
                    let Some(spec) = SPECS.iter().find(|s| s.compact == name || s.flag == name) else {
                        return Err(OptionError::new(format!(
                            "unknown option `{name}` in `{word}` (cute-rs has no NVIDIA pass pipeline to forward raw \
                             options to; the README lists the options it accepts)"
                        )));
                    };
                    if braces.is_some() {
                        return Err(OptionError::new(format!(
                            "`{word}`: {name} has no selectors, so it takes no braces"
                        )));
                    }
                    let shown = match &value {
                        Some(v) => format!("{name}={v}"),
                        None => name.clone(),
                    };
                    self.set(spec, &name, &shown, value)?;
                }
            }
        }
        Ok(count)
    }

    /// Set the option `spec`, spelled `spelled` (`shown` with its value), to `value` (`None`
    /// for a bare boolean flag).
    fn set(&mut self, spec: &Spec, spelled: &str, shown: &str, value: Option<String>) -> Result<(), OptionError> {
        // Whether the option asks for anything: a true flag or a non-empty value.
        let active = match (spec.kind, &value) {
            (Kind::Bool, None) => true,
            (Kind::Bool, Some(v)) => parse_bool(v).ok_or_else(|| {
                OptionError::new(format!(
                    "`{shown}`: {spelled} takes 1, true, yes or on, or 0, false, no or off, not `{v}`"
                ))
            })?,
            (_, None) => {
                return Err(OptionError::new(format!(
                    "{spelled} needs a value (write `{spelled}=<value>` or `--{} <value>`)",
                    spec.flag
                )));
            }
            (_, Some(v)) => !v.trim().is_empty(),
        };
        let text = value.unwrap_or_default();
        match spec.effect {
            Effect::OptLevel => {
                self.opt_level = Some(parse_opt_level(&text).map_err(|m| OptionError::new(format!("`{shown}`: {m}")))?)
            }
            Effect::GpuArch if text.trim().is_empty() => self.gpu_arch = None,
            Effect::GpuArch => {
                self.gpu_arch = Some(parse_arch(&text).map_err(|e| OptionError::new(format!("`{shown}`: {e}")))?)
            }
            Effect::LineInfo => self.line_info = Some(active),
            Effect::PtxasOptions if active => self.ptxas_flags = Some(split_words(&text)?),
            Effect::NvdisasmOptions if active => self.nvdisasm_flags = Some(split_words(&text)?),
            Effect::DumpDir if active => self.dump_dir = Some(PathBuf::from(text.trim())),
            Effect::PtxasOptions | Effect::NvdisasmOptions | Effect::DumpDir => {}
            Effect::Keep(artifact) => self.keep.set(artifact, active),
            Effect::HostTarget => check_host_target(&text).map_err(|m| OptionError::new(format!("`{shown}`: {m}")))?,
            Effect::NoOp(why) if active => debug_log(|| format!("{shown} has no effect: {why}")),
            Effect::EnableAssertions => self.enable_assertions = Some(active),
            Effect::EnableTvmFfi => self.enable_tvm_ffi = Some(active),
            Effect::RemarkFilter => self.remark_filter = active.then(|| text.trim().to_string()),
            Effect::RemarkOutput => self.remark_output = active.then(|| PathBuf::from(text.trim())),
            Effect::Pending { feature } if active => return Err(pending(shown, feature)),
            Effect::Unsupported(why) if active => {
                return Err(OptionError::new(format!("`{shown}` is not supported: {why}")));
            }
            Effect::NoOp(_) | Effect::Pending { .. } | Effect::Unsupported(_) => {}
        }
        Ok(())
    }
}

const NO_PYIR: &str = "cute-rs records structured control flow directly, with no PyIR to convert";

fn pending(shown: &str, feature: &str) -> OptionError {
    OptionError::new(format!("`{shown}`: {feature} is not available yet"))
}

fn opt_level_range(level: &str) -> String {
    format!("the optimization level must be 0 to 3 (0 for none, 3 for the most), not {level}")
}

fn parse_opt_level(text: &str) -> Result<u8, String> {
    let t = text.trim();
    match t.parse::<i64>() {
        Ok(n) if (0..=3).contains(&n) => Ok(n as u8),
        Ok(n) => Err(opt_level_range(&n.to_string())),
        Err(_) => Err(opt_level_range(&format!("`{t}`"))),
    }
}

fn parse_arch(text: &str) -> Result<Arch, OptionError> {
    text.trim().parse::<Arch>().map_err(|e| OptionError::new(e.to_string()))
}

/// CuTe DSL's boolean spellings.
pub(crate) fn parse_bool(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// A word that only separates compact options.
fn is_separator(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c == ',')
}

/// Whether argparse takes `word` for an option rather than for the value of the option before
/// it. A word that starts with `-` (other than `-` itself) is an option if it starts with an
/// option's name (`--gpu=x y`, `-h...`); otherwise it is a value if it is a negative number
/// (`-` then a digit or `.` and a digit, as in current Python) or contains a space
/// (`'-g -c'`).
fn looks_like_flag(word: &str) -> bool {
    let Some(rest) = word.strip_prefix('-').filter(|r| !r.is_empty()) else { return false };
    let names_an_option = match rest.strip_prefix('-') {
        Some(long) => {
            let name = long.split_once('=').map_or(long, |(name, _)| name);
            !name.is_empty() && ("help".starts_with(name) || SPECS.iter().any(|s| s.flag.starts_with(name)))
        }
        // argparse's own `-h`, with or without anything attached.
        None => rest.starts_with('h'),
    };
    let negative_number = rest.strip_prefix('.').unwrap_or(rest).starts_with(|c: char| c.is_ascii_digit());
    names_an_option || !(negative_number || word.contains(' '))
}

/// The flag `name` abbreviates (argparse accepts unambiguous prefixes), or an error.
/// argparse's own `--help` counts, as in CuTe DSL, so `--h` is ambiguous; cute-rs prints no
/// help, so `--help` and its prefixes are errors.
fn abbreviation(name: &str, word: &str) -> Result<&'static Spec, OptionError> {
    let help = !name.is_empty() && "help".starts_with(name);
    let matches: Vec<&Spec> =
        if name.is_empty() { Vec::new() } else { SPECS.iter().filter(|s| s.flag.starts_with(name)).collect() };
    match (matches.as_slice(), help) {
        ([spec], false) => Ok(spec),
        ([], false) => {
            Err(OptionError::new(format!("unknown option `{word}` (the README lists the options cute-rs accepts)")))
        }
        ([], true) => Err(OptionError::new(format!(
            "`{word}` asks for argparse's help, which cute-rs does not print: the README lists the options it accepts"
        ))),
        (several, help) => {
            let names: Vec<String> = help
                .then(|| "--help".to_string())
                .into_iter()
                .chain(several.iter().map(|s| format!("--{}", s.flag)))
                .collect();
            Err(OptionError::new(format!("ambiguous option `{word}`: it could be {}", names.join(", "))))
        }
    }
}

fn is_compact_name(name: &str) -> bool {
    matches!(name, "warnings" | "remarks" | "iket" | "debug") || SPECS.iter().any(|s| s.compact == name)
}

/// What a recognised option does.
#[derive(Clone, Copy)]
enum Effect {
    OptLevel,
    GpuArch,
    LineInfo,
    HostTarget,
    PtxasOptions,
    NvdisasmOptions,
    DumpDir,
    Keep(Artifact),
    /// Accepted, with no effect for the given reason.
    NoOp(&'static str),
    EnableAssertions,
    EnableTvmFfi,
    RemarkFilter,
    RemarkOutput,
    /// Rejected until `part` lands.
    Pending {
        feature: &'static str,
    },
    /// Rejected for good, for the given reason.
    Unsupported(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bool,
    Int,
    Str,
}

/// One CuTe DSL option: its argparse flag (without `--`), its compact name (the
/// `CUTE_DSL_COMPILER_OPT` key, CuTe's `_option_name`), its value kind and its effect.
struct Spec {
    flag: &'static str,
    compact: &'static str,
    kind: Kind,
    effect: Effect,
}

/// CuTe DSL 4.8.1's string options (`cutlass/base_dsl/compiler.py`).
const SPECS: &[Spec] = &[
    Spec { flag: "opt-level", compact: "opt-level", kind: Kind::Int, effect: Effect::OptLevel },
    Spec { flag: "gpu-arch", compact: "cubin-chip", kind: Kind::Str, effect: Effect::GpuArch },
    Spec { flag: "generate-line-info", compact: "preserve-line-info", kind: Kind::Bool, effect: Effect::LineInfo },
    Spec { flag: "host-target", compact: "host-target", kind: Kind::Str, effect: Effect::HostTarget },
    Spec { flag: "enable-pyir", compact: "enable-pyir", kind: Kind::Bool, effect: Effect::NoOp(NO_PYIR) },
    Spec { flag: "ptxas-options", compact: "ptx-options", kind: Kind::Str, effect: Effect::PtxasOptions },
    Spec { flag: "keep-ptx", compact: "dump-ptx-path", kind: Kind::Bool, effect: Effect::Keep(Artifact::Ptx) },
    Spec { flag: "keep-cubin", compact: "dump-cubin-path", kind: Kind::Bool, effect: Effect::Keep(Artifact::Cubin) },
    Spec { flag: "keep-sass", compact: "keep-sass", kind: Kind::Bool, effect: Effect::Keep(Artifact::Sass) },
    Spec { flag: "nvdisasm-options", compact: "nvdisasm-options", kind: Kind::Str, effect: Effect::NvdisasmOptions },
    Spec { flag: "dump-dir", compact: "dump-dir", kind: Kind::Str, effect: Effect::DumpDir },
    Spec {
        flag: "enable-assertions",
        compact: "enable-assertions",
        kind: Kind::Bool,
        effect: Effect::EnableAssertions,
    },
    Spec { flag: "remark-filter", compact: "remark-filter", kind: Kind::Str, effect: Effect::RemarkFilter },
    Spec { flag: "remark-output", compact: "remark-output", kind: Kind::Str, effect: Effect::RemarkOutput },
    Spec { flag: "enable-tvm-ffi", compact: "enable-tvm-ffi", kind: Kind::Bool, effect: Effect::EnableTvmFfi },
    Spec {
        flag: "link-libraries",
        compact: "link-libraries",
        kind: Kind::Str,
        effect: Effect::Pending { feature: "linking device libraries" },
    },
    Spec {
        flag: "rdc",
        compact: "rdc",
        kind: Kind::Bool,
        effect: Effect::Pending { feature: "relocatable device code" },
    },
    Spec {
        flag: "flatten-locs-output-json-path",
        compact: "flatten-locs-output-json-path",
        kind: Kind::Str,
        effect: Effect::Unsupported(
            "cute-rs locations are single Rust file:line:column positions, with no location chains to flatten",
        ),
    },
];

struct CompactToken {
    name: String,
    braces: Option<String>,
    value: Option<String>,
}

/// Tokenise compact options as CuTe DSL does, with the regex
/// `([\w][\w-]*)(?:\{([^}]*)\}|=(\S+))?` (anything between matches, such as commas, spaces and
/// leading dashes, separates tokens), and its checks for malformed tokens. A value runs to the
/// end of the word, so a quoted value may contain spaces.
fn compact_tokens(text: &str, word: &str) -> Result<Vec<CompactToken>, OptionError> {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !is_word(chars[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && (is_word(chars[i]) || chars[i] == '-') {
            i += 1;
        }
        let name: String = chars[start..i].iter().collect();
        let mut token = CompactToken { name, braces: None, value: None };
        match chars.get(i) {
            Some('{') => {
                let Some(close) = chars[i..].iter().position(|&c| c == '}') else {
                    return Err(OptionError::new(format!(
                        "`{word}`: the `{{` after `{}` has no matching `}}`",
                        token.name
                    )));
                };
                let inner: String = chars[i + 1..i + close].iter().collect();
                if inner.trim().is_empty() {
                    return Err(OptionError::new(format!(
                        "`{word}`: the braces after `{}` hold nothing: put selectors in them or leave them out",
                        token.name
                    )));
                }
                token.braces = Some(inner);
                i += close + 1;
            }
            Some('=') => {
                let value: String = chars[i + 1..].iter().collect();
                if value.trim().is_empty() {
                    return Err(OptionError::new(format!(
                        "`{word}`: nothing follows the `=` of `{}`: give a value after it, or drop the `=` to turn a \
                         flag on",
                        token.name
                    )));
                }
                token.value = Some(value);
                i = chars.len();
            }
            _ => {}
        }
        tokens.push(token);
    }
    Ok(tokens)
}

/// Split `text` into words as a POSIX shell (and Python's `shlex.split`) does: whitespace
/// separates words; single quotes keep everything literally; inside double quotes a backslash
/// escapes only `"` and `\`; elsewhere a backslash escapes any character.
pub(crate) fn split_words(text: &str) -> Result<Vec<String>, OptionError> {
    let unterminated = || OptionError::new(format!("unterminated quote in the option string `{text}`"));
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            '\'' => {
                in_word = true;
                loop {
                    match chars.next().ok_or_else(unterminated)? {
                        '\'' => break,
                        c => word.push(c),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next().ok_or_else(unterminated)? {
                        '"' => break,
                        '\\' => match chars.next().ok_or_else(unterminated)? {
                            c @ ('"' | '\\') => word.push(c),
                            c => {
                                word.push('\\');
                                word.push(c);
                            }
                        },
                        c => word.push(c),
                    }
                }
            }
            '\\' => {
                in_word = true;
                let next = chars
                    .next()
                    .ok_or_else(|| OptionError::new(format!("the option string `{text}` ends with a backslash")))?;
                word.push(next);
            }
            c => {
                in_word = true;
                word.push(c);
            }
        }
    }
    if in_word {
        words.push(word);
    }
    Ok(words)
}

/// CuTe DSL's `--host-target` presets.
const HOST_PRESETS: &[(&str, &str)] =
    &[("linux-aarch64", "aarch64-unknown-linux-gnu"), ("qnx8-aarch64", "aarch64-unknown-nto-qnx8.0.0")];

/// Accept a `--host-target` that names this host; reject any other. CuTe DSL uses it to
/// cross-compile the host object of an AOT export. cute-rs has no host object: the host
/// program is Rust, which cargo cross-compiles, and the PTX does not depend on the host.
fn check_host_target(spec: &str) -> Result<(), String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(());
    }
    let triple = if spec.starts_with("llvm") {
        let words = split_words(spec).map_err(|e| e.to_string())?;
        if words.first().map(String::as_str) != Some("llvm") {
            return Err(format!("`{spec}` is not `llvm -mtriple=<triple> [-mcpu=<cpu>] [-mattr=<features>]`"));
        }
        let mut triple = None;
        for w in &words[1..] {
            if let Some(t) = w.strip_prefix("-mtriple=") {
                triple = Some(t.to_string());
            } else if !(w.starts_with("-mcpu=") || w.starts_with("-mattr=")) {
                return Err(format!("`{w}` is not -mtriple=, -mcpu= or -mattr=, the only flags of the long form"));
            }
        }
        triple.ok_or_else(|| format!("`{spec}` has no -mtriple=<triple>"))?
    } else if let Some((_, triple)) = HOST_PRESETS.iter().find(|(name, _)| *name == spec) {
        triple.to_string()
    } else {
        let presets: Vec<&str> = HOST_PRESETS.iter().map(|(n, _)| *n).collect();
        return Err(format!(
            "`{spec}` is neither a preset ({}) nor `llvm -mtriple=<triple> [-mcpu=<cpu>] [-mattr=<features>]`",
            presets.join(", ")
        ));
    };
    if is_native_triple(&triple, std::env::consts::ARCH, std::env::consts::OS) {
        debug_log(|| {
            format!(
                "--host-target {triple} names this host and has no effect: cute-rs builds no host object, and \
                 -mcpu/-mattr tuning of the Rust host program belongs in RUSTFLAGS"
            )
        });
        Ok(())
    } else {
        Err(format!(
            "cute-rs builds no host object to cross-compile for {triple}: the host program is Rust, so cross-compile \
             it with `cargo build --target <triple>` (the PTX does not depend on the host)"
        ))
    }
}

/// Whether LLVM target triple `triple` names a host with Rust's `arch` and `os`
/// (`std::env::consts::ARCH` and `OS`).
pub(crate) fn is_native_triple(triple: &str, arch: &str, os: &str) -> bool {
    let mut parts = triple.split('-');
    let triple_arch = match parts.next().unwrap_or("") {
        "arm64" => "aarch64",
        "amd64" | "x86_64h" => "x86_64",
        other => other,
    };
    let os_names: Vec<&str> = match os {
        "macos" => vec!["darwin", "macos", "macosx"],
        "nto" => vec!["nto", "qnx"],
        other => vec![other],
    };
    triple_arch == arch && parts.any(|p| os_names.iter().any(|n| p.starts_with(n)))
}

/// Log `message` at `debug` level (see [`crate::log`]).
pub(crate) fn debug_log(message: impl FnOnce() -> String) {
    crate::log::log_with(crate::log::Level::Debug, message);
}

/// Hashes [`CompileOptions::cache_key`].
impl std::hash::Hash for CompileOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.cache_key().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> CompileOptions {
        CompileOptions::parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
    }

    fn error(text: &str) -> String {
        match CompileOptions::parse(text) {
            Ok(o) => panic!("{text:?} should be rejected, got {o:?}"),
            Err(e) => e.to_string(),
        }
    }

    fn settings(o: &CompileOptions) -> (Option<Arch>, Option<u32>, Option<bool>) {
        (o.get_gpu_arch(), o.get_opt_level(), o.get_generate_line_info())
    }

    #[test]
    fn opt_level_spellings() {
        for text in [
            "--opt-level 2",
            "--opt-level=2",
            "opt-level=2",
            "--opt 2",
            "--opt-l=2",
            "  --opt-level   '2' ",
            "--opt-level \"2\"",
            "--opt-level +2",
            "--opt-level 0 --opt-level 2",
        ] {
            assert_eq!(parse(text).get_opt_level(), Some(2), "{text}");
        }
        for n in 0..=3 {
            assert_eq!(parse(&format!("--opt-level {n}")).get_opt_level(), Some(n));
            assert_eq!(CompileOptions::new().opt_level(n).get_opt_level(), Some(n));
        }
        assert_eq!(CompileOptions::new().get_opt_level(), None);
        assert_eq!(CompileOptions::new().effective_opt_level(), 3);
    }

    #[test]
    fn invalid_opt_levels() {
        for (text, shown) in [
            ("--opt-level 4", "not 4"),
            ("--opt-level=-1", "not -1"),
            ("--opt-level -1", "not -1"),
            ("opt-level=9", "not 9"),
            ("--opt-level x", "not `x`"),
            ("--opt-level 2.0", "not `2.0`"),
            ("opt-level=99999999999999999999", "not `99999999999999999999`"),
        ] {
            let e = error(text);
            assert!(e.contains("must be 0 to 3") && e.contains(shown), "{text}: {e}");
        }
        for text in ["--opt-level", "--opt-level --generate-line-info", "opt-level"] {
            assert!(error(text).contains("needs a value"), "{text}: {}", error(text));
        }
        assert!(error("opt-level=").contains("nothing follows the `=`"));
    }

    #[test]
    #[should_panic(expected = "the optimization level must be 0 to 3 (0 for none, 3 for the most), not 4")]
    fn typed_opt_level_out_of_range_panics() {
        let _ = CompileOptions::new().opt_level(4);
    }

    #[test]
    fn gpu_arch_spellings() {
        for (text, arch) in [
            ("--gpu-arch sm_90a", Arch::Sm90a),
            ("--gpu-arch=sm_80", Arch::Sm80),
            ("cubin-chip=sm_86", Arch::Sm86),
            ("gpu-arch=sm_120a", Arch::Sm120a),
            ("--cubin-chip=sm_100", Arch::Sm100),
            ("--gpu sm_89", Arch::Sm89),
        ] {
            assert_eq!(parse(text).get_gpu_arch(), Some(arch), "{text}");
        }
        assert_eq!(parse("--gpu-arch sm_80 --gpu-arch ''").get_gpu_arch(), None);
        let e = error("--gpu-arch sm_99");
        assert!(e.contains("sm_99") && e.contains("sm_80"), "{e}");
        assert_eq!(parse("--gpu-arch sm_103a").get_gpu_arch(), Some(Arch::Sm103a));
        assert_eq!(parse("--gpu-arch sm_101a").get_gpu_arch(), Some(Arch::Sm110a), "Thor's old name");
        assert!(error("--gpu-arch sm_88").contains("not in CuTe DSL 4.8.1's Arch"));
        assert!(error("--gpu-arch").contains("needs a value"));
        // argparse does not take a flag-like word as a value.
        assert!(error("--gpu-arch --opt-level 1").contains("needs a value"));
    }

    #[test]
    fn line_info_spellings() {
        for (text, on) in [
            ("--generate-line-info", true),
            ("preserve-line-info", true),
            ("--preserve-line-info", true),
            ("generate-line-info", true),
            ("preserve-line-info=1", true),
            ("preserve-line-info=on", true),
            ("preserve-line-info=YES", true),
            ("preserve-line-info=false", false),
            ("preserve-line-info=0", false),
            ("--generate-line-info preserve-line-info=off", false),
            ("--generate", true),
        ] {
            assert_eq!(parse(text).get_generate_line_info(), Some(on), "{text}");
        }
        let e = error("--generate-line-info=false");
        assert!(e.contains("takes no value") && e.contains("preserve-line-info=false"), "{e}");
        assert!(error("preserve-line-info=maybe").contains("not `maybe`"));
    }

    #[test]
    fn compact_separators_and_mixtures() {
        assert_eq!(settings(&parse("preserve-line-info,cubin-chip=sm_80")), (Some(Arch::Sm80), None, Some(true)));
        assert_eq!(
            settings(&parse("--gpu-arch sm_86 opt-level=0 --generate-line-info")),
            (Some(Arch::Sm86), Some(0), Some(true))
        );
        assert_eq!(settings(&parse("")), (None, None, None));
        assert_eq!(settings(&parse("   ")), (None, None, None));
    }

    /// As in CuTe DSL, a value runs to the end of its word, commas included.
    #[test]
    fn values_with_commas_run_to_the_end_of_the_word() {
        let e = error("opt-level=1,preserve-line-info");
        assert!(e.contains("not `1,preserve-line-info`"), "{e}");
    }

    #[test]
    fn quoting() {
        assert_eq!(
            split_words(r#"a 'b c' "d \"e\" \\ \n" f\ g 'h'i "" x"#).unwrap(),
            ["a", "b c", r#"d "e" \ \n"#, "f g", "hi", "", "x"]
        );
        assert!(error("--opt-level '2").contains("unterminated quote"));
        assert!(error("--opt-level \"2").contains("unterminated quote"));
        assert!(error("--opt-level 2\\").contains("ends with a backslash"));
        // A quoted value keeps its spaces, even in the compact form.
        assert_eq!(parse("'ptx-options=-v -O3'").get_ptxas_options().unwrap(), ["-v", "-O3"]);
    }

    #[test]
    fn unknown_and_malformed_options() {
        for text in ["--frobnicate", "--frobnicate=1", "frobnicate", "--x", "-v", "--"] {
            let e = error(text);
            assert!(e.contains("unknown option") || e.contains("unexpected") || e.contains("ambiguous"), "{text}: {e}");
        }
        let e = error("--keep 1");
        assert!(e.contains("ambiguous") && e.contains("--keep-ptx") && e.contains("--keep-cubin"), "{e}");
        assert!(error("frobnicate=3").contains("no NVIDIA pass pipeline"));
        assert!(error("opt-level{2}").contains("takes no braces"));
        assert!(error("--opt-level{2}").contains("takes no braces"));
        assert!(error("debug{").contains("has no matching"));
        assert!(error("debug{}").contains("hold nothing"));
        assert!(error("-O3").contains("unexpected `-O3`"));
        // A word with no option in it is rejected, as argparse rejects a stray word; only
        // commas may stand alone, as separators of the compact syntax.
        assert!(error("--opt-level 1 -").contains("unexpected `-`"));
        for text in ["''", "--opt-level 1 \"\"", "=", "{}", ";", ",-", "+"] {
            let e = error(text);
            assert!(e.contains("holds no option"), "{text}: {e}");
        }
        assert!(error("''").contains("unexpected empty word"));
        assert_eq!(settings(&parse("preserve-line-info , opt-level=2 ,,")), (None, Some(2), Some(true)));
        // argparse's own --help makes --h ambiguous; cute-rs prints no help.
        let e = error("--h x");
        assert!(e.contains("ambiguous option `--h`: it could be --help, --host-target"), "{e}");
        for text in ["--help", "--he", "--help=1"] {
            assert!(error(text).contains("asks for argparse's help"), "{text}: {}", error(text));
        }
        assert!(error("--ho 'llvm -mtriple=riscv64-unknown-linux-gnu'").contains("cargo build --target"));
    }

    /// A value that starts with `-` follows its flag as argparse allows: a negative number, a
    /// word with a space, or anything after a literal `--ptxas-options`.
    #[test]
    fn values_that_start_with_a_dash() {
        for (word, flag) in [
            ("-v", true),
            ("--opt-level", true),
            ("--opt", true),
            ("--", true),
            ("-h", true),
            ("-h x", true),
            ("--gpu-arch=x y", true),
            ("--gpu=x y", true),
            ("-x\ty", true),
            ("-inf", true),
            ("-1", false),
            ("-2.5", false),
            ("-.5", false),
            ("-1e5", false),
            ("-g -c", false),
            ("--gpu x", false),
            ("- x", false),
            ("-", false),
            ("x", false),
            ("", false),
        ] {
            assert_eq!(looks_like_flag(word), flag, "{word:?}");
        }
        assert!(error("--opt-level -1").contains("not -1"));
        // CuTe DSL's own default for --nvdisasm-options, '-g -c', is a value.
        assert_eq!(parse("--nvdisasm-options '-g -c'").get_nvdisasm_options().unwrap(), ["-g", "-c"]);
        assert!(error("--nvdisasm-options -g").contains("--nvdisasm-options needs a value"));
        assert_eq!(parse("--dump-dir -1").get_dump_dir(), Some(Path::new("-1")));
        // Only the literal --ptxas-options takes a following option-like word.
        assert_eq!(parse("--ptxas-options -v").get_ptxas_options().unwrap(), ["-v"]);
        assert_eq!(parse("--ptxas-options --opt-level").get_ptxas_options().unwrap(), ["--opt-level"]);
        assert!(error("--ptxas -v").contains("--ptxas-options needs a value"));
    }

    #[test]
    fn unavailable_options_say_so() {
        for text in ["--link-libraries libdevice.bc", "--rdc", "rdc", "iket", "--iket"] {
            let e = error(text);
            assert!(e.contains("not available yet"), "{text}: {e}");
        }
        // The value after --ptxas-options is consumed even though it starts with '-'; other
        // flags take such a value only after `=` (see values_that_start_with_a_dash).
        assert!(error("--nvdisasm-options -c").contains("needs a value (write `--nvdisasm-options=<value>`"));
        // Turning an option off is accepted.
        let o = parse("enable-assertions=false dump-ptx-path=0 --opt-level 1");
        assert_eq!(o.get_enable_assertions(), Some(false));
        assert_eq!(o.get_keep(Artifact::Ptx), Some(false));
        assert_eq!(o.get_opt_level(), Some(1));
        assert!(error("iket{a}").contains("takes no value or selectors"));
        let e = error("--flatten-locs-output-json-path locs.json");
        assert!(e.contains("not supported") && e.contains("location chains"), "{e}");
    }

    #[test]
    fn front_end_options_are_accepted_and_do_nothing() {
        for text in ["--enable-pyir", "enable-pyir", "enable-pyir=1", "debug{launch-check}", "--debug{launch-check}"] {
            assert_eq!(settings(&parse(text)), (None, None, None), "{text}");
        }
        let typed = CompileOptions::new().frontend_next(true).enable_pyir(true).disable_cute_ext_compile(true);
        assert_eq!(settings(&typed), (None, None, None));
        assert!(error("debug").contains("braced selector list"));
        assert!(error("debug{launch-check,trace}").contains("`trace` is not a debug selector"));
    }

    #[test]
    fn host_target() {
        let native = format!(
            "{}-unknown-{}",
            std::env::consts::ARCH,
            if std::env::consts::OS == "macos" { "apple-darwin".to_string() } else { std::env::consts::OS.to_string() }
        );
        for text in [
            format!("--host-target 'llvm -mtriple={native}'"),
            format!("--host-target \"llvm -mtriple={native} -mcpu=native -mattr=+lse\""),
            format!("host-target='llvm -mtriple={native}'"),
            "--host-target ''".to_string(),
        ] {
            assert_eq!(settings(&parse(&text)), (None, None, None), "{text}");
        }
        let foreign = if std::env::consts::ARCH == "riscv64" { "x86_64" } else { "riscv64" };
        let e = error(&format!("--host-target 'llvm -mtriple={foreign}-unknown-linux-gnu'"));
        assert!(e.contains("cargo build --target"), "{e}");
        let e = error("--host-target qnx8-aarch64");
        assert!(e.contains("aarch64-unknown-nto-qnx8.0.0") && e.contains("cargo build --target"), "{e}");
        assert!(error("--host-target windows-x86").contains("neither a preset"));
        assert!(error("--host-target 'llvm -mcpu=znver3'").contains("no -mtriple"));
        assert!(error("--host-target 'llvm -mtriple=x -O3'").contains("the only flags of the long form"));
        if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            assert!(CompileOptions::parse("--host-target linux-aarch64").is_ok());
        } else {
            assert!(CompileOptions::parse("--host-target linux-aarch64").is_err());
        }
    }

    #[test]
    fn native_triples() {
        assert!(is_native_triple("aarch64-apple-darwin", "aarch64", "macos"));
        assert!(is_native_triple("arm64-apple-macosx14.0.0", "aarch64", "macos"));
        assert!(is_native_triple("x86_64-unknown-linux-gnu", "x86_64", "linux"));
        assert!(is_native_triple("aarch64-unknown-linux-gnu", "aarch64", "linux"));
        assert!(is_native_triple("aarch64-unknown-nto-qnx8.0.0", "aarch64", "nto"));
        assert!(is_native_triple("x86_64-pc-windows-msvc", "x86_64", "windows"));
        assert!(is_native_triple("x86_64-unknown-freebsd", "x86_64", "freebsd"));
        assert!(!is_native_triple("aarch64-unknown-linux-gnu", "x86_64", "linux"));
        assert!(!is_native_triple("aarch64-unknown-linux-gnu", "aarch64", "macos"));
        assert!(!is_native_triple("aarch64-unknown-nto-qnx8.0.0", "aarch64", "linux"));
    }

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| vars.iter().find(|(n, _)| *n == name).map(|(_, v)| v.to_string())
    }

    #[test]
    fn environment() {
        let o = CompileOptions::from_env_with(env(&[])).unwrap();
        assert_eq!(settings(&o), (None, None, None));
        let o = CompileOptions::from_env_with(env(&[("CUTE_RS_COMPILER_OPT", "--opt-level 1 preserve-line-info")]))
            .unwrap();
        assert_eq!(settings(&o), (None, Some(1), Some(true)));
        let o = CompileOptions::from_env_with(env(&[("CUTE_RS_ARCH", "sm_90a"), ("CUTE_RS_LINEINFO", "1")])).unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm90a), None, Some(true)));
        // As in CuTe DSL, the option string's target beats CUTE_RS_ARCH, and CUTE_RS_LINEINFO
        // can only switch line information on.
        let o = CompileOptions::from_env_with(env(&[
            ("CUTE_RS_COMPILER_OPT", "cubin-chip=sm_80 preserve-line-info"),
            ("CUTE_RS_ARCH", " sm_86 "),
            ("CUTE_RS_LINEINFO", "off"),
        ]))
        .unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm80), None, Some(true)));
        let o = CompileOptions::from_env_with(env(&[
            ("CUTE_RS_COMPILER_OPT", "preserve-line-info=0 --opt-level 2"),
            ("CUTE_RS_ARCH", "sm_86"),
            ("CUTE_RS_LINEINFO", "yes"),
        ]))
        .unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm86), Some(2), Some(true)));
        let o = CompileOptions::from_env_with(env(&[
            ("CUTE_RS_COMPILER_OPT", "preserve-line-info=0"),
            ("CUTE_RS_LINEINFO", "0"),
        ]))
        .unwrap();
        assert_eq!(settings(&o), (None, None, Some(false)));
        let o = CompileOptions::from_env_with(env(&[("CUTE_RS_LINEINFO", "false")])).unwrap();
        assert_eq!(settings(&o), (None, None, None));
        // A target cleared in the option string leaves CUTE_RS_ARCH's.
        let o =
            CompileOptions::from_env_with(env(&[("CUTE_RS_COMPILER_OPT", "--gpu-arch ''"), ("CUTE_RS_ARCH", "sm_89")]))
                .unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm89), None, None));
        // Empty variables are ignored.
        let o = CompileOptions::from_env_with(env(&[
            ("CUTE_RS_COMPILER_OPT", " "),
            ("CUTE_RS_ARCH", ""),
            ("CUTE_RS_LINEINFO", ""),
            ("CUTE_RS_KEEP", ""),
        ]))
        .unwrap();
        assert_eq!(settings(&o), (None, None, None));
        // CuTe DSL's own variables are not read.
        let o = CompileOptions::from_env_with(env(&[("CUTE_DSL_ARCH", "sm_90a"), ("CUTE_DSL_LINEINFO", "1")])).unwrap();
        assert_eq!(settings(&o), (None, None, None));
        assert!(CompileOptions::from_env_with(env(&[("CUTE_RS_ENABLE_PYIR", "1")])).is_ok());
    }

    #[test]
    fn environment_errors_name_the_variable() {
        for (vars, want) in [
            (vec![("CUTE_RS_COMPILER_OPT", "--opt-level 5")], "CUTE_RS_COMPILER_OPT: `--opt-level 5`"),
            (vec![("CUTE_RS_ARCH", "sm_70")], "CUTE_RS_ARCH: "),
            (vec![("CUTE_RS_LINEINFO", "2")], "CUTE_RS_LINEINFO=\"2\" is not a boolean"),
            (vec![("CUTE_RS_KEEP", "ptx,asm")], "CUTE_RS_KEEP: `asm` is not something to keep"),
            (vec![("CUTE_RS_PRINT_IR", "2")], "CUTE_RS_PRINT_IR=\"2\" is not a boolean"),
            (vec![("CUTE_RS_DEBUG", "maybe")], "CUTE_RS_DEBUG=\"maybe\" is not a boolean"),
            (vec![("CUTE_RS_ENABLE_ASSERTIONS", "maybe")], "CUTE_RS_ENABLE_ASSERTIONS=\"maybe\" is not a boolean"),
            (vec![("CUTE_RS_ENABLE_TVM_FFI", "maybe")], "CUTE_RS_ENABLE_TVM_FFI=\"maybe\" is not a boolean"),
            (vec![("CUTE_RS_COMPILER_OPT", "iket")], "CUTE_RS_COMPILER_OPT: `iket`"),
            (vec![("CUTE_RS_KEEP_PTX", "ptx")], "CUTE_RS_KEEP_PTX=\"ptx\" is not a boolean"),
            // A set variable is checked even when another takes precedence over it.
            (vec![("CUTE_RS_COMPILER_OPT", "cubin-chip=sm_80"), ("CUTE_RS_ARCH", "sm_1")], "CUTE_RS_ARCH: "),
            (vec![("CUTE_RS_COMPILER_OPT", "preserve-line-info"), ("CUTE_RS_LINEINFO", "2")], "CUTE_RS_LINEINFO=\"2\""),
        ] {
            let e = CompileOptions::from_env_with(env(&vars)).unwrap_err().to_string();
            assert!(e.contains(want), "{vars:?}: {e}");
            if want.contains("m6-") {
                assert!(e.contains("m6-") && e.contains("not available yet"), "{vars:?}: {e}");
            }
        }
        // The artifact variables switch things on.
        let o = CompileOptions::from_env_with(env(&[
            ("CUTE_RS_KEEP", "ptx, sass"),
            ("CUTE_RS_DUMP_DIR", "/tmp/k"),
            ("CUTE_RS_PRINT_IR", "1"),
            ("CUTE_RS_KEEP_CUBIN", "on"),
            ("CUTE_RS_KEEP_IR", "yes"),
        ]))
        .unwrap();
        let kept: Vec<Artifact> = Artifact::ALL.into_iter().filter(|a| o.keeps(*a)).collect();
        assert_eq!(kept, [Artifact::Mlir, Artifact::Ptx, Artifact::Cubin, Artifact::Sass]);
        assert_eq!((o.get_dump_dir(), o.get_print_ir()), (Some(Path::new("/tmp/k")), Some(true)));
        // Debug mode: line information and everything kept.
        let o = CompileOptions::from_env_with(env(&[("CUTE_RS_DEBUG", "1")])).unwrap();
        assert_eq!(o.get_generate_line_info(), Some(true));
        assert!(Artifact::ALL.into_iter().all(|a| o.keeps(a)));
        let o = CompileOptions::new().debug(true);
        assert!(o.keeps(Artifact::Llvm) && o.get_generate_line_info() == Some(true));
        // Switched off, they ask for nothing.
        let off = [
            ("CUTE_RS_DEBUG", "0"),
            ("CUTE_RS_ENABLE_ASSERTIONS", "false"),
            ("CUTE_RS_KEEP_IR", "no"),
            ("CUTE_RS_KEEP_PTX", "0"),
            ("CUTE_RS_KEEP_CUBIN", "off"),
            ("CUTE_RS_KEEP_SASS", ""),
        ];
        assert_eq!(settings(&CompileOptions::from_env_with(env(&off)).unwrap()), (None, None, None));
    }

    /// Explicit options win over the environment, a deliberate difference from CuTe DSL (which
    /// applies CUTE_DSL_COMPILER_OPT on top of them and lets CUTE_DSL_LINEINFO=1 override an
    /// explicit GenerateLineInfo(False)).
    #[test]
    fn explicit_options_override_the_environment() {
        let vars = [("CUTE_RS_COMPILER_OPT", "opt-level=0 preserve-line-info"), ("CUTE_RS_ARCH", "sm_80")];
        let o = CompileOptions::new().opt_level(2).generate_line_info(false).with_env_from(env(&vars)).unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm80), Some(2), Some(false)));
        let o = CompileOptions::parse("--gpu-arch sm_90").unwrap().with_env_from(env(&vars)).unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm90), Some(0), Some(true)));
        let o = CompileOptions::new().with_env_from(env(&vars)).unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm80), Some(0), Some(true)));
        let lineinfo = [("CUTE_RS_LINEINFO", "1"), ("CUTE_RS_COMPILER_OPT", "cubin-chip=sm_86")];
        let o = CompileOptions::new().generate_line_info(false).gpu_arch(Arch::Sm90a).with_env_from(env(&lineinfo));
        assert_eq!(settings(&o.unwrap()), (Some(Arch::Sm90a), None, Some(false)));
        let o = CompileOptions::new().with_env_from(env(&lineinfo)).unwrap();
        assert_eq!(settings(&o), (Some(Arch::Sm86), None, Some(true)));
        // An invalid environment is reported even when explicit options cover it.
        assert!(CompileOptions::new().opt_level(1).with_env_from(env(&[("CUTE_RS_ARCH", "sm_1")])).is_err());
    }

    #[test]
    fn hooks_accumulate() {
        let o = CompileOptions::new().stage_hook(|_| Ok(())).or(CompileOptions::new().stage_hook(|_| Ok(())));
        assert_eq!(o.stage_hook_count(), 2);
        assert!(format!("{o:?}").contains("stage_hooks: 2"));
    }
}

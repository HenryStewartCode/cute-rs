//! Ahead-of-time export (CuTe DSL's `export_to_c` and `cute.runtime.load_module`).
//!
//! [`CompiledKernel::export_c`] writes a compiled kernel as files that run without cute-rs,
//! `mlir-opt` or the kernel's Rust code:
//!
//! * `<name>.h`: version macros, the argument struct `<prefix>_Args_t` (one field per kernel
//!   parameter, in order), the launch configuration `<prefix>_Launch_Config_t`, the module
//!   struct `<prefix>_Kernel_Module_t` and the functions `<prefix>_Kernel_Module_Load`,
//!   `<prefix>_Kernel_Module_Unload` and `<prefix>_Kernel_Launch`;
//! * `<name>.c`: the PTX (and the cubin, when there is one) as byte arrays, and those functions,
//!   written against the CUDA driver API (`cuModuleLoadData`, `cuModuleGetFunction`,
//!   `cuLaunchKernelEx`), with the launch checks of cute-rs's own launcher;
//! * `<name>.o`, when a C compiler and `cuda.h` are found;
//! * `<name>.sig`, the manifest: the export format, the cute-rs version, the target, the PTX
//!   version and the [`KernelSignature`] as text ([`KernelSignature::to_text`]), with
//!   `<name>.ptx` and `<name>.cubin` beside it, which [`ExportedKernel::read`] and
//!   `CudaContext::load_exported` (feature `cuda`) load back.
//!
//! CuTe DSL's export uses the CUDA runtime API (`cudaError_t`, `cudaLibrary_t`, loading onto
//! every device). This one uses the driver API, as cute-rs's launcher does: functions return
//! `CUresult`, the program links `-lcuda` only, and a module is loaded into the current context
//! (the first device's primary context when none is current), so a multi-GPU program loads one
//! module per context. A failed check returns `CUDA_ERROR_INVALID_VALUE` and leaves a message in
//! the module's `error` field.

use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend::{Arch, CompiledKernel, driver_ptx_ceiling};
use crate::ir::DType;
use crate::signature::{KernelSignature, ParamKind};

/// The version of the export format (the manifest and the generated C), which loading checks.
pub const EXPORT_FORMAT_VERSION: u32 = 1;

// Several kernels in one module: export format 2.
mod module;
pub use module::{ExportedModule, MODULE_EXPORT_FORMAT_VERSION};

/// The first line of a manifest, before the format version.
const MAGIC: &str = "cute-rs-export";

/// Why an export could not be written or read, or an exported kernel cannot run here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AotError(String);

impl AotError {
    fn new(message: impl Into<String>) -> AotError {
        AotError(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AotError {}

/// What an export is stamped with, which loading checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportStamp {
    /// [`EXPORT_FORMAT_VERSION`] of the cute-rs that wrote it.
    pub format: u32,
    /// The version of the cute-rs that wrote it ([`VERSION`](crate::VERSION)).
    pub crate_version: String,
    /// The architecture the cubin's SASS is for, if there is a cubin.
    pub cubin_target: Option<Arch>,
}

/// Which image of an exported kernel a device loads ([`ExportedKernel::choose_image`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageChoice {
    Cubin,
    Ptx,
}

/// An exported kernel: its stamp, signature (with the target and PTX version) and images.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportedKernel {
    pub stamp: ExportStamp,
    pub signature: KernelSignature,
    pub ptx: Option<String>,
    pub cubin: Option<Vec<u8>>,
}

/// Options of [`CompiledKernel::export_c_with`].
#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Build `<name>.o` when a C compiler and `cuda.h` are found (default on).
    pub compile_object: bool,
    /// The C compiler: default `$CC`, else `cc`.
    pub cc: Option<PathBuf>,
    /// The directory holding `cuda.h`: default the first of `$CUDA_HOME/include`,
    /// `$CUDA_PATH/include` and `/usr/local/cuda/include` that has one.
    pub cuda_include: Option<PathBuf>,
}

impl Default for ExportOptions {
    fn default() -> ExportOptions {
        ExportOptions { compile_object: true, cc: None, cuda_include: None }
    }
}

/// The files an export wrote, and why any was left out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportReport {
    pub header: PathBuf,
    pub source: PathBuf,
    pub manifest: PathBuf,
    pub ptx: PathBuf,
    pub cubin: Option<PathBuf>,
    pub object: Option<PathBuf>,
    /// Why something was not written (no C compiler, no `cuda.h`, no cubin).
    pub notes: Vec<String>,
}

/// Whether a device of compute capability `cc` runs PTX compiled for `target`: an `a` target
/// on its own chip, an `f` target on its family, a baseline one on it and newer chips.
pub fn ptx_runs_on(target: Arch, cc: (u32, u32)) -> bool {
    match target.suffix() {
        "a" => cc == target.version(),
        "f" => cc.0 == target.major() && cc.1 >= target.minor(),
        _ => cc >= target.version(),
    }
}

/// Whether a device of compute capability `cc` runs a cubin made for `target`
/// ([`Arch::can_run_binary_built_for`] by compute capability).
pub fn binary_runs_on(target: Arch, cc: (u32, u32)) -> bool {
    cc.0 == target.major() && if target.suffix() == "a" { cc.1 == target.minor() } else { cc.1 >= target.minor() }
}

/// The oldest CUDA driver version (`1000 * major + 10 * minor`) whose JIT accepts PTX ISA
/// `ptx` ([`driver_ptx_ceiling`]), if a known one does.
pub fn min_driver_for_ptx(ptx: u32) -> Option<u32> {
    let mut candidates =
        std::iter::once(10_020).chain((11..=19).flat_map(|major| (0..=9).map(move |m| major * 1000 + m * 10)));
    candidates.find(|&d| driver_ptx_ceiling(d) >= ptx)
}

impl ExportedKernel {
    /// The export of `kernel`: its signature, PTX and cubin, stamped with this cute-rs.
    pub fn from_compiled(kernel: &CompiledKernel) -> ExportedKernel {
        let mut signature = kernel.signature().clone();
        signature.target = Some(kernel.arch());
        signature.ptx_version = Some(kernel.ptx_version());
        // The export carries the shared-memory sizes, not each buffer's start.
        signature.smem.starts.clear();
        let cubin = kernel.cubin().map(<[u8]>::to_vec);
        ExportedKernel {
            stamp: ExportStamp {
                format: EXPORT_FORMAT_VERSION,
                crate_version: crate::VERSION.to_string(),
                cubin_target: cubin.as_ref().and(kernel.cubin_target().or(Some(kernel.arch()))),
            },
            signature,
            ptx: Some(kernel.ptx().to_string()),
            cubin,
        }
    }

    /// An export from parts, stamped with this cute-rs: `signature` must have its target and
    /// PTX version (as [`CompiledKernel::signature`] has them). The shared buffers' starts are
    /// dropped, as the manifest does not carry them.
    pub fn new(
        mut signature: KernelSignature,
        ptx: Option<String>,
        cubin: Option<(Vec<u8>, Arch)>,
    ) -> Result<ExportedKernel, AotError> {
        signature.smem.starts.clear();
        let (cubin, cubin_target) = cubin.map_or((None, None), |(c, t)| (Some(c), Some(t)));
        let k = ExportedKernel {
            stamp: ExportStamp {
                format: EXPORT_FORMAT_VERSION,
                crate_version: crate::VERSION.to_string(),
                cubin_target,
            },
            signature,
            ptx,
            cubin,
        };
        k.validate()?;
        Ok(k)
    }

    fn validate(&self) -> Result<(), AotError> {
        let name = &self.signature.name;
        if self.signature.target.is_none() || self.signature.ptx_version.is_none() {
            return Err(AotError::new(format!(
                "exported kernel `{name}` has no target or PTX version in its signature"
            )));
        }
        if self.ptx.is_none() && self.cubin.is_none() {
            return Err(AotError::new(format!("exported kernel `{name}` has neither PTX nor a cubin")));
        }
        if self.cubin.is_some() != self.stamp.cubin_target.is_some() {
            return Err(AotError::new(format!(
                "exported kernel `{name}`: a cubin needs its target (`cubin_target`), and only a cubin has one"
            )));
        }
        Ok(())
    }

    /// The kernel's target.
    pub fn target(&self) -> Arch {
        self.signature.target.expect("validated")
    }

    /// The kernel's PTX ISA version (`10 * major + minor`).
    pub fn ptx_version(&self) -> u32 {
        self.signature.ptx_version.expect("validated")
    }

    /// The manifest text, naming the image files beside it.
    pub fn manifest(&self, ptx_file: Option<&str>, cubin_file: Option<&str>) -> String {
        let mut out = format!("{MAGIC} {}\ncrate {}\n", self.stamp.format, self.stamp.crate_version);
        if let Some(t) = self.stamp.cubin_target {
            out += &format!("cubin_target {}\n", t.chip());
        }
        if let Some(f) = ptx_file {
            out += &format!("image ptx {f}\n");
        }
        if let Some(f) = cubin_file {
            out += &format!("image cubin {f}\n");
        }
        out += "signature\n";
        out += &self.signature.to_text();
        out
    }

    /// Read a manifest: the stamp, the signature and the image files it names (PTX, cubin).
    /// Fails on another format version, a malformed line or a malformed signature.
    pub fn parse_manifest(
        text: &str,
    ) -> Result<(ExportStamp, KernelSignature, Option<String>, Option<String>), AotError> {
        let mut lines = text.lines().enumerate();
        let bad = |line: usize, m: String| AotError::new(format!("export manifest, line {}: {m}", line + 1));
        let format = match lines.next().map(|(_, l)| l.split_whitespace().collect::<Vec<_>>()) {
            Some(t) if t.len() == 2 && t[0] == MAGIC => {
                t[1].parse::<u32>().map_err(|_| bad(0, format!("bad format version `{}`", t[1])))?
            }
            _ => return Err(bad(0, format!("not a cute-rs export manifest (it starts with `{MAGIC} <version>`)"))),
        };
        if format == MODULE_EXPORT_FORMAT_VERSION {
            return Err(AotError::new(format!(
                "the export is format version 2, and this cute-rs ({}) reads it as a module of several kernels: \
                 read it with ExportedModule::read, or load it with CudaContext::load_exported_module",
                crate::VERSION
            )));
        }
        if format != EXPORT_FORMAT_VERSION {
            return Err(AotError::new(format!(
                "the export is format version {format}, and this cute-rs ({}) reads version {EXPORT_FORMAT_VERSION}: \
                 export the kernel again with this cute-rs",
                crate::VERSION
            )));
        }
        let (mut crate_version, mut cubin_target, mut ptx, mut cubin) = (None, None, None, None);
        let mut sig_start = None;
        for (i, line) in lines.by_ref() {
            let t: Vec<&str> = line.split_whitespace().collect();
            match t.as_slice() {
                [] => {}
                ["signature"] => {
                    sig_start = Some(i + 1);
                    break;
                }
                ["crate", v] if crate_version.is_none() => crate_version = Some(v.to_string()),
                ["cubin_target", c] if cubin_target.is_none() => {
                    cubin_target = Some(c.parse::<Arch>().map_err(|e| bad(i, e.to_string()))?)
                }
                ["image", _, f] if !is_file_name(f) => {
                    return Err(bad(i, format!("image file `{f}` is not a file name beside the manifest")));
                }
                ["image", "ptx", f] if ptx.is_none() => ptx = Some(f.to_string()),
                ["image", "cubin", f] if cubin.is_none() => cubin = Some(f.to_string()),
                _ => return Err(bad(i, format!("unexpected line `{line}`"))),
            }
        }
        let start = sig_start.ok_or_else(|| AotError::new("export manifest: no `signature` line"))?;
        let crate_version = crate_version.ok_or_else(|| AotError::new("export manifest: no `crate` line"))?;
        let rest: Vec<&str> = text.lines().skip(start).collect();
        let signature = KernelSignature::from_text(&rest.join("\n"))
            .map_err(|e| AotError::new(format!("export manifest, line {}: {}", e.line + start, e.message)))?;
        Ok((ExportStamp { format, crate_version, cubin_target }, signature, ptx, cubin))
    }

    /// An export from a manifest and its images (for instance `include_str!` and
    /// `include_bytes!` of an export's files, to deploy without the files).
    pub fn from_parts(manifest: &str, ptx: Option<&str>, cubin: Option<&[u8]>) -> Result<ExportedKernel, AotError> {
        let (stamp, signature, _, _) = ExportedKernel::parse_manifest(manifest)?;
        let k = ExportedKernel { stamp, signature, ptx: ptx.map(str::to_string), cubin: cubin.map(<[u8]>::to_vec) };
        k.validate()?;
        Ok(k)
    }

    /// Read an export from its manifest (`<name>.sig`) and the images it names, beside it.
    pub fn read(path: impl AsRef<Path>) -> Result<ExportedKernel, AotError> {
        let path = path.as_ref();
        let io = |p: &Path, e: std::io::Error| AotError::new(format!("cannot read {}: {e}", p.display()));
        let text = std::fs::read_to_string(path).map_err(|e| io(path, e))?;
        let (stamp, signature, ptx_file, cubin_file) = ExportedKernel::parse_manifest(&text)?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let ptx = match ptx_file {
            Some(f) => Some(std::fs::read_to_string(dir.join(&f)).map_err(|e| io(&dir.join(&f), e))?),
            None => None,
        };
        let cubin = match cubin_file {
            Some(f) => Some(std::fs::read(dir.join(&f)).map_err(|e| io(&dir.join(&f), e))?),
            None => None,
        };
        let k = ExportedKernel { stamp, signature, ptx, cubin };
        k.validate()?;
        Ok(k)
    }

    /// Which image a device of compute capability `cc` loads, whose driver's JIT accepts PTX up
    /// to `driver_ptx` (when known): the cubin when the device runs a binary built for its
    /// target, else the PTX when the device runs the kernel's target and the driver its PTX
    /// version. Fails, naming the reason, when neither does.
    pub fn choose_image(&self, cc: (u32, u32), driver_ptx: Option<u32>) -> Result<ImageChoice, AotError> {
        let name = &self.signature.name;
        let target = self.target();
        if !ptx_runs_on(target, cc) {
            return Err(AotError::new(format!(
                "exported kernel `{name}` was compiled for {}, which a device of compute capability {}.{} (sm_{}{}) \
                 cannot run",
                target.chip(),
                cc.0,
                cc.1,
                cc.0,
                cc.1
            )));
        }
        if let (Some(_), Some(t)) = (&self.cubin, self.stamp.cubin_target)
            && binary_runs_on(t, cc)
        {
            return Ok(ImageChoice::Cubin);
        }
        if self.ptx.is_none() {
            let t = self.stamp.cubin_target.map_or("?", Arch::chip);
            return Err(AotError::new(format!(
                "exported kernel `{name}` has only a cubin for {t}, which sm_{}{} cannot run as a binary, and no PTX",
                cc.0, cc.1
            )));
        }
        let v = self.ptx_version();
        if let Some(ceiling) = driver_ptx
            && v > ceiling
        {
            return Err(AotError::new(format!(
                "exported kernel `{name}` is PTX ISA {}.{}, but the CUDA driver loads PTX ISA {}.{} at most{}",
                v / 10,
                v % 10,
                ceiling / 10,
                ceiling % 10,
                min_driver_for_ptx(v).map_or(String::new(), |d| format!(
                    " (PTX ISA {}.{} needs CUDA {}.{})",
                    v / 10,
                    v % 10,
                    d / 1000,
                    d % 1000 / 10
                ))
            )));
        }
        Ok(ImageChoice::Ptx)
    }

    /// The C header of the export (`<name>.h`), with symbols starting `prefix` (`name` when
    /// empty).
    pub fn c_header(&self, name: &str, prefix: &str) -> String {
        c_header(self, name, &c_ident(if prefix.is_empty() { name } else { prefix }))
    }

    /// The C source of the export (`<name>.c`), which includes `<name>.h`.
    pub fn c_source(&self, name: &str, prefix: &str) -> String {
        c_source(self, name, &c_ident(if prefix.is_empty() { name } else { prefix }))
    }

    /// Write the export to `dir` (created if needed) as `<name>.h`, `<name>.c`, `<name>.sig`,
    /// `<name>.ptx`, `<name>.cubin` (when there is a cubin) and `<name>.o` (when a C compiler
    /// and `cuda.h` are found and `options.compile_object` is on). An empty `prefix` is `name`.
    pub fn write_c(
        &self,
        dir: impl AsRef<Path>,
        name: &str,
        prefix: &str,
        options: &ExportOptions,
    ) -> Result<ExportReport, AotError> {
        let dir = dir.as_ref();
        if !is_file_name(name) {
            return Err(AotError::new(format!(
                "`{name}` is not a file name for an export: use letters, digits, `_`, `-` and `.`"
            )));
        }
        if let Some(e) = self.signature.text_error() {
            return Err(AotError::new(e));
        }
        if let Some(e) = unexportable_ptx(self.ptx.as_deref(), &self.signature.name) {
            return Err(AotError::new(e));
        }
        let prefix = c_ident(if prefix.is_empty() { name } else { prefix });
        let io = |p: &Path, e: std::io::Error| AotError::new(format!("cannot write {}: {e}", p.display()));
        std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
        let write = |file: &str, bytes: &[u8]| -> Result<PathBuf, AotError> {
            let p = dir.join(file);
            std::fs::write(&p, bytes).map_err(|e| io(&p, e))?;
            Ok(p)
        };
        let mut notes = vec![];
        let ptx_file = format!("{name}.ptx");
        let cubin_file = format!("{name}.cubin");
        let ptx = write(&ptx_file, self.ptx.as_deref().unwrap_or("").as_bytes())?;
        let cubin = match &self.cubin {
            Some(c) => Some(write(&cubin_file, c)?),
            None => {
                notes.push("no cubin (the kernel was compiled without ptxas): the export carries the PTX only".into());
                None
            }
        };
        let manifest = write(
            &format!("{name}.sig"),
            self.manifest(self.ptx.as_ref().map(|_| ptx_file.as_str()), cubin.as_ref().map(|_| cubin_file.as_str()))
                .as_bytes(),
        )?;
        let header = write(&format!("{name}.h"), c_header(self, name, &prefix).as_bytes())?;
        let source = write(&format!("{name}.c"), c_source(self, name, &prefix).as_bytes())?;
        let object = if options.compile_object { compile_object(dir, name, options, &mut notes)? } else { None };
        Ok(ExportReport { header, source, manifest, ptx, cubin, object, notes })
    }
}

/// Whether `name` is a plain file name of an export: letters, digits, `_`, `-` and `.`, not
/// starting with `.` (so no directories, `..` or hidden files).
fn is_file_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// The directory with `cuda.h`, from the options or the usual places.
fn find_cuda_include(options: &ExportOptions) -> Option<PathBuf> {
    match &options.cuda_include {
        Some(d) => Some(d.clone()).filter(|d| d.join("cuda.h").is_file()),
        None => cuda_include_dir(),
    }
}

/// The first of `$CUDA_HOME/include`, `$CUDA_PATH/include` and `/usr/local/cuda/include` that
/// holds `cuda.h`: what an export's object file, and a C program using it, compile against.
pub fn cuda_include_dir() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(|v| std::env::var_os(v).filter(|s| !s.is_empty()))
        .map(|h| Path::new(&h).join("include"))
        .collect();
    dirs.push(PathBuf::from("/usr/local/cuda/include"));
    dirs.into_iter().find(|d| d.join("cuda.h").is_file())
}

/// Build `<name>.o` from `<name>.c`, or say why not in `notes`. A compiler that fails on the
/// generated code is an error.
fn compile_object(
    dir: &Path,
    name: &str,
    options: &ExportOptions,
    notes: &mut Vec<String>,
) -> Result<Option<PathBuf>, AotError> {
    let Some(include) = find_cuda_include(options) else {
        let looked = match &options.cuda_include {
            Some(d) => d.display().to_string(),
            None => "$CUDA_HOME/include, $CUDA_PATH/include and /usr/local/cuda/include".to_string(),
        };
        notes.push(format!(
            "{name}.o was not built: no cuda.h (looked in {looked}); build {name}.c with the CUDA toolkit's headers"
        ));
        return Ok(None);
    };
    let cc = options
        .cc
        .clone()
        .or_else(|| std::env::var_os("CC").filter(|s| !s.is_empty()).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("cc"));
    let object = dir.join(format!("{name}.o"));
    let out = Command::new(&cc)
        .args(["-std=c99", "-O2", "-fPIC", "-c"])
        .arg("-I")
        .arg(dir)
        .arg("-I")
        .arg(&include)
        .arg(dir.join(format!("{name}.c")))
        .arg("-o")
        .arg(&object)
        .output();
    match out {
        Err(e) => {
            notes.push(format!("{name}.o was not built: cannot run the C compiler `{}`: {e}", cc.display()));
            Ok(None)
        }
        Ok(o) if !o.status.success() => Err(AotError::new(format!(
            "`{}` could not compile {name}.c:\n{}",
            cc.display(),
            String::from_utf8_lossy(&o.stderr)
        ))),
        Ok(_) => Ok(Some(object)),
    }
}

/// `s` as a C identifier: other characters become `_`, and a leading digit gets a `_`.
pub fn c_ident(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

/// C and C++ keywords (the header is included from both) that cannot be field names.
#[rustfmt::skip]
const C_KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float",
    "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "bool", "true", "false",
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
    "_Thread_local", "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor", "catch", "char8_t", "char16_t",
    "char32_t", "class", "compl", "concept", "consteval", "constexpr", "constinit", "const_cast", "co_await",
    "co_return", "co_yield", "decltype", "delete", "dynamic_cast", "explicit", "export", "friend", "mutable",
    "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected",
    "public", "reinterpret_cast", "requires", "static_assert", "static_cast", "template", "this", "thread_local",
    "throw", "try", "typeid", "typename", "using", "virtual", "wchar_t", "xor", "xor_eq", "NULL",
];

/// The C field name of each parameter: a C identifier, unique, not a keyword.
fn field_names(sig: &KernelSignature) -> Vec<String> {
    let mut names: Vec<String> = vec![];
    for p in &sig.params {
        let mut n = c_ident(&p.name);
        if C_KEYWORDS.contains(&n.as_str()) {
            n.push('_');
        }
        while names.contains(&n) {
            n.push('_');
        }
        names.push(n);
    }
    names
}

/// The C type of a scalar of `t`, and the array length when it is a byte array.
fn c_scalar_type(t: DType) -> (&'static str, Option<usize>) {
    match t {
        DType::F32 | DType::TF32 => ("float", None),
        DType::F64 => ("double", None),
        _ => match (t.bytes().max(1), t.is_signed() && !t.is_float()) {
            (1, true) => ("int8_t", None),
            (2, true) => ("int16_t", None),
            (4, true) => ("int32_t", None),
            (8, true) => ("int64_t", None),
            (1, false) => ("uint8_t", None),
            (2, false) => ("uint16_t", None),
            (4, false) => ("uint32_t", None),
            (8, false) => ("uint64_t", None),
            (n, _) => ("uint8_t", Some(n)),
        },
    }
}

/// `s` safe inside a C block comment: no `*/` (nor `/*`, which -Wcomment flags), printable
/// ASCII only.
fn c_comment(s: &str) -> String {
    let s: String = s.chars().map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '?' }).collect();
    s.replace("*/", "* /").replace("/*", "/ *")
}

/// The upper-case form of `prefix`, for macros.
fn upper(prefix: &str) -> String {
    prefix.to_ascii_uppercase()
}

/// A C string literal of `s` (printable ASCII; anything else escaped).
fn c_string(s: &str) -> String {
    let mut out = String::from("\"");
    for b in s.bytes() {
        match b {
            b'"' | b'\\' | b'?' => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7e => out.push(b as char),
            _ => out += &format!("\\{b:03o}"),
        }
    }
    out.push('"');
    out
}

fn c_header(k: &ExportedKernel, name: &str, p: &str) -> String {
    let sig = &k.signature;
    let up = upper(p);
    let target = k.target();
    let v = k.ptx_version();
    let fields = field_names(sig);
    let mut h = String::new();
    let _ = writeln!(
        h,
        "/* {name}.h: the kernel `{}`, exported by cute-rs {} (export format {}) for {}, PTX ISA {}.{}.\n *\n * \
         Generated code: do not edit. Build {name}.c with the CUDA toolkit's headers and link with -lcuda:\n *\n \
         *   {p}_Kernel_Module_t module;\n *   {p}_Kernel_Module_Load(&module);   (a CUresult; module.error says \
         why on failure)\n *   {p}_Args_t args = {{ ... }};          (one field per kernel parameter)\n *   \
         {p}_Launch_Config_t config = {{ {{grid}}, {{block}}, {{1, 1, 1}}, 0, stream }};\n *   \
         {p}_Kernel_Launch(&module, &config, &args);\n *   {p}_Kernel_Module_Unload(&module);\n */",
        c_comment(&sig.name),
        k.stamp.crate_version,
        k.stamp.format,
        target.chip(),
        v / 10,
        v % 10
    );
    let _ = writeln!(h, "#ifndef {up}_CUTE_RS_EXPORT_H\n#define {up}_CUTE_RS_EXPORT_H\n");
    h += "#include <stddef.h>\n#include <stdint.h>\n#include <cuda.h>\n\n";
    let _ = writeln!(h, "#define {up}_CUTE_RS_EXPORT_FORMAT {}", k.stamp.format);
    let _ = writeln!(h, "#define {up}_CUTE_RS_VERSION {}", c_string(&k.stamp.crate_version));
    let _ = writeln!(h, "#define {up}_KERNEL_NAME {}", c_string(&sig.name));
    let _ = writeln!(h, "#define {up}_TARGET {}", c_string(target.chip()));
    let _ = writeln!(h, "#define {up}_PTX_VERSION {v}");
    if let Some(t) = k.stamp.cubin_target {
        let _ = writeln!(h, "#define {up}_CUBIN_TARGET {}", c_string(t.chip()));
    }
    if let Some([x, y, z]) = sig.block_dim {
        let _ = writeln!(h, "#define {up}_BLOCK_DIM_X {x}\n#define {up}_BLOCK_DIM_Y {y}\n#define {up}_BLOCK_DIM_Z {z}");
    }
    let _ = writeln!(h, "#define {up}_NUM_PARAMS {}", sig.params.len());
    h += "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
    h += &c_args_struct(sig, &fields, "The kernel's parameters, in order.", &format!("{p}_Args_t"));
    h += &c_launch_config(p);
    let _ = writeln!(
        h,
        "/* The loaded kernel. `error` holds a message when a function returns an error. */\ntypedef struct {{\n    \
         CUcontext context;\n    CUmodule module;\n    CUfunction function;\n    CUdevice device;\n    \
         int retained_primary;\n    int from_cubin;\n    char error[256];\n}} {p}_Kernel_Module_t;\n"
    );
    let _ = writeln!(
        h,
        "/* Load the kernel into the current context (the first device's primary context when none is current):\n \
         * the cubin when the device runs it as a binary, else the PTX, which the driver's JIT compiles. */\n\
         CUresult {p}_Kernel_Module_Load({p}_Kernel_Module_t *module);\n\n\
         /* Unload the module (and release the primary context Load retained). */\n\
         CUresult {p}_Kernel_Module_Unload({p}_Kernel_Module_t *module);\n\n\
         /* Check the launch against the kernel (block size, clusters, pointer alignment and size, divisibility)\n \
         * and queue it (cuLaunchKernelEx), from a thread whose current context is the module's. Returns\n \
         * CUDA_ERROR_INVALID_VALUE, with module->error set, when a check fails. */\n\
         CUresult {p}_Kernel_Launch({p}_Kernel_Module_t *module, const {p}_Launch_Config_t *config, const \
         {p}_Args_t *args);\n"
    );
    let _ = writeln!(h, "#ifdef __cplusplus\n}}\n#endif\n\n#endif /* {up}_CUTE_RS_EXPORT_H */");
    h
}

/// The C struct of a kernel's arguments, one field per parameter.
fn c_args_struct(sig: &KernelSignature, fields: &[String], comment: &str, type_name: &str) -> String {
    let mut h = format!("/* {comment} */\ntypedef struct {{\n");
    if sig.params.is_empty() {
        h += "    char unused; /* the kernel takes no parameters */\n";
    }
    for (param, field) in sig.params.iter().zip(fields) {
        let (decl, what) = match &param.kind {
            ParamKind::Pointer { elem, min_len, align } => (
                format!("CUdeviceptr {field};"),
                format!(
                    "{} *{}, {align}-byte aligned",
                    elem.cute_name(),
                    min_len.map_or(String::new(), |n| format!(", at least {n} elements"))
                ),
            ),
            ParamKind::Scalar { dtype, multiple_of } => {
                let (t, n) = c_scalar_type(*dtype);
                let decl = match n {
                    Some(n) => format!("{t} {field}[{n}];"),
                    None => format!("{t} {field};"),
                };
                let div = if *multiple_of > 1 { format!(", a multiple of {multiple_of}") } else { String::new() };
                (decl, format!("{}{div}", dtype.cute_name()))
            }
            ParamKind::ByValue { size, align, .. } => {
                (format!("uint8_t {field}[{}];", size.max(&1)), format!("{size} bytes by value, {align}-byte aligned"))
            }
        };
        let _ = writeln!(h, "    {decl} /* `{}`: {what} */", c_comment(&param.name));
    }
    let _ = writeln!(h, "}} {type_name};\n");
    h
}

/// The C struct of a launch configuration.
fn c_launch_config(p: &str) -> String {
    let mut h = String::new();
    let _ = writeln!(
        h,
        "/* A launch: grid and block sizes, the cluster shape (entries of 0 mean 1; clusters need SM90 and CUDA 12),\n \
         * the bytes of the kernel's dynamic shared-memory region, and the stream (NULL for the legacy stream). */\n\
         typedef struct {{\n    unsigned int grid[3];\n    unsigned int block[3];\n    unsigned int cluster[3];\n    \
         unsigned int dynamic_smem_bytes;\n    CUstream stream;\n}} {p}_Launch_Config_t;\n"
    );
    h
}

/// `bytes` as the lines of a C array initializer, 16 bytes a line.
fn c_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for chunk in bytes.chunks(16) {
        out += "    ";
        let items: Vec<String> = chunk.iter().map(|b| format!("0x{b:02x}")).collect();
        out += &items.join(", ");
        out += ",\n";
    }
    out
}

/// The C condition that a device of compute capability `major.minor` runs `target`'s code: as
/// PTX (`binary` false) or as a binary.
fn c_runs_on(target: Arch, binary: bool) -> String {
    let (ma, mi) = target.version();
    match (target.suffix(), binary) {
        ("a", _) => format!("major == {ma} && minor == {mi}"),
        ("f", _) | (_, true) => format!("major == {ma} && minor >= {mi}"),
        _ => format!("major > {ma} || (major == {ma} && minor >= {mi})"),
    }
}

fn c_source(k: &ExportedKernel, name: &str, p: &str) -> String {
    let sig = &k.signature;
    let up = upper(p);
    let target = k.target();
    let fields = field_names(sig);
    let mut c = String::new();
    let _ = writeln!(
        c,
        "/* {name}.c: the kernel `{}`, exported by cute-rs {} for {}. Generated code: do not edit. */\n",
        c_comment(&sig.name),
        k.stamp.crate_version,
        target.chip()
    );
    let image = CImage::of_kernel(k);
    c += &c_images(&image, name, &up, p);
    let names = CModuleNames {
        load: format!("{p}_Kernel_Module_Load"),
        unload: format!("{p}_Kernel_Module_Unload"),
        module_t: format!("{p}_Kernel_Module_t"),
        what: "kernel",
        name_macro: format!("{up}_KERNEL_NAME"),
    };
    let unload = &names.unload;
    let mut lookup = String::new();
    let _ = writeln!(
        lookup,
        "    r = cuModuleGetFunction(&m->function, m->module, {up}_KERNEL_NAME);\n    if (r != CUDA_SUCCESS) {{ {unload}(m); \
         return {up}_FAIL(m, r, \"cuModuleGetFunction failed (%d) for `%s`\", (int)r, {up}_KERNEL_NAME); }}\n    m->error[0] = 0;\n    \
         return CUDA_SUCCESS;\n}}\n"
    );
    c += &c_load(&image, p, &up, &names, &lookup);
    c += &c_unload(&names, "m->function = NULL;");
    c += &c_launch(
        sig,
        &fields,
        p,
        &up,
        &format!("{p}_Kernel_Launch"),
        &names.module_t,
        &format!("{p}_Args_t"),
        "m->function",
    );
    c
}

/// What a C export embeds and checks: the stamp, the images, the target and PTX version.
struct CImage<'a> {
    stamp: &'a ExportStamp,
    ptx: Option<&'a str>,
    cubin: Option<&'a [u8]>,
    target: Arch,
    v: u32,
}

impl CImage<'_> {
    fn of_kernel(k: &ExportedKernel) -> CImage<'_> {
        CImage {
            stamp: &k.stamp,
            ptx: k.ptx.as_deref(),
            cubin: k.cubin.as_deref(),
            target: k.target(),
            v: k.ptx_version(),
        }
    }
}

/// The names of an export's module-level C functions and struct, and what its messages call it.
struct CModuleNames {
    load: String,
    unload: String,
    module_t: String,
    /// `kernel` or `module`.
    what: &'static str,
    /// The macro of the kernel's or module's name.
    name_macro: String,
}

/// The includes, macros and image arrays of a C export's source.
fn c_images(k: &CImage<'_>, name: &str, up: &str, p: &str) -> String {
    let mut c = String::new();
    let _ = writeln!(c, "#include \"{name}.h\"\n\n#include <stdio.h>\n#include <string.h>\n");
    let _ = writeln!(
        c,
        "#if defined(__GNUC__) || defined(__clang__)\n#define {up}_ALIGNED __attribute__((aligned(16)))\n#else\n\
         #define {up}_ALIGNED\n#endif\n"
    );
    let _ = writeln!(c, "#define {up}_FAIL(m, r, ...) (snprintf((m)->error, sizeof((m)->error), __VA_ARGS__), (r))\n");
    if let Some(ptx) = k.ptx {
        let mut bytes = ptx.as_bytes().to_vec();
        bytes.push(0);
        let _ = write!(
            c,
            "/* The PTX, NUL-terminated. */\nstatic const unsigned char {p}_ptx[] {up}_ALIGNED = {{\n{}}};\n\n",
            c_bytes(&bytes)
        );
    }
    if let (Some(cubin), Some(t)) = (k.cubin, k.stamp.cubin_target) {
        let _ = write!(
            c,
            "/* The cubin, for {}. */\nstatic const unsigned char {p}_cubin[] {up}_ALIGNED = {{\n{}}};\n\n",
            t.chip(),
            c_bytes(cubin)
        );
    }
    c
}

/// The load function of a C export, ending with `lookup` (the `cuModuleGetFunction` calls).
fn c_load(k: &CImage<'_>, p: &str, up: &str, names: &CModuleNames, lookup: &str) -> String {
    let CModuleNames { load, unload, module_t, what, name_macro } = names;
    let (target, v) = (k.target, k.v);
    let mut c = String::new();
    let _ = writeln!(c, "CUresult {load}({module_t} *m) {{");
    c += "    CUresult r;\n    CUcontext ctx = NULL;\n    CUdevice dev = 0;\n    int major = 0, minor = 0;\n";
    c += "    if (m == NULL) return CUDA_ERROR_INVALID_VALUE;\n    memset(m, 0, sizeof(*m));\n";
    let _ = writeln!(
        c,
        "    r = cuInit(0);\n    if (r != CUDA_SUCCESS) return {up}_FAIL(m, r, \"cuInit failed (%d)\", (int)r);"
    );
    let _ = writeln!(
        c,
        "    r = cuCtxGetCurrent(&ctx);\n    if (r != CUDA_SUCCESS) return {up}_FAIL(m, r, \"cuCtxGetCurrent failed (%d)\", (int)r);\n    \
         if (ctx == NULL) {{\n        r = cuDeviceGet(&dev, 0);\n        if (r != CUDA_SUCCESS) return {up}_FAIL(m, r, \"no CUDA device (%d)\", (int)r);\n        \
         r = cuDevicePrimaryCtxRetain(&ctx, dev);\n        if (r != CUDA_SUCCESS) return {up}_FAIL(m, r, \"cuDevicePrimaryCtxRetain failed (%d)\", (int)r);\n        \
         m->retained_primary = 1;\n        m->device = dev;\n        r = cuCtxSetCurrent(ctx);\n        \
         if (r != CUDA_SUCCESS) {{ {unload}(m); return {up}_FAIL(m, r, \"cuCtxSetCurrent failed (%d)\", (int)r); }}\n    }}"
    );
    let _ = writeln!(
        c,
        "    m->context = ctx;\n    r = cuCtxGetDevice(&dev);\n    if (r == CUDA_SUCCESS) m->device = dev;\n    if (r == CUDA_SUCCESS) r = cuDeviceGetAttribute(&major, \
         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);\n    if (r == CUDA_SUCCESS) r = cuDeviceGetAttribute(&minor, \
         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);\n    if (r != CUDA_SUCCESS) {{ {unload}(m); \
         return {up}_FAIL(m, r, \"cannot read the device's compute capability (%d)\", (int)r); }}"
    );
    let _ = writeln!(
        c,
        "    if (!({})) {{\n        {unload}(m);\n        return {up}_FAIL(m, CUDA_ERROR_NO_BINARY_FOR_GPU, \
         \"{what} `%s` was compiled for %s, which this device (sm_%d%d) cannot run\", {name_macro}, {up}_TARGET, major, minor);\n    }}",
        c_runs_on(target, false)
    );
    if let Some(t) = k.stamp.cubin_target.filter(|_| k.cubin.is_some()) {
        let _ = writeln!(
            c,
            " r = CUDA_ERROR_NO_BINARY_FOR_GPU;\n if ({}) {{\n r = cuModuleLoadData(&m->module, {p}_cubin);\n if (r == CUDA_SUCCESS) m->from_cubin = 1;\n else m->module = NULL;\n }}",
            c_runs_on(t, true)
        );
    }
    if k.ptx.is_some() {
        let d = min_driver_for_ptx(v).unwrap_or(u32::MAX / 2);
        let _ = writeln!(
            c,
            "    if (m->module == NULL) {{\n        int driver = 0;\n        if (cuDriverGetVersion(&driver) == CUDA_SUCCESS && driver < {d}) {{\n            \
             {unload}(m);\n            return {up}_FAIL(m, CUDA_ERROR_UNSUPPORTED_PTX_VERSION, \"{what} `%s` is PTX ISA %d.%d, which \
             needs a CUDA %d.%d driver; this one is %d.%d\", {name_macro}, {}, {}, {}, {}, driver / 1000, driver % 1000 / 10);\n        }}\n        \
             r = cuModuleLoadData(&m->module, {p}_ptx);\n        if (r != CUDA_SUCCESS) {{ m->module = NULL; {unload}(m); \
             return {up}_FAIL(m, r, \"cuModuleLoadData failed (%d) on the PTX of `%s`\", (int)r, {name_macro}); }}\n    }}",
            v / 10,
            v % 10,
            d / 1000,
            d % 1000 / 10
        );
    } else {
        let _ = writeln!(
            c,
            "    if (m->module == NULL) {{\n        {unload}(m);\n        return {up}_FAIL(m, r, \
             \"{what} `%s` has only a cubin, which this device (sm_%d%d) cannot load (%d)\", {name_macro}, major, minor, (int)r);\n    }}"
        );
    }
    c += lookup;
    c
}

/// The unload function of a C export; `reset` clears the module's function handles.
fn c_unload(names: &CModuleNames, reset: &str) -> String {
    let CModuleNames { unload, module_t, .. } = names;
    let mut c = String::new();
    let _ = writeln!(
        c,
        "CUresult {unload}({module_t} *m) {{\n    CUresult r = CUDA_SUCCESS;\n    \
         CUcontext current = NULL;\n    if (m == NULL) return CUDA_ERROR_INVALID_VALUE;\n    \
         if (m->module != NULL) {{\n        /* In the module's context, which need not be current now. */\n        \
         if (m->context != NULL && cuCtxPushCurrent(m->context) == CUDA_SUCCESS) {{\n            \
         r = cuModuleUnload(m->module);\n            cuCtxPopCurrent(&current);\n        }} else {{\n            \
         r = cuModuleUnload(m->module);\n        }}\n    }}\n    m->module = NULL;\n    {reset}\n    \
         if (m->retained_primary) {{\n        /* Load found no current context and made the primary one current: undo both. */\n        \
         if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == m->context) cuCtxSetCurrent(NULL);\n        \
         cuDevicePrimaryCtxRelease(m->device);\n        m->retained_primary = 0;\n    }}\n    \
         m->context = NULL;\n    return r;\n}}\n"
    );

    c
}

/// The launch function `launch_fn` of kernel `sig` (whose loaded function is `function`).
fn c_launch(
    sig: &KernelSignature,
    fields: &[String],
    p: &str,
    up: &str,
    launch_fn: &str,
    module_t: &str,
    args_t: &str,
    function: &str,
) -> String {
    let mut c = String::new();
    let _ = writeln!(c, "CUresult {launch_fn}({module_t} *m, const {p}_Launch_Config_t *c, const {args_t} *a) {{");
    c += "    CUresult r;\n    unsigned int cl[3];\n    size_t launch_bytes, block_bytes;\n    int i;\n";
    if sig.params.is_empty() {
        c += "    void **params = NULL;\n    (void)a;\n";
    } else {
        let _ = writeln!(c, "    void *params[{}];", sig.params.len());
    }
    let n = c_string(&sig.name);
    let _ = writeln!(
        c,
        "    if (m == NULL) return CUDA_ERROR_INVALID_VALUE;\n    \
         if ({function} == NULL) return {up}_FAIL(m, CUDA_ERROR_NOT_INITIALIZED, \"kernel `%s` is not loaded\", {n});\n    \
         if (c == NULL{}) return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \"kernel `%s`: no launch configuration or arguments\", {n});",
        if sig.params.is_empty() { "" } else { " || a == NULL" }
    );
    for (i, f) in fields.iter().enumerate() {
        let _ = writeln!(c, "    params[{i}] = (void *)&a->{f};");
    }
    c += "    for (i = 0; i < 3; i++) cl[i] = c->cluster[i] ? c->cluster[i] : 1;\n";
    let _ = writeln!(
        c,
        "    for (i = 0; i < 3; i++) {{\n        if (c->grid[i] == 0 || c->block[i] == 0 || c->grid[i] % cl[i] != 0)\n            \
         return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \"kernel `%s`: grid (%u, %u, %u) is not a whole number of clusters of (%u, %u, %u) \
         with blocks of (%u, %u, %u)\", {n}, c->grid[0], c->grid[1], c->grid[2], cl[0], cl[1], cl[2], c->block[0], c->block[1], c->block[2]);\n    }}"
    );
    if let Some([x, y, z]) = sig.block_dim {
        let _ = writeln!(
            c,
            "    if (c->block[0] != {x} || c->block[1] != {y} || c->block[2] != {z})\n        return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \
             \"kernel `%s` was built for blocks of ({x}, {y}, {z}), launched with (%u, %u, %u)\", {n}, c->block[0], c->block[1], c->block[2]);"
        );
    }
    if let Some([x, y, z]) = sig.max_threads {
        let total = u64::from(x) * u64::from(y) * u64::from(z);
        let _ = writeln!(
            c,
            "    if (c->block[0] > {x} || c->block[1] > {y} || c->block[2] > {z} || (unsigned long long)c->block[0] * c->block[1] * c->block[2] > {total}ULL)\n        \
             return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \"kernel `%s` takes blocks of at most ({x}, {y}, {z}) (max_threads), launched with (%u, %u, %u)\", \
             {n}, c->block[0], c->block[1], c->block[2]);"
        );
    }
    for (i, (param, field)) in sig.params.iter().zip(fields).enumerate() {
        let pname = c_string(&param.name);
        match &param.kind {
            ParamKind::Pointer { elem, min_len, align } => {
                if *align > 1 {
                    let _ = writeln!(
                        c,
                        "    if (a->{field} % {align}ULL != 0)\n        return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \"argument {i} (`%s`) must be \
                         {align}-byte aligned, got 0x%llx\", {pname}, (unsigned long long)a->{field});"
                    );
                }
                if let Some(len) = min_len {
                    let need = elem.storage_bytes(*len);
                    let _ = writeln!(
                        c,
                        "    {{\n        CUdeviceptr base = 0;\n        size_t size = 0;\n        if (cuMemGetAddressRange(&base, &size, a->{field}) == \
                         CUDA_SUCCESS && base + size - a->{field} < {need}ULL)\n            return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \"argument {i} \
                         (`%s`) needs {len} {} elements ({need} bytes), the pointer reaches %llu\", {pname}, (unsigned long long)(base + size - a->{field}));\n    }}",
                        elem.cute_name()
                    );
                }
            }
            ParamKind::Scalar { dtype, multiple_of } if *multiple_of > 1 && dtype.is_int() && dtype.bytes() <= 8 => {
                let (cast, fmt) =
                    if dtype.is_signed() { ("long long", "%lld") } else { ("unsigned long long", "%llu") };
                // In 64 bits: a divisor past the leaf's range divides only 0.
                let rem = match (dtype.is_signed(), i64::try_from(*multiple_of)) {
                    (true, Ok(m)) => format!("(long long)a->{field} % {m}LL"),
                    (false, _) => format!("(unsigned long long)a->{field} % {multiple_of}ULL"),
                    (true, Err(_)) => format!("a->{field}"),
                };
                let _ = writeln!(
                    c,
                    "    if ({rem} != 0)\n        return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \"argument {i} (`%s`) must be \
                     a multiple of {multiple_of}, got {fmt}\", {pname}, ({cast})a->{field});"
                );
            }
            _ => {}
        }
    }
    let s = &sig.smem;
    let _ = writeln!(c, "    launch_bytes = (size_t){} + c->dynamic_smem_bytes;", s.dynamic_start);
    if s.in_window {
        c += "    block_bytes = launch_bytes;\n";
    } else {
        let _ = writeln!(c, "    block_bytes = (size_t){} + c->dynamic_smem_bytes;", s.static_bytes);
    }
    let _ = writeln!(
        c,
        "    if (launch_bytes > 0x7fffffffu || block_bytes > 0x7fffffffu)\n        return {up}_FAIL(m, CUDA_ERROR_INVALID_VALUE, \
         \"kernel `%s`: %u bytes of dynamic shared memory is too many\", {n}, c->dynamic_smem_bytes);"
    );
    let _ = writeln!(
        c,
        "    if (block_bytes > {}) {{\n        r = cuFuncSetAttribute({function}, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)launch_bytes);\n        \
         if (r != CUDA_SUCCESS) return {up}_FAIL(m, r, \"kernel `%s` needs %llu bytes of shared memory per block, which the device does not \
         allow (%d)\", {n}, (unsigned long long)block_bytes, (int)r);\n    }}",
        crate::validate::MAX_STATIC_SHARED_BYTES
    );
    let _ = writeln!(
        c,
        "#if CUDA_VERSION >= 12000\n    {{\n        CUlaunchConfig lc;\n        CUlaunchAttribute attr;\n        memset(&lc, 0, sizeof(lc));\n        \
         memset(&attr, 0, sizeof(attr));\n        lc.gridDimX = c->grid[0];\n        lc.gridDimY = c->grid[1];\n        lc.gridDimZ = c->grid[2];\n        \
         lc.blockDimX = c->block[0];\n        lc.blockDimY = c->block[1];\n        lc.blockDimZ = c->block[2];\n        \
         lc.sharedMemBytes = (unsigned int)launch_bytes;\n        lc.hStream = c->stream;\n        if (cl[0] != 1 || cl[1] != 1 || cl[2] != 1) {{\n            \
         attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;\n            attr.value.clusterDim.x = cl[0];\n            attr.value.clusterDim.y = cl[1];\n            \
         attr.value.clusterDim.z = cl[2];\n            lc.attrs = &attr;\n            lc.numAttrs = 1;\n        }}\n        \
         r = cuLaunchKernelEx(&lc, {function}, params, NULL);\n    }}\n#else\n    if (cl[0] != 1 || cl[1] != 1 || cl[2] != 1)\n        \
         return {up}_FAIL(m, CUDA_ERROR_NOT_SUPPORTED, \"kernel `%s`: clusters need cuLaunchKernelEx (CUDA 12 headers)\", {n});\n    \
         r = cuLaunchKernel({function}, c->grid[0], c->grid[1], c->grid[2], c->block[0], c->block[1], c->block[2], (unsigned int)launch_bytes, \
         c->stream, params, NULL);\n#endif\n    if (r != CUDA_SUCCESS) {{\n        const char *e = NULL;\n        cuGetErrorName(r, &e);\n        \
         return {up}_FAIL(m, r, \"launching `%s` failed: %s\", {n}, e != NULL ? e : \"unknown error\");\n    }}\n    m->error[0] = 0;\n    \
         return CUDA_SUCCESS;\n}}"
    );
    c
}

/// Why PTX cannot be exported on its own: it calls an external
/// function it does not define (an `.extern .func`, from an `ExternFn` whose code is linked
/// at load time), which loading the export would fail to resolve, or it has no kernel (a
/// standalone `DeviceFunction`, which is linked into other code rather than launched).
pub(crate) fn unexportable_ptx(ptx: Option<&str>, name: &str) -> Option<String> {
    let ptx = ptx?;
    if let Some(line) = ptx.lines().find(|l| l.trim_start().starts_with(".extern .func")) {
        let symbol = line.split(['(', ')', ' ', '\t']).rfind(|w| !w.is_empty()).unwrap_or("?");
        return Some(format!(
            "`{name}` cannot be exported: its PTX calls external function `{symbol}`, which it does not \
             define (an ExternFn that is not inlined from bitcode); link the defining module at load time \
             (CudaContext::load_linked) instead"
        ));
    }
    if ptx.contains(".visible .func") && !ptx.contains(".entry") {
        return Some(format!(
            "`{name}` cannot be exported: its PTX has no kernel (a DeviceFunction is linked into other \
             code, e.g. with compile_relocatable, not exported)"
        ));
    }
    None
}

impl CompiledKernel {
    /// Export the kernel to C (CuTe DSL's `export_to_c(file_path, file_name, function_prefix)`):
    /// `<name>.h`, `<name>.c`, `<name>.sig`, `<name>.ptx`, `<name>.cubin` and, when a C compiler
    /// and `cuda.h` are found, `<name>.o`, in `dir`. Symbols start with `prefix` (`name` when
    /// empty). See [`aot`](crate::aot).
    pub fn export_c(&self, dir: impl AsRef<Path>, name: &str, prefix: &str) -> Result<ExportReport, AotError> {
        self.export_c_with(dir, name, prefix, &ExportOptions::default())
    }

    /// [`export_c`](Self::export_c) with options: whether to build the object file, the C
    /// compiler and where `cuda.h` is.
    pub fn export_c_with(
        &self,
        dir: impl AsRef<Path>,
        name: &str,
        prefix: &str,
        options: &ExportOptions,
    ) -> Result<ExportReport, AotError> {
        ExportedKernel::from_compiled(self).write_c(dir, name, prefix, options)
    }

    /// The kernel as an [`ExportedKernel`], without writing files.
    pub fn to_exported(&self) -> ExportedKernel {
        ExportedKernel::from_compiled(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_on_rules() {
        assert!(ptx_runs_on(Arch::Sm80, (8, 9)) && ptx_runs_on(Arch::Sm80, (9, 0)));
        assert!(!ptx_runs_on(Arch::Sm89, (8, 6)));
        assert!(ptx_runs_on(Arch::Sm90a, (9, 0)) && !ptx_runs_on(Arch::Sm90a, (10, 0)));
        assert!(ptx_runs_on(Arch::Sm100f, (10, 3)) && !ptx_runs_on(Arch::Sm100f, (12, 0)));
        assert!(binary_runs_on(Arch::Sm80, (8, 9)) && !binary_runs_on(Arch::Sm80, (9, 0)));
        assert!(!binary_runs_on(Arch::Sm100a, (10, 3)));
        for t in Arch::ALL {
            for cc in [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)] {
                if let Some(d) = Arch::from_compute_capability(cc.0, cc.1) {
                    assert_eq!(binary_runs_on(t, cc), d.can_run_binary_built_for(t), "{t:?} on {cc:?}");
                }
            }
        }
    }

    #[test]
    fn min_driver() {
        assert_eq!(min_driver_for_ptx(78), Some(11_080));
        assert_eq!(min_driver_for_ptx(87), Some(12_080));
        assert_eq!(min_driver_for_ptx(90), Some(13_000));
        assert_eq!(min_driver_for_ptx(65), Some(10_020));
    }

    #[test]
    fn identifiers() {
        assert_eq!(c_ident("a.b-c"), "a_b_c");
        assert_eq!(c_ident("3x"), "_3x");
        assert_eq!(c_string("a\"b\n"), "\"a\\\"b\\012\"");
    }
}

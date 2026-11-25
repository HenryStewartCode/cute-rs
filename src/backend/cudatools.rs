//! The CUDA toolkit's command-line tools: `ptxas` (PTX to cubin), `nvdisasm` and `cuobjdump`
//! (cubin to SASS), and the driver-JIT equivalents of `ptxas`'s flags for machines that have
//! only the driver.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use super::options::OptionError;
use super::toolchain::CompileError;
use crate::log::{Level, log_with};

/// Environment variable naming the `ptxas` to use (CuTe-style override). A value that names no
/// file disables `ptxas`, and with it the cubin path; the driver compiles the PTX instead.
pub const ENV_PTXAS: &str = "CUTE_RS_PTXAS";

/// A located command-line tool and the release it reports (`release 12.8`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tool {
    pub path: PathBuf,
    /// `(major, minor)` of the CUDA release, if `--version` printed one.
    pub release: Option<(u32, u32)>,
}

impl Tool {
    fn probe(path: PathBuf) -> Tool {
        let release = Command::new(&path).arg("--version").output().ok().and_then(|o| {
            parse_release(&format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
        });
        Tool { path, release }
    }
}

/// The toolkit tools that were found. Any of them may be missing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaTools {
    pub ptxas: Option<Tool>,
    pub nvdisasm: Option<Tool>,
    pub cuobjdump: Option<Tool>,
}

impl CudaTools {
    /// Find the tools: `ptxas` in `$CUTE_RS_PTXAS` (a file; one that does not exist disables
    /// `ptxas` and does not fall back to a search), else in `$CUDA_HOME/bin`, `$CUDA_PATH/bin`,
    /// `PATH` and `/usr/local/cuda/bin`; `nvdisasm` and `cuobjdump` next to the `ptxas`, then
    /// in the same places.
    pub fn discover() -> CudaTools {
        let path: Vec<PathBuf> =
            std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
        CudaTools::discover_with(|n| std::env::var_os(n), &path)
    }

    /// [`discover`](Self::discover) with the environment and `PATH` given.
    pub fn discover_with(env: impl Fn(&str) -> Option<std::ffi::OsString>, path: &[PathBuf]) -> CudaTools {
        let mut dirs: Vec<PathBuf> = Vec::new();
        for var in ["CUDA_HOME", "CUDA_PATH"] {
            if let Some(h) = env(var).filter(|h| !h.is_empty()) {
                dirs.push(Path::new(&h).join("bin"));
            }
        }
        dirs.extend(path.iter().cloned());
        dirs.push(PathBuf::from("/usr/local/cuda/bin"));
        let find = |name: &str, first: Option<&Path>| -> Option<Tool> {
            first
                .map(|d| d.join(name))
                .into_iter()
                .chain(dirs.iter().map(|d| d.join(name)))
                .find(|p| p.is_file())
                .map(Tool::probe)
        };
        let ptxas = match env(ENV_PTXAS).filter(|p| !p.is_empty()) {
            Some(p) => Some(PathBuf::from(p)).filter(|p| p.is_file()).map(Tool::probe),
            None => find("ptxas", None),
        };
        let beside = ptxas.as_ref().and_then(|t| t.path.parent()).map(Path::to_path_buf);
        let nvdisasm = find("nvdisasm", beside.as_deref());
        let cuobjdump = find("cuobjdump", beside.as_deref());
        CudaTools { ptxas, nvdisasm, cuobjdump }
    }
}

/// The CUDA release in `ptxas --version`'s text (`Cuda compilation tools, release 12.8, V12.8.93`).
pub fn parse_release(text: &str) -> Option<(u32, u32)> {
    let rest = &text[text.find("release ")? + "release ".len()..];
    let mut parts = rest.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty());
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

/// The newest PTX ISA version (`10 * major + minor`) the `ptxas` of CUDA release
/// `major.minor` assembles: 8.m for CUDA 12.m up to 12.8 (PTX 8.7), 8.8 for 12.9, then 9.N for
/// CUDA 13.N; 7.m for CUDA 11.m (7.8 for 11.8).
pub fn ptxas_max_ptx(major: u32, minor: u32) -> u32 {
    match (major, minor) {
        (..=10, _) => 65,
        (11, m) => 70 + m.min(8),
        (12, m @ 0..=5) => 80 + m,
        (12, 6) => 85,
        (12, 7) => 86,
        (12, 8) => 87,
        (12, _) => 88,
        (13, n) => 90 + n,
        (_, _) => u32::MAX,
    }
}

/// The result of one `ptxas` run.
#[derive(Clone, Debug)]
pub struct Assembled {
    pub cubin: Vec<u8>,
    /// What `ptxas -v` printed (resource usage), parsed into `CompiledKernel::resource_usage`.
    pub log: String,
    /// The command line that ran.
    pub command: String,
}

/// A temporary directory removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    /// A new directory under the system's temporary directory, created by this call (never one
    /// that already exists, which someone else could control: the cubin read back from it is
    /// what runs) and, on Unix, private to this user.
    fn new(tag: &str) -> std::io::Result<TempDir> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
        let mut last = None;
        for _ in 0..64 {
            let dir = std::env::temp_dir().join(format!(
                "cute-rs-{tag}-{}-{}-{nanos:08x}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
            match builder.create(&dir) {
                Ok(()) => return Ok(TempDir(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| std::io::Error::other("no free temporary directory name")))
    }
}

/// `args` as one line a shell would split back into them (for logs and error messages; the
/// tools themselves are run without a shell).
fn shell_line(program: &Path, args: &[String]) -> String {
    let quote = |w: &str| -> String {
        if !w.is_empty() && w.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_=.,/:+@%".contains(&b)) {
            w.to_string()
        } else {
            format!("'{}'", w.replace('\'', "'\\''"))
        }
    };
    std::iter::once(program.display().to_string().as_str())
        .chain(args.iter().map(String::as_str))
        .map(quote)
        .collect::<Vec<_>>()
        .join(" ")
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Whether `flags` already choose the optimization level (`-O2`, `--opt-level 2`) or line
/// information, so that cute-rs's own are left out.
fn sets_opt_level(flags: &[String]) -> bool {
    flags.iter().any(|f| {
        (f.starts_with("-O") && f[2..].chars().all(|c| c.is_ascii_digit()) && f.len() > 2)
            || f.starts_with("--opt-level")
            || f.starts_with("-opt-level")
    })
}

fn sets_line_info(flags: &[String]) -> bool {
    flags.iter().any(|f| matches!(f.as_str(), "-lineinfo" | "--generate-line-info"))
}

/// Assemble `ptx` for `chip` with `ptxas -arch=<chip> -O<n> [-lineinfo] -v <flags> -o ... ...`
/// in a temporary directory that is removed afterwards.
pub fn run_ptxas(
    ptxas: &Tool,
    chip: &str,
    ptx: &str,
    opt_level: u32,
    line_info: bool,
    flags: &[String],
) -> Result<Assembled, CompileError> {
    let dir = TempDir::new("ptxas").map_err(|e| CompileError::Tool(format!("creating a temporary directory: {e}")))?;
    let (source, out) = (dir.0.join("kernel.ptx"), dir.0.join("kernel.cubin"));
    std::fs::write(&source, ptx).map_err(|e| CompileError::Tool(format!("writing {}: {e}", source.display())))?;
    let mut args: Vec<String> = vec![format!("-arch={chip}")];
    if !sets_opt_level(flags) {
        args.push(format!("-O{opt_level}"));
    }
    if line_info && !sets_line_info(flags) {
        args.push("-lineinfo".into());
    }
    args.push("-v".into());
    args.extend(flags.iter().cloned());
    args.push("-o".into());
    args.push(out.display().to_string());
    args.push(source.display().to_string());
    let command = shell_line(&ptxas.path, &args);
    log_with(Level::Debug, || format!("running {command}"));
    let output = Command::new(&ptxas.path)
        .args(&args)
        .output()
        .map_err(|e| CompileError::Tool(format!("{}: {e}", ptxas.path.display())))?;
    let log = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(CompileError::Ptxas(format!("`{command}` failed ({}):\n{log}", output.status)));
    }
    let cubin = std::fs::read(&out).map_err(|e| CompileError::Tool(format!("reading {}: {e}", out.display())))?;
    Ok(Assembled { cubin, log, command })
}

/// Disassemble `cubin` to SASS: with `nvdisasm <options> file` (default options `-g -c`), else
/// with `cuobjdump -sass file`. The error says which tools were missing.
pub fn disassemble(tools: &CudaTools, cubin: &[u8], nvdisasm_options: &[String]) -> Result<String, String> {
    let (tool, via_nvdisasm) = match (&tools.nvdisasm, &tools.cuobjdump) {
        (Some(t), _) => (t, true),
        (None, Some(t)) => (t, false),
        (None, None) => {
            return Err("neither nvdisasm nor cuobjdump was found (install the CUDA toolkit; they are looked up \
                        next to ptxas, in $CUDA_HOME/bin, $CUDA_PATH/bin, PATH and /usr/local/cuda/bin)"
                .to_string());
        }
    };
    let dir = TempDir::new("sass").map_err(|e| format!("creating a temporary directory: {e}"))?;
    let file = dir.0.join("kernel.cubin");
    std::fs::write(&file, cubin).map_err(|e| format!("writing {}: {e}", file.display()))?;
    let mut args: Vec<String> = if via_nvdisasm { nvdisasm_options.to_vec() } else { vec!["-sass".into()] };
    args.push(file.display().to_string());
    log_with(Level::Debug, || format!("running {}", shell_line(&tool.path, &args)));
    let output = Command::new(&tool.path).args(&args).output().map_err(|e| format!("{}: {e}", tool.path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed ({}): {}{}",
            tool.path.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One option of the CUDA driver's JIT that `ptxas` flags map onto (`CUjit_option`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitOption {
    /// `CU_JIT_MAX_REGISTERS` (0): the register limit per thread (`--maxrregcount`).
    MaxRegisters(u32),
    /// `CU_JIT_OPTIMIZATION_LEVEL` (7): 0 to 4; 4, the default, matches `ptxas -O3`.
    OptimizationLevel(u32),
    /// `CU_JIT_LOG_VERBOSE` (12): verbose assembler output in the info log (`-v`).
    LogVerbose(bool),
    /// `CU_JIT_GENERATE_LINE_INFO` (13): keep the PTX's line information in the SASS.
    GenerateLineInfo(bool),
}

impl JitOption {
    /// The `CUjit_option` number.
    pub fn code(self) -> u32 {
        match self {
            JitOption::MaxRegisters(_) => 0,
            JitOption::OptimizationLevel(_) => 7,
            JitOption::LogVerbose(_) => 12,
            JitOption::GenerateLineInfo(_) => 13,
        }
    }

    /// The value, as the driver takes it (in place of the option's pointer).
    pub fn value(self) -> u32 {
        match self {
            JitOption::MaxRegisters(n) | JitOption::OptimizationLevel(n) => n,
            JitOption::LogVerbose(on) | JitOption::GenerateLineInfo(on) => u32::from(on),
        }
    }
}

/// Map `ptxas` flags onto driver JIT options, for machines without a `ptxas`:
/// `--maxrregcount=N` to `CU_JIT_MAX_REGISTERS`, `-O<n>` to `CU_JIT_OPTIMIZATION_LEVEL` (3 is
/// the JIT's 4), `-lineinfo` and `--generate-line-info` to `CU_JIT_GENERATE_LINE_INFO`, and
/// `-v` and `--verbose` to `CU_JIT_LOG_VERBOSE`. Any other flag is an error naming `ptxas`,
/// which is the only thing that takes it.
pub fn jit_options_for_ptxas_flags(flags: &[String]) -> Result<Vec<JitOption>, OptionError> {
    let mut out: Vec<JitOption> = Vec::new();
    let mut put = |o: JitOption| {
        out.retain(|e| e.code() != o.code());
        out.push(o);
    };
    let mut i = 0;
    while i < flags.len() {
        let flag = flags[i].as_str();
        i += 1;
        let (name, inline) = match flag.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (flag, None),
        };
        let mut value = |what: &str| -> Result<String, OptionError> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => {
                    let v = flags
                        .get(i)
                        .cloned()
                        .ok_or_else(|| OptionError::new(format!("the ptxas flag `{flag}` needs a value ({what})")))?;
                    i += 1;
                    Ok(v)
                }
            }
        };
        let number = |text: &str, flag: &str| -> Result<u32, OptionError> {
            text.trim()
                .parse::<u32>()
                .map_err(|_| OptionError::new(format!("the ptxas flag `{flag}` takes a number, not `{text}`")))
        };
        match name {
            "--maxrregcount" | "-maxrregcount" => {
                let n = number(&value("a register count")?, flag)?;
                put(JitOption::MaxRegisters(n));
            }
            "--opt-level" | "-opt-level" => {
                let n = number(&value("0 to 3")?, flag)?;
                put(JitOption::OptimizationLevel(ptx_level_to_jit(n, flag)?));
            }
            f if f.len() == 3 && f.starts_with("-O") && f.as_bytes()[2].is_ascii_digit() => {
                put(JitOption::OptimizationLevel(ptx_level_to_jit(u32::from(f.as_bytes()[2] - b'0'), flag)?));
            }
            "-lineinfo" | "--generate-line-info" => put(JitOption::GenerateLineInfo(true)),
            "-v" | "--verbose" => put(JitOption::LogVerbose(true)),
            _ => {
                return Err(OptionError::new(format!(
                    "the ptxas flag `{flag}` has no CUDA driver JIT equivalent, so it needs ptxas itself: install a \
                     CUDA toolkit (ptxas is looked up in $CUTE_RS_PTXAS, $CUDA_HOME/bin, $CUDA_PATH/bin, PATH and \
                     /usr/local/cuda/bin); without one, only --maxrregcount=N, -O<n>, -lineinfo and -v can be \
                     forwarded"
                )));
            }
        }
    }
    Ok(out)
}

fn ptx_level_to_jit(level: u32, flag: &str) -> Result<u32, OptionError> {
    match level {
        0..=2 => Ok(level),
        3 => Ok(4),
        _ => Err(OptionError::new(format!("the ptxas flag `{flag}`: the optimization level must be 0 to 3"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn release_and_ptx_ceiling() {
        assert_eq!(parse_release("Cuda compilation tools, release 12.8, V12.8.93"), Some((12, 8)));
        assert_eq!(parse_release("release 13.0"), Some((13, 0)));
        assert_eq!(parse_release("nothing"), None);
        assert_eq!(ptxas_max_ptx(12, 8), 87);
        assert_eq!(ptxas_max_ptx(12, 0), 80);
        assert_eq!(ptxas_max_ptx(12, 1), 81);
        assert_eq!(ptxas_max_ptx(12, 9), 88);
        assert_eq!(ptxas_max_ptx(13, 1), 91);
        assert_eq!(ptxas_max_ptx(11, 8), 78);
    }

    #[test]
    fn ptxas_flags_map_onto_jit_options() {
        let map = |t: &str| jit_options_for_ptxas_flags(&flags(t)).unwrap();
        assert_eq!(map("--maxrregcount=128"), [JitOption::MaxRegisters(128)]);
        assert_eq!(map("-maxrregcount 32"), [JitOption::MaxRegisters(32)]);
        assert_eq!(map("-O3"), [JitOption::OptimizationLevel(4)]);
        assert_eq!(map("-O1 -O2"), [JitOption::OptimizationLevel(2)]);
        assert_eq!(map("--opt-level=0"), [JitOption::OptimizationLevel(0)]);
        assert_eq!(map("-lineinfo -v"), [JitOption::GenerateLineInfo(true), JitOption::LogVerbose(true)]);
        assert_eq!(
            map("--generate-line-info --verbose"),
            [JitOption::GenerateLineInfo(true), JitOption::LogVerbose(true)]
        );
        assert!(map("").is_empty());
        let codes: Vec<(u32, u32)> =
            map("--maxrregcount=64 -O2 -v -lineinfo").iter().map(|o| (o.code(), o.value())).collect();
        assert_eq!(codes, [(0, 64), (7, 2), (12, 1), (13, 1)]);
        for bad in ["--gpu-name=sm_80", "-warn-spills", "-O4", "--maxrregcount=x", "--maxrregcount"] {
            let e = jit_options_for_ptxas_flags(&flags(bad)).unwrap_err().to_string();
            assert!(e.contains("ptxas"), "{bad}: {e}");
        }
    }

    #[test]
    fn command_lines_are_quoted_for_display() {
        let args = flags("-arch=sm_80 -O3");
        assert_eq!(shell_line(Path::new("/usr/bin/ptxas"), &args), "/usr/bin/ptxas -arch=sm_80 -O3");
        let odd = vec!["--x=a b".to_string(), "it's".to_string(), String::new()];
        assert_eq!(shell_line(Path::new("/a dir/ptxas"), &odd), "'/a dir/ptxas' '--x=a b' 'it'\\''s' ''");
    }

    #[test]
    fn temporary_directories_are_new_and_removed() {
        let (a, b) = (TempDir::new("test").unwrap(), TempDir::new("test").unwrap());
        assert_ne!(a.0, b.0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            if mode(&a.0) != 0o700 {
                // Some filesystems (network or FUSE mounts, such as a pod's volume under
                // TMPDIR) ignore permission bits. Tell those apart from a directory created
                // too open by asking for 0700 outright: a filesystem that honours modes
                // reports it, and then the directory should have been created that way.
                std::fs::set_permissions(&a.0, std::fs::Permissions::from_mode(0o700)).unwrap();
                assert_ne!(
                    mode(&a.0),
                    0o700,
                    "{} was created with mode {:o}, not 0700, on a filesystem that honours modes",
                    a.0.display(),
                    mode(&b.0)
                );
                eprintln!("skipping the mode check: {} ignores permission bits", a.0.display());
            }
        }
        let path = a.0.clone();
        drop(a);
        assert!(!path.exists());
    }

    #[test]
    fn user_flags_replace_ours() {
        assert!(sets_opt_level(&flags("-O2")) && sets_opt_level(&flags("--opt-level=1")));
        assert!(!sets_opt_level(&flags("-v --maxrregcount=3")));
        assert!(sets_line_info(&flags("-lineinfo")) && !sets_line_info(&flags("-v")));
    }

    #[test]
    fn discovery_honours_the_override() {
        let none: Vec<PathBuf> = vec![];
        // A value that names no file disables ptxas without searching.
        let t = CudaTools::discover_with(
            |n| (n == ENV_PTXAS).then(|| "/definitely/not/ptxas".into()),
            &[PathBuf::from("/usr/local/cuda/bin")],
        );
        assert_eq!(t.ptxas, None);
        let t = CudaTools::discover_with(|_| None, &none);
        assert!(t.ptxas.is_none() || t.ptxas.unwrap().path.is_file());
    }
}

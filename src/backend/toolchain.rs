//! Compiling MLIR to PTX with an installed `mlir-opt`.
//!
//! The kernel's MLIR is piped through MLIR's standard GPU pipeline, which
//! lowers it to NVVM, runs LLVM's NVPTX backend and embeds the resulting
//! PTX in a `gpu.binary` op; the PTX is then extracted from that op. No CUDA
//! toolkit is needed.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Target GPU architectures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    /// Hopper with the architecture-specific features (WGMMA).
    Sm90a,
    Sm100,
    Sm100a,
    Sm120,
    Sm120a,
}

impl Arch {
    /// The `sm_XX` name.
    pub fn chip(self) -> &'static str {
        match self {
            Arch::Sm80 => "sm_80",
            Arch::Sm86 => "sm_86",
            Arch::Sm89 => "sm_89",
            Arch::Sm90 => "sm_90",
            Arch::Sm90a => "sm_90a",
            Arch::Sm100 => "sm_100",
            Arch::Sm100a => "sm_100a",
            Arch::Sm120 => "sm_120",
            Arch::Sm120a => "sm_120a",
        }
    }

    /// The Hopper-or-newer architectures.
    pub fn is_hopper_or_newer(self) -> bool {
        !matches!(self, Arch::Sm80 | Arch::Sm86 | Arch::Sm89)
    }

    /// The architecture number: 80 for sm_80, 90 for sm_90 and sm_90a.
    pub fn sm(self) -> u32 {
        match self {
            Arch::Sm80 => 80,
            Arch::Sm86 => 86,
            Arch::Sm89 => 89,
            Arch::Sm90 | Arch::Sm90a => 90,
            Arch::Sm100 | Arch::Sm100a => 100,
            Arch::Sm120 | Arch::Sm120a => 120,
        }
    }

    /// The PTX ISA version kernels are compiled with unless an instruction
    /// needs a newer one, as `10 * major + minor` (70 for 7.0).
    pub fn min_ptx(self) -> u32 {
        match self {
            Arch::Sm80 => 70,
            Arch::Sm86 => 71,
            Arch::Sm89 => 78,
            // mbarrier.arrive.expect_tx needs PTX 8.0.
            Arch::Sm90 | Arch::Sm90a => 80,
            Arch::Sm100 | Arch::Sm100a => 86,
            Arch::Sm120 | Arch::Sm120a => 87,
        }
    }

    /// The LLVM feature selecting [`Arch::min_ptx`].
    pub fn ptx_features(self) -> &'static str {
        match self.min_ptx() {
            70 => "+ptx70",
            71 => "+ptx71",
            78 => "+ptx78",
            80 => "+ptx80",
            86 => "+ptx86",
            _ => "+ptx87",
        }
    }
}

#[derive(Debug)]
pub enum CompileError {
    /// `mlir-opt` could not be run.
    Tool(String),
    /// `mlir-opt` rejected the input.
    Mlir(String),
    /// The kernel uses an instruction the target architecture lacks.
    Unsupported(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Tool(m) => write!(f, "could not run mlir-opt: {m}"),
            CompileError::Mlir(m) => write!(f, "mlir-opt failed:\n{m}"),
            CompileError::Unsupported(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for CompileError {}

/// A located `mlir-opt` binary.
#[derive(Clone, Debug)]
pub struct Toolchain {
    pub mlir_opt: PathBuf,
}

impl Toolchain {
    /// Use a specific `mlir-opt`.
    pub fn new(mlir_opt: impl Into<PathBuf>) -> Toolchain {
        Toolchain { mlir_opt: mlir_opt.into() }
    }

    /// Find `mlir-opt`: `$CUTE_MLIR_OPT`, then `$LLVM_DIR/bin`, then `PATH`,
    /// then common Homebrew and distribution install locations.
    pub fn discover() -> Option<Toolchain> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(p) = std::env::var_os("CUTE_MLIR_OPT") {
            candidates.push(p.into());
        }
        if let Some(dir) = std::env::var_os("LLVM_DIR") {
            candidates.push(Path::new(&dir).join("bin").join("mlir-opt"));
        }
        if let Some(path) = std::env::var_os("PATH") {
            candidates.extend(std::env::split_paths(&path).map(|d| d.join("mlir-opt")));
        }
        for root in ["/opt/homebrew/opt", "/usr/local/opt"] {
            if let Ok(entries) = std::fs::read_dir(root) {
                let mut dirs: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("llvm")))
                    .collect();
                dirs.sort();
                candidates.extend(dirs.into_iter().rev().map(|d| d.join("bin").join("mlir-opt")));
            }
        }
        for v in (17..=30).rev() {
            candidates.push(PathBuf::from(format!("/usr/lib/llvm-{v}/bin/mlir-opt")));
        }
        candidates.into_iter().find(|p| p.is_file()).map(Toolchain::new)
    }

    fn run(&self, input: &str, args: &[String]) -> Result<String, CompileError> {
        let mut child = Command::new(&self.mlir_opt)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CompileError::Tool(format!("{}: {e}", self.mlir_opt.display())))?;
        let mut stdin = child.stdin.take().expect("piped stdin");
        let text = input.to_string();
        let writer = std::thread::spawn(move || stdin.write_all(text.as_bytes()));
        let output = child.wait_with_output().map_err(|e| CompileError::Tool(e.to_string()))?;
        writer
            .join()
            .expect("stdin writer thread")
            .map_err(|e| CompileError::Tool(format!("writing to mlir-opt: {e}")))?;
        if !output.status.success() {
            return Err(CompileError::Mlir(String::from_utf8_lossy(&output.stderr).into_owned()));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Parse and verify MLIR text.
    pub fn verify(&self, mlir: &str) -> Result<(), CompileError> {
        self.run(mlir, &[]).map(|_| ())
    }

    /// Lower MLIR text to PTX for `arch`, as PTX ISA version
    /// [`Arch::min_ptx`].
    ///
    /// This is the raw path: it neither checks that the instructions exist
    /// on `arch` nor raises the PTX version for instructions that need a
    /// newer one, so a mismatch only shows up later, in `ptxas` or the
    /// driver's JIT. For a [`Kernel`](crate::Kernel), use
    /// [`Kernel::compile_ptx`](crate::Kernel::compile_ptx), which does both.
    pub fn compile_ptx(&self, mlir: &str, arch: Arch) -> Result<String, CompileError> {
        self.compile_ptx_isa(mlir, arch, arch.min_ptx())
    }

    /// Lower MLIR text to PTX for `arch` as PTX ISA version `ptx` (`10 *
    /// major + minor`, at least [`Arch::min_ptx`]), for kernels whose inline
    /// PTX needs a newer version than the architecture's minimum. Like
    /// [`compile_ptx`](Self::compile_ptx), it does not check the target.
    pub fn compile_ptx_isa(&self, mlir: &str, arch: Arch, ptx: u32) -> Result<String, CompileError> {
        // Vector accesses are lowered first with their natural (vector)
        // alignment: the DSL only emits vector loads and stores whose
        // alignment it has proven, and the default lowering would otherwise
        // fall back to element alignment and lose the 128-bit accesses.
        let pipeline = format!(
            "--pass-pipeline=builtin.module(gpu.module(convert-vector-to-llvm{{use-vector-alignment=1}}),\
             gpu-lower-to-nvvm-pipeline{{cubin-chip={} cubin-features={} cubin-format=isa \
             kernel-bare-ptr-calling-convention=1 opt-level=3}})",
            arch.chip(),
            format_args!("+ptx{}", ptx.max(arch.min_ptx()))
        );
        let out = self.run(mlir, &[pipeline])?;
        extract_assembly(&out).ok_or_else(|| CompileError::Mlir(format!("no PTX found in the pipeline output:\n{out}")))
    }
}

/// Pull the PTX string out of `gpu.binary ... assembly = "..."`.
fn extract_assembly(mlir: &str) -> Option<String> {
    let start = mlir.find("assembly = \"")? + "assembly = \"".len();
    let mut bytes = Vec::new();
    let mut chars = mlir[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(String::from_utf8_lossy(&bytes).into_owned()),
            '\\' => {
                let a = chars.next()?;
                match a {
                    '\\' | '"' => bytes.push(a as u8),
                    'n' => bytes.push(b'\n'),
                    't' => bytes.push(b'\t'),
                    _ => {
                        let b = chars.next()?;
                        let hex: String = [a, b].iter().collect();
                        bytes.push(u8::from_str_radix(&hex, 16).ok()?);
                    }
                }
            }
            c => {
                let mut buf = [0; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn decodes_mlir_string_escapes() {
        let text = r#"gpu.binary @k [#gpu.object<#nvvm.target, assembly = "//\0A.version 7.0\0A\09ld \22x\22 \\">]"#;
        assert_eq!(super::extract_assembly(text).unwrap(), "//\n.version 7.0\n\tld \"x\" \\");
    }
}

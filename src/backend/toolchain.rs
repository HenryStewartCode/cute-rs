//! Compiling MLIR to PTX with an installed `mlir-opt`.
//!
//! The kernel's MLIR goes through MLIR's standard GPU pipeline
//! (`gpu-lower-to-nvvm-pipeline`), which lowers it to NVVM, runs LLVM's NVPTX
//! backend and embeds the resulting PTX in a `gpu.binary` op; the PTX is then
//! extracted from that op. No CUDA toolkit is needed. The pipeline runs as two
//! `mlir-opt` invocations split at the LLVM dialect (see
//! [`LoweringPipeline`]), so that stage hooks can edit the kernel in between.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use super::cudatools::CudaTools;
use super::options::{OptionError, StageHook};
use super::passes::{self, LoweringPipeline};
use super::stage::LoweredModule;
use crate::log::Level;

/// Target GPU architectures: CuTe DSL 4.8.1's `Arch` (`cutlass.base_dsl.Arch`),
/// each a `(major, minor, suffix)` triple. The suffix is `""` for a baseline
/// target, whose code runs on that architecture and the later ones, `"a"`
/// for an architecture-specific target (`sm_90a`), whose extra features
/// only that chip has and whose code runs only there, and `"f"` for a
/// family-specific target (`sm_100f`), whose features the later chips of
/// the family (the same major version) have too.
///
/// `sm_101`, `sm_101a` and `sm_101f` are the names Thor had before PTX ISA
/// 9.0 renamed it `sm_110`: they parse to the `Sm110` variants, as in CuTe
/// DSL. `sm_88` is in PTX ISA 9.0 but not in CuTe DSL 4.8.1's `Arch`, so
/// cute-rs leaves it out.
///
/// `Arch` is not ordered (`sm_90` and `sm_90a` are different targets of one
/// version): compare [`version`](Arch::version)s instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Arch {
    Sm80,
    Sm86,
    Sm87,
    Sm89,
    Sm90,
    /// Hopper with the architecture-specific features (WGMMA).
    Sm90a,
    Sm100,
    Sm100a,
    Sm100f,
    Sm103,
    Sm103a,
    Sm103f,
    Sm107,
    Sm107a,
    Sm107f,
    Sm110,
    Sm110a,
    Sm110f,
    Sm120,
    Sm120a,
    Sm120f,
    Sm121,
    Sm121a,
    Sm121f,
}

impl Arch {
    /// The most shared memory one block of this architecture can use, in
    /// bytes (CuTe DSL's `SMEM_CAPACITY_MAP`, read by
    /// `SmemAllocator.capacity_in_bytes`): the SM's shared memory less the 1
    /// KiB the system reserves per block. sm_80 and sm_87 have 163 KiB;
    /// sm_86, sm_89, sm_120 and sm_121 99 KiB; sm_90, sm_100, sm_103 and
    /// sm_110 227 KiB; sm_107 327 KiB. The `a` and `f` targets have their
    /// version's. [`check_target`](crate::validate::check_target) holds a
    /// kernel's build-time shared memory to it.
    pub const fn smem_capacity_bytes(self) -> usize {
        let kib = match self.version() {
            (8, 0) | (8, 7) => 164,
            (8, 6) | (8, 9) | (12, _) => 100,
            (10, 7) => 328,
            _ => 228,
        };
        (kib - 1) * 1024
    }
}

impl Arch {
    /// Every target, by version, the baseline, `a` and `f` targets of a
    /// version in that order.
    pub const ALL: [Arch; 24] = [
        Arch::Sm80,
        Arch::Sm86,
        Arch::Sm87,
        Arch::Sm89,
        Arch::Sm90,
        Arch::Sm90a,
        Arch::Sm100,
        Arch::Sm100a,
        Arch::Sm100f,
        Arch::Sm103,
        Arch::Sm103a,
        Arch::Sm103f,
        Arch::Sm107,
        Arch::Sm107a,
        Arch::Sm107f,
        Arch::Sm110,
        Arch::Sm110a,
        Arch::Sm110f,
        Arch::Sm120,
        Arch::Sm120a,
        Arch::Sm120f,
        Arch::Sm121,
        Arch::Sm121a,
        Arch::Sm121f,
    ];

    /// CuTe DSL's `Arch.AmpereArchs()`.
    pub const AMPERE: [Arch; 3] = [Arch::Sm80, Arch::Sm86, Arch::Sm87];
    /// CuTe DSL's `Arch.AdaArchs()`.
    pub const ADA: [Arch; 1] = [Arch::Sm89];
    /// CuTe DSL's `Arch.HopperArchs()`.
    pub const HOPPER: [Arch; 2] = [Arch::Sm90, Arch::Sm90a];
    /// CuTe DSL's `Arch.BlackwellArchs()` (the `sm_101` aliases are the
    /// `Sm110` targets, listed once): datacenter Blackwell, Rubin, Thor,
    /// and GeForce and RTX PRO Blackwell and Spark.
    pub const BLACKWELL: [Arch; 18] = [
        Arch::Sm100,
        Arch::Sm100a,
        Arch::Sm100f,
        Arch::Sm110,
        Arch::Sm110a,
        Arch::Sm110f,
        Arch::Sm103,
        Arch::Sm103a,
        Arch::Sm103f,
        Arch::Sm107,
        Arch::Sm107a,
        Arch::Sm107f,
        Arch::Sm120,
        Arch::Sm120a,
        Arch::Sm120f,
        Arch::Sm121,
        Arch::Sm121a,
        Arch::Sm121f,
    ];
    /// CuTe DSL's `Arch.RubinArchs()`.
    pub const RUBIN: [Arch; 3] = [Arch::Sm107, Arch::Sm107a, Arch::Sm107f];

    /// `(major, minor, suffix)`, CuTe DSL's value of the target.
    pub const fn triple(self) -> (u32, u32, &'static str) {
        match self {
            Arch::Sm80 => (8, 0, ""),
            Arch::Sm86 => (8, 6, ""),
            Arch::Sm87 => (8, 7, ""),
            Arch::Sm89 => (8, 9, ""),
            Arch::Sm90 => (9, 0, ""),
            Arch::Sm90a => (9, 0, "a"),
            Arch::Sm100 => (10, 0, ""),
            Arch::Sm100a => (10, 0, "a"),
            Arch::Sm100f => (10, 0, "f"),
            Arch::Sm103 => (10, 3, ""),
            Arch::Sm103a => (10, 3, "a"),
            Arch::Sm103f => (10, 3, "f"),
            Arch::Sm107 => (10, 7, ""),
            Arch::Sm107a => (10, 7, "a"),
            Arch::Sm107f => (10, 7, "f"),
            Arch::Sm110 => (11, 0, ""),
            Arch::Sm110a => (11, 0, "a"),
            Arch::Sm110f => (11, 0, "f"),
            Arch::Sm120 => (12, 0, ""),
            Arch::Sm120a => (12, 0, "a"),
            Arch::Sm120f => (12, 0, "f"),
            Arch::Sm121 => (12, 1, ""),
            Arch::Sm121a => (12, 1, "a"),
            Arch::Sm121f => (12, 1, "f"),
        }
    }

    /// The major version: 9 for sm_90a, 10 for sm_103.
    pub const fn major(self) -> u32 {
        self.triple().0
    }

    /// The minor version: 3 for sm_103.
    pub const fn minor(self) -> u32 {
        self.triple().1
    }

    /// The suffix: `""`, `"a"` or `"f"`.
    pub const fn suffix(self) -> &'static str {
        self.triple().2
    }

    /// `(major, minor)`, by which targets are ordered (CuTe DSL's `<` on
    /// `Arch`): `sm_90` and `sm_90a` have the same.
    pub const fn version(self) -> (u32, u32) {
        (self.major(), self.minor())
    }

    /// The `sm_XX` name (CuTe DSL's `to_string`).
    pub fn chip(self) -> &'static str {
        match self {
            Arch::Sm80 => "sm_80",
            Arch::Sm86 => "sm_86",
            Arch::Sm87 => "sm_87",
            Arch::Sm89 => "sm_89",
            Arch::Sm90 => "sm_90",
            Arch::Sm90a => "sm_90a",
            Arch::Sm100 => "sm_100",
            Arch::Sm100a => "sm_100a",
            Arch::Sm100f => "sm_100f",
            Arch::Sm103 => "sm_103",
            Arch::Sm103a => "sm_103a",
            Arch::Sm103f => "sm_103f",
            Arch::Sm107 => "sm_107",
            Arch::Sm107a => "sm_107a",
            Arch::Sm107f => "sm_107f",
            Arch::Sm110 => "sm_110",
            Arch::Sm110a => "sm_110a",
            Arch::Sm110f => "sm_110f",
            Arch::Sm120 => "sm_120",
            Arch::Sm120a => "sm_120a",
            Arch::Sm120f => "sm_120f",
            Arch::Sm121 => "sm_121",
            Arch::Sm121a => "sm_121a",
            Arch::Sm121f => "sm_121f",
        }
    }

    /// The name the chip had before PTX ISA 9.0 (`sm_101a` for
    /// [`Arch::Sm110a`]), for the Thor targets.
    pub fn legacy_chip(self) -> Option<&'static str> {
        match self {
            Arch::Sm110 => Some("sm_101"),
            Arch::Sm110a => Some("sm_101a"),
            Arch::Sm110f => Some("sm_101f"),
            _ => None,
        }
    }

    /// Whether this is an architecture-specific target (`sm_90a`, `sm_100a`,
    /// ...; the names ending in `a`), whose features newer architectures do
    /// not inherit and whose code runs on that architecture only.
    /// [`validate::ArchSpecific`](crate::validate::ArchSpecific) lists the
    /// targets that have such a feature.
    pub fn is_arch_specific(self) -> bool {
        self.suffix() == "a"
    }

    /// Whether this is a family-specific target (`sm_100f`, ...).
    pub fn is_family_specific(self) -> bool {
        self.suffix() == "f"
    }

    /// The architecture number: 80 for sm_80, 90 for sm_90 and sm_90a, 103
    /// for sm_103f.
    pub const fn sm(self) -> u32 {
        self.major() * 10 + self.minor()
    }

    /// CuTe DSL's `Arch.is_family_of(arch)`: whether this target can use the
    /// family-specific features of `arch`, being of the same major version,
    /// at least its minor version, and an `a` or `f` target.
    pub fn is_family_of(self, arch: Arch) -> bool {
        self.major() == arch.major() && self.minor() >= arch.minor() && matches!(self.suffix(), "a" | "f")
    }

    /// CuTe DSL's `Arch.can_run_binary_built_for(target)`: whether hardware
    /// of this architecture runs a binary built for `target`. An `a` target
    /// runs only on its own chip; `f` and baseline targets on the chips of
    /// the same major version and at least their minor version.
    pub fn can_run_binary_built_for(self, target: Arch) -> bool {
        if self == target {
            return true;
        }
        if self.major() != target.major() {
            return false;
        }
        if target.suffix() == "a" {
            return self == target;
        }
        self.minor() >= target.minor()
    }

    /// The PTX ISA version kernels are compiled with unless an instruction
    /// needs a newer one, as `10 * major + minor` (70 for 7.0): the first
    /// that has the target, from the PTX ISA's release history (sm_90 from
    /// 7.8 and sm_90a from 8.0, the `f` targets and sm_103 and sm_121 from
    /// 8.8, sm_110 (renamed from sm_101) from 9.0, sm_107 from 9.4).
    pub const fn min_ptx(self) -> u32 {
        match self {
            Arch::Sm80 => 70,
            Arch::Sm86 => 71,
            Arch::Sm87 => 74,
            Arch::Sm89 | Arch::Sm90 => 78,
            Arch::Sm90a => 80,
            Arch::Sm100 | Arch::Sm100a => 86,
            Arch::Sm120 | Arch::Sm120a => 87,
            Arch::Sm100f
            | Arch::Sm103
            | Arch::Sm103a
            | Arch::Sm103f
            | Arch::Sm120f
            | Arch::Sm121
            | Arch::Sm121a
            | Arch::Sm121f => 88,
            Arch::Sm110 | Arch::Sm110a | Arch::Sm110f => 90,
            Arch::Sm107 | Arch::Sm107a | Arch::Sm107f => 94,
        }
    }

    /// The PTX ISA version from which the chip's old name ([`legacy_chip`](Arch::legacy_chip))
    /// exists: 8.6 for sm_101 and sm_101a, 8.8 for sm_101f. PTX ISA 9.0
    /// dropped it, so 8.8 is also the newest.
    pub fn legacy_min_ptx(self) -> Option<u32> {
        match self {
            Arch::Sm110 | Arch::Sm110a => Some(86),
            Arch::Sm110f => Some(88),
            _ => None,
        }
    }

    /// The LLVM feature selecting [`Arch::min_ptx`].
    pub fn ptx_features(self) -> String {
        format!("+ptx{}", self.min_ptx())
    }

    /// The target CuTe DSL picks for a device of compute capability `major.minor`:
    /// the architecture-specific target from Hopper on (`sm_90a` for 9.0,
    /// `sm_103a` for 10.3), and the baseline before (`sm_86` for 8.6), or
    /// `None` for a device older than SM80 or one cute-rs does not know.
    /// [`best_target`] falls back from it to what a toolchain and driver
    /// handle.
    pub fn from_compute_capability(major: u32, minor: u32) -> Option<Arch> {
        Some(match (major, minor) {
            (8, 0) => Arch::Sm80,
            (8, 6) => Arch::Sm86,
            (8, 7) => Arch::Sm87,
            (8, 9) => Arch::Sm89,
            (9, 0) => Arch::Sm90a,
            (10, 0) => Arch::Sm100a,
            (10, 3) => Arch::Sm103a,
            (10, 7) => Arch::Sm107a,
            (11, 0) => Arch::Sm110a,
            (12, 0) => Arch::Sm120a,
            (12, 1) => Arch::Sm121a,
            _ => return None,
        })
    }
}

/// The CuTe DSL version cute-rs follows (`cutlass.__version__` of the
/// release it recreates).
pub const CUTE_DSL_VERSION: &str = "4.8.1";

/// The version of cute-rs itself.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The newest PTX ISA version a CUDA driver's JIT accepts, from the driver
/// version `cuDriverGetVersion` gives (`1000 * major + 10 * minor`: 12080
/// for CUDA 12.8), by the PTX ISA's release history: PTX 8.0 came with CUDA
/// 12.0, 8.5 with 12.5 (and 12.6 kept it), 8.6 with 12.7, 8.7 with 12.8, 8.8
/// with 12.9, and 9.N with 13.N; before, 7.N came with CUDA 11.N (7.0 with
/// 11.0, 7.8 with 11.8), and CUDA 10.2 had 6.5.
pub fn driver_ptx_ceiling(driver_version: u32) -> u32 {
    let (major, minor) = (driver_version / 1000, driver_version % 1000 / 10);
    match (major, minor) {
        (0..=10, _) => 65,
        (11, m) => 70 + m.min(8),
        (12, 0..=5) => 80 + minor,
        (12, 6) => 85,
        (12, m) => (79 + m).min(88),
        (13.., m) => 90 + 10 * (major - 13) + m,
    }
}

/// What a toolchain's NVPTX backend can emit: the targets it knows, by name
/// (`sm_100a`, the old `sm_101a` included), and the newest PTX ISA version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetSupport {
    pub chips: Vec<String>,
    pub max_ptx: u32,
    /// Where the answer comes from: `llc -mcpu=help`, or the table by LLVM
    /// version.
    pub source: String,
}

impl TargetSupport {
    /// What the NVPTX backend of LLVM `major` has, from the CPUs and
    /// `ptxNN` features its `llc` lists: LLVM 21 has sm_87, the sm_100,
    /// sm_101, sm_103, sm_120 and sm_121 targets (`a` and `f` too) and PTX
    /// 8.8; LLVM 22 adds sm_88 and sm_110 and PTX 9.0; LLVM 23 PTX 9.3.
    /// None has sm_107 yet.
    pub fn for_llvm(major: u32) -> TargetSupport {
        let mut chips: Vec<&str> = vec!["sm_80", "sm_86", "sm_87", "sm_89", "sm_90", "sm_90a"];
        if major >= 20 {
            chips.extend(["sm_100", "sm_100a", "sm_101", "sm_101a", "sm_120", "sm_120a"]);
        }
        if major >= 21 {
            chips.extend([
                "sm_100f", "sm_101f", "sm_103", "sm_103a", "sm_103f", "sm_120f", "sm_121", "sm_121a", "sm_121f",
            ]);
        }
        if major >= 22 {
            chips.extend(["sm_88", "sm_110", "sm_110a", "sm_110f"]);
        }
        let max_ptx = match major {
            0..=19 => 85,
            20 => 87,
            21 => 88,
            22 => 90,
            _ => 93,
        };
        TargetSupport {
            chips: chips.into_iter().map(String::from).collect(),
            max_ptx,
            source: format!("the targets of LLVM {major}"),
        }
    }

    /// Parse `llc -march=nvptx64 -mcpu=help`: the `sm_XX` CPUs and the
    /// `ptxNN` features.
    pub fn from_llc_help(text: &str) -> Option<TargetSupport> {
        let mut chips = Vec::new();
        let mut max_ptx = 0;
        for word in text.lines().filter_map(|l| l.split_whitespace().next()) {
            if word.starts_with("sm_") {
                if !chips.iter().any(|c| c == word) {
                    chips.push(word.to_string());
                }
            } else if let Some(v) = word.strip_prefix("ptx").and_then(|v| v.parse::<u32>().ok()) {
                max_ptx = max_ptx.max(v);
            }
        }
        (!chips.is_empty() && max_ptx > 0).then(|| TargetSupport { chips, max_ptx, source: "llc -mcpu=help".into() })
    }

    /// Whether the backend knows the target named `chip`.
    pub fn has(&self, chip: &str) -> bool {
        self.chips.iter().any(|c| c == chip)
    }

    /// The name to emit for `arch` and the PTX ISA versions it allows, as
    /// `(name, min, max)`: its own name, or for a Thor target that this
    /// backend knows only by its old name, `sm_101*` (PTX 8.8 at most), or
    /// `None` if it has neither.
    pub fn emit(&self, arch: Arch) -> Option<(&'static str, u32, u32)> {
        if self.has(arch.chip()) {
            return Some((arch.chip(), arch.min_ptx(), self.max_ptx));
        }
        let legacy = arch.legacy_chip()?;
        self.has(legacy).then(|| (legacy, arch.legacy_min_ptx().unwrap_or(86), self.max_ptx.min(88)))
    }
}

/// The most specific target a device of compute capability `cc` can run
/// that the toolchain (`support`) can emit and the driver (whose JIT
/// accepts PTX up to `driver_ptx`, if known) can load: the device's own
/// `a` target, then the `f` targets it runs (the newest first), then the
/// baseline ones, each only if its PTX ISA version is within both
/// ceilings. On a CC 10.3 GPU with a CUDA 12.8 driver (PTX 8.7), sm_103a,
/// sm_103f, sm_100f and sm_103 need PTX 8.8, so it is sm_100. Last come the
/// baseline targets of older architectures, whose PTX the driver
/// JIT-compiles for any later device (sm_90 on a CC 10.0 GPU whose
/// toolchain predates sm_100); a device cute-rs does not know (a newer
/// one) gets only those. `None` for a device older than SM80 or one whose
/// targets none of them handle.
pub fn best_target(cc: (u32, u32), driver_ptx: Option<u32>, support: &TargetSupport) -> Option<Arch> {
    let device = Arch::from_compute_capability(cc.0, cc.1);
    let runs = |a: Arch| match device {
        Some(d) if d.major() == a.major() => d.can_run_binary_built_for(a),
        _ => a.suffix().is_empty() && a.version() <= cc,
    };
    let rank = |a: &Arch| {
        let kind = match a.suffix() {
            "a" => 2,
            "f" => 1,
            _ => 0,
        };
        (kind, a.version())
    };
    let mut candidates: Vec<Arch> = Arch::ALL.into_iter().filter(|&a| runs(a)).collect();
    candidates.sort_by_key(|a| std::cmp::Reverse(rank(a)));
    candidates
        .into_iter()
        .find(|&a| support.emit(a).is_some_and(|(_, min, max)| min <= max && driver_ptx.is_none_or(|d| min <= d)))
}

/// A string that names no [`Arch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseArchError {
    message: String,
}

impl fmt::Display for ParseArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseArchError {}

/// Parses CuTe DSL's names (`Arch.from_string`): `sm_80` to `sm_121f`, and
/// `sm_101`, `sm_101a` and `sm_101f`, Thor's old names, as the `Sm110`
/// targets.
impl FromStr for Arch {
    type Err = ParseArchError;

    fn from_str(s: &str) -> Result<Arch, ParseArchError> {
        if let Some(arch) = Arch::ALL.into_iter().find(|a| a.chip() == s || a.legacy_chip() == Some(s)) {
            return Ok(arch);
        }
        let message = if s == "sm_88" {
            "sm_88 is in the PTX ISA but not in CuTe DSL 4.8.1's Arch, so cute-rs does not target it".to_string()
        } else {
            let names: Vec<&str> = Arch::ALL.iter().map(|a| a.chip()).collect();
            format!(
                "unknown GPU architecture `{s}`; cute-rs targets {} (and sm_101, sm_101a, sm_101f)",
                names.join(", ")
            )
        };
        Err(ParseArchError { message })
    }
}

/// The `sm_XX` name, as CuTe DSL's `to_string` (`sm_110a` for `sm_101a`).
impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.chip())
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
    /// The compile options are invalid or incomplete.
    Options(OptionError),
    /// A stage hook failed.
    Hook(String),
    /// `ptxas` rejected the PTX, or the PTX is newer than it assembles; the message has its
    /// command line and output.
    Ptxas(String),
    /// Warnings were reported with warnings-as-errors on
    /// ([`CompileOptions::warnings_as_errors`](crate::CompileOptions::warnings_as_errors),
    /// `CUTE_RS_WARNINGS_AS_ERRORS`).
    Warnings(Vec<crate::Diagnostic>),
}

impl CompileError {
    /// The source location of the failure, where one is known: the first
    /// `file:line:col: error` of `mlir-opt`'s output, the first warning's
    /// location for warnings-as-errors.
    pub fn loc(&self) -> Option<crate::SourceLoc> {
        match self {
            CompileError::Mlir(m) | CompileError::Ptxas(m) => crate::diagnostics::tool_error_location(m),
            CompileError::Warnings(ws) => ws.iter().find_map(|w| w.loc.clone()),
            _ => None,
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Tool(m) => write!(f, "could not run mlir-opt: {m}"),
            CompileError::Mlir(m) => write!(f, "mlir-opt failed:\n{m}"),
            CompileError::Unsupported(m) => f.write_str(m),
            CompileError::Options(e) => write!(f, "invalid compile options: {e}"),
            CompileError::Hook(m) => write!(f, "stage hook failed: {m}"),
            CompileError::Ptxas(m) => write!(f, "ptxas: {m}"),
            CompileError::Warnings(ws) => {
                write!(f, "warnings treated as errors:")?;
                ws.iter().try_for_each(|w| write!(f, "\n{w}"))
            }
        }
    }
}

impl std::error::Error for CompileError {}

impl From<OptionError> for CompileError {
    fn from(e: OptionError) -> CompileError {
        CompileError::Options(e)
    }
}

type PipelineKey = (PathBuf, &'static str, String, u32);
type PipelineCache = Mutex<HashMap<PipelineKey, Arc<LoweringPipeline>>>;

/// A located `mlir-opt` binary.
///
/// It remembers the lowering pipeline it derived for each target and optimization level
/// (see [`lowering_pipeline`](Self::lowering_pipeline)); clones share what it remembers.
#[derive(Clone)]
pub struct Toolchain {
    pub mlir_opt: PathBuf,
    pipelines: Arc<PipelineCache>,
    version: Arc<std::sync::OnceLock<Option<u32>>>,
    targets: Arc<std::sync::OnceLock<Option<TargetSupport>>>,
    cuda: Arc<std::sync::OnceLock<CudaTools>>,
}

impl fmt::Debug for Toolchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Toolchain").field("mlir_opt", &self.mlir_opt).finish_non_exhaustive()
    }
}

impl Toolchain {
    /// Use a specific `mlir-opt`.
    pub fn new(mlir_opt: impl Into<PathBuf>) -> Toolchain {
        Toolchain {
            mlir_opt: mlir_opt.into(),
            pipelines: Arc::default(),
            version: Arc::default(),
            targets: Arc::default(),
            cuda: Arc::default(),
        }
    }

    /// The CUDA toolkit's `ptxas`, `nvdisasm` and `cuobjdump`, found once
    /// ([`CudaTools::discover`]): in `$CUTE_RS_PTXAS` (a path to `ptxas`; a value that names
    /// no file disables it), `$CUDA_HOME/bin`, `$CUDA_PATH/bin`, `PATH` and
    /// `/usr/local/cuda/bin`. Any of them may be missing: the Mac has none, and a GPU machine
    /// may have only the driver.
    pub fn cuda_tools(&self) -> &CudaTools {
        self.cuda.get_or_init(CudaTools::discover)
    }

    /// This toolchain with these CUDA tools instead of the ones discovered
    /// (`CudaTools::default()` has none, so no cubin is made on the host).
    #[must_use]
    pub fn with_cuda_tools(self, tools: CudaTools) -> Toolchain {
        let cell = std::sync::OnceLock::new();
        let _ = cell.set(tools);
        Toolchain { cuda: Arc::new(cell), ..self }
    }

    /// The NVPTX targets and PTX ISA versions this toolchain's LLVM emits:
    /// from `llc -march=nvptx64 -mcpu=help` when an `llc` sits next to
    /// `mlir-opt`, else from [`TargetSupport::for_llvm`] by its LLVM
    /// version, else `None` (nothing is checked then). Run once and
    /// remembered.
    pub fn targets(&self) -> Option<&TargetSupport> {
        self.targets
            .get_or_init(|| {
                let llc = self.mlir_opt.with_file_name("llc");
                if llc.is_file()
                    && let Ok(out) = Command::new(&llc).args(["-march=nvptx64", "-mcpu=help"]).output()
                {
                    let text =
                        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
                    if let Some(support) = TargetSupport::from_llc_help(&text) {
                        return Some(support);
                    }
                }
                self.llvm_version().map(TargetSupport::for_llvm)
            })
            .as_ref()
    }

    /// The target name and PTX ISA version to compile for `arch` when its
    /// instructions need PTX `need` (0 for none): `arch`'s own name and the
    /// newer of `need` and its minimum, or for a Thor target this LLVM knows
    /// only as `sm_101*`, that name, with a note. Fails, naming the numbers
    /// and `why` (the instruction that raised the version), when the LLVM
    /// cannot emit the target or the PTX version.
    pub(crate) fn emission(
        &self,
        arch: Arch,
        need: u32,
        why: Option<&str>,
    ) -> Result<(&'static str, u32, Option<String>), CompileError> {
        let Some(support) = self.targets() else { return Ok((arch.chip(), need.max(arch.min_ptx()), None)) };
        let llvm = self.llvm_version().map_or_else(String::new, |v| format!(" (LLVM {v})"));
        let Some((name, min, max)) = support.emit(arch) else {
            return Err(CompileError::Unsupported(format!(
                "{}{llvm} cannot emit {}: its NVPTX backend ({}) has no such target; set CUTE_MLIR_OPT to the \
                 mlir-opt of an LLVM that has it",
                self.mlir_opt.display(),
                arch.chip(),
                support.source
            )));
        };
        let ptx = need.max(min);
        if ptx > max {
            let why = why.map_or_else(|| format!("{} itself", arch.chip()), |w| w.to_string());
            let via =
                if name == arch.chip() { String::new() } else { format!(" (as {name}, its name before PTX 9.0)") };
            return Err(CompileError::Unsupported(format!(
                "{} needs PTX ISA {}.{} on {}{via}, but {}{llvm} emits PTX ISA {}.{} at most for it: its LLVM \
                 predates PTX ISA {}.{} (set CUTE_MLIR_OPT to the mlir-opt of a newer LLVM)",
                why,
                ptx / 10,
                ptx % 10,
                arch.chip(),
                self.mlir_opt.display(),
                max / 10,
                max % 10,
                ptx / 10,
                ptx % 10
            )));
        }
        let note = (name != arch.chip()).then(|| {
            format!(
                "compiled for {} as {name} (PTX ISA {}.{}): {} has no {}; sm_101 is the same chip's name before PTX \
                 ISA 9.0, which CUDA 13 drivers may no longer load",
                arch.chip(),
                ptx / 10,
                ptx % 10,
                self.mlir_opt.display(),
                arch.chip()
            )
        });
        Ok((name, ptx, note))
    }

    /// The major version of the LLVM this `mlir-opt` belongs to, from `mlir-opt --version`
    /// (run once and remembered), or `None` if it cannot be read.
    pub fn llvm_version(&self) -> Option<u32> {
        *self.version.get_or_init(|| {
            let out = Command::new(&self.mlir_opt).arg("--version").output().ok()?;
            llvm_major_version(&String::from_utf8_lossy(&out.stdout))
        })
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
        self.run_capture(input, args).map(|(stdout, _)| stdout)
    }

    /// Run `mlir-opt` on `input`, returning its standard output and error.
    /// It fails if `mlir-opt` fails or reports an error ([`reports_error`]),
    /// even with a successful exit.
    fn run_capture(&self, input: &str, args: &[String]) -> Result<(String, String), CompileError> {
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
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if !output.status.success() || reports_error(&stderr) {
            return Err(CompileError::Mlir(stderr));
        }
        Ok((String::from_utf8_lossy(&output.stdout).into_owned(), stderr))
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
    /// [`compile_ptx`](Self::compile_ptx), it does not check the target. It
    /// compiles at the default optimization level (3), with no stage hooks.
    pub fn compile_ptx_isa(&self, mlir: &str, arch: Arch, ptx: u32) -> Result<String, CompileError> {
        self.lower(mlir, arch.chip(), ptx.max(arch.min_ptx()), super::options::DEFAULT_OPT_LEVEL, "", &[], false, &[])
            .map(|(ptx, _)| ptx)
    }

    /// Lower MLIR text to PTX for the target named `chip` (any name this LLVM knows, such as
    /// `sm_101a`, Thor's name before PTX ISA 9.0) as PTX ISA version `ptx`, with no checks.
    /// For tests that assemble a name no [`Arch`] prints.
    #[doc(hidden)]
    pub fn compile_ptx_for_chip(&self, mlir: &str, chip: &'static str, ptx: u32) -> Result<String, CompileError> {
        self.lower(mlir, chip, ptx, super::options::DEFAULT_OPT_LEVEL, "", &[], false, &[]).map(|(ptx, _)| ptx)
    }

    /// The passes that compile for `arch`, as PTX ISA version `ptx_version`, at optimization
    /// level `opt_level` (0 to 3): `gpu-lower-to-nvvm-pipeline`'s expansion by this
    /// `mlir-opt`, split at the LLVM dialect. It is derived once (an extra `mlir-opt` run) and
    /// then remembered.
    pub fn lowering_pipeline(&self, arch: Arch, ptx_version: u32, opt_level: u32) -> Arc<LoweringPipeline> {
        self.lowering_pipeline_for(arch.chip(), ptx_version.max(arch.min_ptx()), opt_level)
    }

    /// [`lowering_pipeline`](Self::lowering_pipeline) for the target named `chip` (an old name
    /// such as `sm_101a` too) and exactly PTX ISA version `ptx_version`.
    pub(crate) fn lowering_pipeline_for(
        &self,
        chip: &'static str,
        ptx_version: u32,
        opt_level: u32,
    ) -> Arc<LoweringPipeline> {
        assert!(opt_level <= 3, "cute: optimization level {opt_level} is not 0 to 3");
        let features = format!("+ptx{ptx_version}");
        let key = (self.mlir_opt.clone(), chip, features.clone(), opt_level);
        if let Some(p) = lock(&self.pipelines).get(&key) {
            return p.clone();
        }
        let request = passes::expansion_request(chip, &features, opt_level);
        let args = [format!("--pass-pipeline={request}"), "--dump-pass-pipeline".to_string()];
        let derived = match self.run_capture("module {}", &args) {
            // The dump goes to standard error.
            Ok((stdout, stderr)) => passes::from_dump(&format!("{stderr}\n{stdout}"), chip, &features, opt_level),
            Err(e) => passes::fallback(
                chip,
                &features,
                opt_level,
                format!("mlir-opt could not expand gpu-lower-to-nvvm-pipeline: {e}"),
            ),
        };
        lock(&self.pipelines).entry(key).or_insert_with(|| Arc::new(derived)).clone()
    }

    /// Lower MLIR text to PTX in the two stages of [`lowering_pipeline`](Self::lowering_pipeline),
    /// running `hooks` on the module in between.
    ///
    /// An `mlir-opt` whose LLVM predates the requested PTX ISA version ignores the request
    /// (with a warning) and writes its own default `.version`, below what the kernel's
    /// instructions need (PTX ISA 9.1, for the s2f6x2 conversions, needs LLVM 23); that is
    /// refused here, as [`CompileError::Unsupported`], rather than left to `ptxas`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower(
        &self,
        mlir: &str,
        chip: &'static str,
        ptx: u32,
        opt_level: u32,
        kernel: &str,
        hooks: &[StageHook],
        want_llvm: bool,
        link_libraries: &[PathBuf],
    ) -> Result<(String, Option<String>), CompileError> {
        let pipeline = self.lowering_pipeline_for(chip, ptx, opt_level);
        let pipeline = if link_libraries.is_empty() {
            pipeline
        } else {
            Arc::new(pipeline.with_link_libraries(link_libraries).map_err(CompileError::Unsupported)?)
        };
        let (out, llvm) = self.lower_in_stages(&pipeline, mlir, kernel, hooks, want_llvm)?;
        let want = ptx;
        match ptx_isa_version(&out) {
            Some(emitted) if emitted < want => {
                let kernel = if kernel.is_empty() { String::new() } else { format!("kernel `{kernel}` ") };
                Err(CompileError::Unsupported(format!(
                    "{kernel}needs PTX ISA {}.{} on {}, but {} wrote PTX ISA {}.{}: its LLVM predates PTX ISA \
                     {}.{} (set CUTE_MLIR_OPT to the mlir-opt of a newer LLVM)",
                    want / 10,
                    want % 10,
                    chip,
                    self.mlir_opt.display(),
                    emitted / 10,
                    emitted % 10,
                    want / 10,
                    want % 10
                )))
            }
            _ => Ok((out, llvm)),
        }
    }

    /// Lower MLIR text to PTX with `pipeline`, one from
    /// [`lowering_pipeline`](Self::lowering_pipeline) or
    /// [`LoweringPipeline::fallback_for`], in its two stages, with no stage hooks. Like
    /// [`compile_ptx`](Self::compile_ptx), it does not check the target.
    #[doc(hidden)]
    pub fn lower_with(&self, pipeline: &LoweringPipeline, mlir: &str) -> Result<String, CompileError> {
        self.lower_in_stages(pipeline, mlir, "", &[], false).map(|(ptx, _)| ptx)
    }

    fn lower_in_stages(
        &self,
        pipeline: &LoweringPipeline,
        mlir: &str,
        kernel: &str,
        hooks: &[StageHook],
        want_llvm: bool,
    ) -> Result<(String, Option<String>), CompileError> {
        super::cache::MLIR_OPT_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let started = std::time::Instant::now();
        // Printed with its locations, which the second stage would otherwise not see (they
        // carry the line information).
        let stage1 = [format!("--pass-pipeline={}", pipeline.stage1()), "--mlir-print-debuginfo".to_string()];
        let mut module = LoweredModule::new(kernel, self.run(mlir, &stage1)?);
        crate::cute_log!(
            Level::Info,
            "kernel `{kernel}`: lowering stage 1 (to the LLVM dialect) took {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        let llvm = want_llvm.then(|| module.text().to_string());
        let stage2_started = std::time::Instant::now();
        for (i, hook) in hooks.iter().enumerate() {
            hook(&mut module).map_err(|e| CompileError::Hook(format!("hook {} of {}: {e}", i + 1, hooks.len())))?;
        }
        let out =
            self.run(&module.into_text(), &[format!("--pass-pipeline={}", pipeline.stage2())]).map_err(
                |e| match e {
                    CompileError::Mlir(m) if !hooks.is_empty() => {
                        CompileError::Mlir(format!("(after {} stage hook(s) edited the module)\n{m}", hooks.len()))
                    }
                    e => e,
                },
            )?;
        crate::cute_log!(
            Level::Info,
            "kernel `{kernel}`: lowering stage 2 (to PTX, after {} hook(s)) took {:.1} ms",
            hooks.len(),
            stage2_started.elapsed().as_secs_f64() * 1e3
        );
        extract_assembly(&out)
            .map(|ptx| (ptx, llvm))
            .ok_or_else(|| CompileError::Mlir(format!("no PTX found in the pipeline output:\n{out}")))
    }

    /// Lower MLIR text to PTX with `gpu-lower-to-nvvm-pipeline` in a single `mlir-opt` run,
    /// the way cute-rs compiled before the pipeline was split. It is kept as a reference: at
    /// default options, [`compile_ptx_isa`](Self::compile_ptx_isa) and
    /// [`Kernel::compile_with`](crate::Kernel::compile_with) produce the same PTX.
    pub fn compile_ptx_reference(&self, mlir: &str, arch: Arch, ptx: u32) -> Result<String, CompileError> {
        let pipeline = format!(
            "--pass-pipeline=builtin.module(gpu.module(convert-vector-to-llvm{{use-vector-alignment=1}}),{})",
            passes::nvvm_pipeline(arch.chip(), &passes::ptx_features(arch, ptx), 3)
        );
        let out = self.run(mlir, &[pipeline])?;
        extract_assembly(&out).ok_or_else(|| CompileError::Mlir(format!("no PTX found in the pipeline output:\n{out}")))
    }
}

/// The remembered pipelines (a panic while the lock was held leaves nothing
/// half-updated, so a poisoned lock is still usable).
fn lock(pipelines: &PipelineCache) -> std::sync::MutexGuard<'_, HashMap<PipelineKey, Arc<LoweringPipeline>>> {
    pipelines.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether `mlir-opt`'s standard error reports an error of LLVM's own: a
/// line starting `error:`, as LLVM prints a diagnostic. LLVM's errors in
/// `gpu-module-to-binary`, such as an inline-assembly constraint NVPTX does
/// not know, are printed while LLVM 23's `mlir-opt` still exits with
/// success, and the PTX then lacks the statement (LLVM prints its warnings
/// the same way, `warning: ...`). MLIR's own diagnostics carry a location
/// (`<unknown>:0: error: ...`), and one that does not fail `mlir-opt` comes
/// from a conversion that was tried and then done another way: LLVM 21's
/// and 23's print "conversion of memref memory space
/// #gpu.address_space<workgroup> to integer address space failed" for the
/// shared memory of the `sgemm` example and then convert it, so those are
/// left to the exit status.
fn reports_error(stderr: &str) -> bool {
    stderr.lines().any(|line| line.starts_with("error:"))
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

/// The major version in `mlir-opt --version`'s `LLVM version 22.1.8`.
fn llvm_major_version(text: &str) -> Option<u32> {
    let rest = &text[text.find("LLVM version ")? + "LLVM version ".len()..];
    rest.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

/// The PTX ISA version of `ptx` (`10 * major + minor`), from its `.version` directive.
fn ptx_isa_version(ptx: &str) -> Option<u32> {
    let line = ptx.lines().map(str::trim).find(|l| l.starts_with(".version"))?;
    let (major, minor) = line[".version".len()..].trim().split_once('.')?;
    Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?)
}

#[cfg(test)]
mod tests {
    use super::{Arch, TargetSupport, best_target, driver_ptx_ceiling};

    /// Every CuTe DSL spelling parses and prints back; the sm_101 names are
    /// the Sm110 targets, which print as sm_110.
    #[test]
    fn arch_names_round_trip() {
        for arch in Arch::ALL {
            assert_eq!(arch.chip().parse::<Arch>(), Ok(arch));
            assert_eq!(arch.to_string(), arch.chip());
            let (major, minor, suffix) = arch.triple();
            assert_eq!(arch.chip(), format!("sm_{}{}{suffix}", major, minor));
        }
        for (old, new) in [("sm_101", Arch::Sm110), ("sm_101a", Arch::Sm110a), ("sm_101f", Arch::Sm110f)] {
            assert_eq!(old.parse::<Arch>(), Ok(new));
            assert_eq!(new.legacy_chip(), Some(old));
        }
        assert!("sm_88".parse::<Arch>().unwrap_err().to_string().contains("not in CuTe DSL 4.8.1's Arch"));
        assert!("sm_75".parse::<Arch>().unwrap_err().to_string().contains("unknown GPU architecture"));
        let mut all: Vec<Arch> = Arch::ALL.to_vec();
        all.dedup();
        assert_eq!(all.len(), 24);
        assert_eq!(Arch::BLACKWELL.len() + Arch::HOPPER.len() + Arch::ADA.len() + Arch::AMPERE.len(), 24);
        assert!(Arch::RUBIN.iter().all(|a| Arch::BLACKWELL.contains(a)));
    }

    /// CuTe DSL's `is_family_of` and `can_run_binary_built_for`, ported
    /// literally on the `(major, minor, suffix)` triples, agree with Arch's
    /// over every pair.
    #[test]
    fn family_and_binary_compatibility_follow_cute_dsl() {
        type T = (u32, u32, &'static str);
        let is_family_of = |s: T, a: T| s.0 == a.0 && s.1 >= a.1 && ["a", "f"].contains(&s.2);
        let can_run = |s: T, t: T| {
            if s == t {
                return true;
            }
            if s.0 != t.0 {
                return false;
            }
            if t.2 == "a" {
                return s == t;
            }
            s.1 >= t.1
        };
        for a in Arch::ALL {
            for b in Arch::ALL {
                assert_eq!(a.is_family_of(b), is_family_of(a.triple(), b.triple()), "{a} family of {b}");
                assert_eq!(a.can_run_binary_built_for(b), can_run(a.triple(), b.triple()), "{a} runs {b}");
            }
        }
        assert!(Arch::Sm103f.is_family_of(Arch::Sm100f) && !Arch::Sm103.is_family_of(Arch::Sm100f));
        assert!(
            Arch::Sm103a.can_run_binary_built_for(Arch::Sm100f) && !Arch::Sm103a.can_run_binary_built_for(Arch::Sm100a)
        );
        assert!(
            Arch::Sm121a.can_run_binary_built_for(Arch::Sm120f) && !Arch::Sm110a.can_run_binary_built_for(Arch::Sm100)
        );
    }

    /// The first PTX ISA version of each target, from the PTX ISA's release
    /// history.
    #[test]
    fn ptx_minimums() {
        let want = [
            (Arch::Sm80, 70),
            (Arch::Sm86, 71),
            (Arch::Sm87, 74),
            (Arch::Sm89, 78),
            (Arch::Sm90, 78),
            (Arch::Sm90a, 80),
            (Arch::Sm100, 86),
            (Arch::Sm100a, 86),
            (Arch::Sm100f, 88),
            (Arch::Sm103, 88),
            (Arch::Sm103a, 88),
            (Arch::Sm103f, 88),
            (Arch::Sm107, 94),
            (Arch::Sm107a, 94),
            (Arch::Sm107f, 94),
            (Arch::Sm110, 90),
            (Arch::Sm110a, 90),
            (Arch::Sm110f, 90),
            (Arch::Sm120, 87),
            (Arch::Sm120a, 87),
            (Arch::Sm120f, 88),
            (Arch::Sm121, 88),
            (Arch::Sm121a, 88),
            (Arch::Sm121f, 88),
        ];
        assert_eq!(want.len(), Arch::ALL.len());
        for (arch, ptx) in want {
            assert_eq!(arch.min_ptx(), ptx, "{arch}");
        }
        assert_eq!((Arch::Sm110a.legacy_min_ptx(), Arch::Sm110f.legacy_min_ptx()), (Some(86), Some(88)));
        // The driver's JIT: PTX 8.0 with CUDA 12.0 ... 8.5 with 12.5 and
        // 12.6, 8.6 with 12.7, 8.7 with 12.8, 8.8 with 12.9, 9.N with 13.N.
        let drivers = [
            (10020, 65),
            (11000, 70),
            (11040, 74),
            (11080, 78),
            (12000, 80),
            (12050, 85),
            (12060, 85),
            (12070, 86),
            (12080, 87),
            (12090, 88),
        ];
        for (driver, ptx) in drivers.into_iter().chain([(13000, 90), (13020, 92), (13040, 94)]) {
            assert_eq!(driver_ptx_ceiling(driver), ptx, "driver {driver}");
        }
    }

    /// The device mapping and its fallbacks, without a GPU.
    #[test]
    fn device_mapping_and_best_targets() {
        let cc = [
            ((8, 0), Arch::Sm80),
            ((8, 6), Arch::Sm86),
            ((8, 7), Arch::Sm87),
            ((8, 9), Arch::Sm89),
            ((9, 0), Arch::Sm90a),
            ((10, 0), Arch::Sm100a),
            ((10, 3), Arch::Sm103a),
            ((10, 7), Arch::Sm107a),
            ((11, 0), Arch::Sm110a),
            ((12, 0), Arch::Sm120a),
            ((12, 1), Arch::Sm121a),
        ];
        for ((major, minor), arch) in cc {
            assert_eq!(Arch::from_compute_capability(major, minor), Some(arch));
        }
        assert_eq!(Arch::from_compute_capability(7, 5), None);
        let (l21, l22) = (TargetSupport::for_llvm(21), TargetSupport::for_llvm(22));
        // The SM89 GPU: sm_89 with any toolchain and driver.
        assert_eq!(best_target((8, 9), Some(92), &l21), Some(Arch::Sm89));
        assert_eq!(best_target((9, 0), Some(87), &l21), Some(Arch::Sm90a));
        // CC 10.3: sm_103a with a CUDA 12.9 driver; with 12.8 (PTX 8.7)
        // sm_103a, sm_103f, sm_100f and sm_103 need 8.8, so sm_100.
        assert_eq!(best_target((10, 3), Some(88), &l21), Some(Arch::Sm103a));
        assert_eq!(best_target((10, 3), Some(87), &l21), Some(Arch::Sm100));
        // An old toolchain caps it the same way.
        assert_eq!(best_target((10, 3), None, &TargetSupport::for_llvm(20)), Some(Arch::Sm100));
        assert_eq!(best_target((12, 1), Some(87), &l21), Some(Arch::Sm120));
        assert_eq!(best_target((12, 1), None, &l21), Some(Arch::Sm121a));
        // Thor: sm_110a where the toolchain has it, its old name sm_101a
        // (PTX 8.6) where it does not.
        assert_eq!(best_target((11, 0), Some(90), &l22), Some(Arch::Sm110a));
        assert_eq!(l21.emit(Arch::Sm110a), Some(("sm_101a", 86, 88)));
        assert_eq!(l21.emit(Arch::Sm110f), Some(("sm_101f", 88, 88)));
        assert_eq!(l22.emit(Arch::Sm110a), Some(("sm_110a", 90, 90)));
        assert_eq!(best_target((11, 0), Some(88), &l21), Some(Arch::Sm110a));
        // Rubin: no LLVM emits sm_107 yet, so the newest family target of
        // its family that one does.
        assert_eq!(best_target((10, 7), None, &TargetSupport::for_llvm(23)), Some(Arch::Sm103f));
        assert_eq!(l22.emit(Arch::Sm107a), None);
        // Baseline PTX of an older architecture JIT-compiles on later
        // devices: a toolchain without sm_100 still serves a CC 10.0 GPU, and
        // a device cute-rs does not know gets the newest baseline before it.
        assert_eq!(best_target((10, 0), None, &TargetSupport::for_llvm(19)), Some(Arch::Sm90));
        assert_eq!(best_target((13, 0), None, &l22), Some(Arch::Sm121));
        assert_eq!(best_target((8, 8), None, &l22), Some(Arch::Sm87));
        assert_eq!(best_target((7, 5), None, &l22), None);
    }

    /// `llc -mcpu=help` is parsed for the CPUs and the newest ptxNN feature.
    #[test]
    fn reads_llc_targets() {
        let text = "Available CPUs for this target:\n\n  sm_100  - Select the sm_100 processor.\n  sm_90a  - x\n\n\
                    Available features for this target:\n\n  ptx87   - Use PTX version 87.\n  ptx88   - Use PTX \
                    version 88.\n  sm_100  - Target SM 100.\n";
        let support = TargetSupport::from_llc_help(text).unwrap();
        assert_eq!((support.chips.clone(), support.max_ptx), (vec!["sm_100".to_string(), "sm_90a".to_string()], 88));
        assert!(support.has("sm_90a") && !support.has("sm_110a"));
        assert_eq!(TargetSupport::from_llc_help("nothing"), None);
    }

    #[test]
    fn decodes_mlir_string_escapes() {
        let text = r#"gpu.binary @k [#gpu.object<#nvvm.target, assembly = "//\0A.version 7.0\0A\09ld \22x\22 \\">]"#;
        assert_eq!(super::extract_assembly(text).unwrap(), "//\n.version 7.0\n\tld \"x\" \\");
    }

    /// LLVM's errors are found in what `mlir-opt` prints; MLIR's located
    /// diagnostics (left to the exit status), the pass pipeline it dumps,
    /// warnings and remarks are not counted.
    #[test]
    fn finds_reported_errors() {
        for stderr in [
            "error: unknown asm constraint 'x'\n",
            "warning: x\nerror: constraint 'n' expects an integer constant expression\n",
        ] {
            assert!(super::reports_error(stderr), "{stderr:?}");
        }
        for stderr in [
            "",
            "<unknown>:0: error: conversion of memref memory space #gpu.address_space<workgroup> to integer address \
             space failed. Consider adding memory space conversions.\n",
            "Pass Manager with 23 passes:\nbuiltin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm))\n",
            "warning: unused\n",
            "remark: no errors here\n",
        ] {
            assert!(!super::reports_error(stderr), "{stderr:?}");
        }
    }

    /// The major version is read from `mlir-opt --version`.
    #[test]
    fn reads_the_llvm_version() {
        assert_eq!(super::llvm_major_version("LLVM (http://llvm.org/):\n  LLVM version 22.1.8\n"), Some(22));
        assert_eq!(super::llvm_major_version("Homebrew LLVM version 23.1.4\n  Optimized build."), Some(23));
        assert_eq!(super::llvm_major_version("mlir-opt"), None);
    }

    /// The `.version` directive is read as `10 * major + minor`; PTX without one has none.
    #[test]
    fn reads_the_ptx_isa_version() {
        let ptx = "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 9.1\n.target sm_100a\n";
        assert_eq!(super::ptx_isa_version(ptx), Some(91));
        assert_eq!(super::ptx_isa_version("\t.version 8.6\n.target sm_100a"), Some(86));
        assert_eq!(super::ptx_isa_version(".target sm_80\n"), None);
    }
}

//! Errors and diagnostics: [`Error`], one type over every failure cute-rs
//! reports (CuTe DSL's `DSLUserCodeError`, `DSLRuntimeError`,
//! `CompilerDiagnosticError` and `DSLCudaRuntimeError`), each with an owned
//! [`SourceLoc`] and a code frame; [`Diagnostic`], the compiler's warnings
//! and remarks; and [`ResourceUsage`], a kernel's registers, spills, stack
//! and memory as `ptxas -v`, the JIT log or the driver report them.
//!
//! cute-rs is Rust, so a panic's backtrace is the standard library's:
//! `RUST_BACKTRACE=1` stands in for CuTe DSL's `CUTE_DSL_SHOW_STACKTRACE`,
//! and messages start with the user's `file:line:col` where CuTe DSL filters
//! its own frames out (`CUTE_DSL_FILTER_STACKTRACE`).

use std::fmt;
use std::path::{Path, PathBuf};

use crate::backend::{CompileError, OptionError};
#[cfg(feature = "cuda")]
use crate::cuda::CudaError;
use crate::ir::Loc;
use crate::sim::SimError;
use crate::validate::ValidationError;

/// A source location, owned: from a Rust [`Location`](std::panic::Location)
/// or parsed out of a tool's output.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceLoc {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl From<Loc> for SourceLoc {
    fn from(loc: Loc) -> SourceLoc {
        SourceLoc { file: loc.file().to_string(), line: loc.line(), column: loc.column() }
    }
}

impl fmt::Display for SourceLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

impl SourceLoc {
    /// The file, if it can be read: as given (Rust locations are relative to
    /// where cargo ran), then under `CARGO_MANIFEST_DIR`.
    fn source(&self) -> Option<String> {
        let given = Path::new(&self.file);
        std::fs::read_to_string(given).ok().or_else(|| {
            let dir = std::env::var_os("CARGO_MANIFEST_DIR")?;
            std::fs::read_to_string(PathBuf::from(dir).join(given)).ok()
        })
    }

    /// The source line with a caret under the column, if the file can be
    /// read:
    ///
    /// ```text
    ///    --> tests/diagnostics.rs:12:5
    ///     |
    ///  12 |     printf("%d\n", &[x.into()]);
    ///     |     ^
    /// ```
    pub fn code_frame(&self) -> Option<String> {
        let text = self.source()?;
        let line = text.lines().nth((self.line as usize).checked_sub(1)?)?;
        let num = self.line.to_string();
        let pad = " ".repeat(num.len());
        // The caret under the column's character (columns count characters
        // from 1); tabs are kept so it lines up.
        let lead: String = line
            .chars()
            .take((self.column as usize).saturating_sub(1))
            .map(|c| if c == '\t' { '\t' } else { ' ' })
            .collect();
        Some(format!("{pad}--> {self}\n{pad} |\n{num} | {line}\n{pad} | {lead}^"))
    }
}

/// The first `file:line:col: error` of a tool's output (as `mlir-opt`
/// prints a diagnostic at a `loc("file":line:col)`), as a location.
pub(crate) fn tool_error_location(output: &str) -> Option<SourceLoc> {
    output.lines().find_map(|l| {
        let i = l.find(": error")?;
        let head = l[..i].trim_start_matches("loc(");
        let (rest, col) = head.rsplit_once(':')?;
        let (file, line) = rest.rsplit_once(':')?;
        let file = file.trim().trim_matches('"');
        let (line, column) = (line.trim().parse().ok()?, col.trim().trim_end_matches(')').parse().ok()?);
        (!file.is_empty() && !file.contains('<')).then(|| SourceLoc { file: file.to_string(), line, column })
    })
}

/// A device assertion that failed: [`testing::RuntimeAssertion::verify`](crate::testing::RuntimeAssertion::verify).
#[derive(Clone, Debug)]
pub struct AssertionFailure {
    pub message: String,
    pub loc: SourceLoc,
    /// The assertion's index in the [`RuntimeAssertion`](crate::testing::RuntimeAssertion).
    pub index: usize,
}

impl fmt::Display for AssertionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: device assertion {} failed: {}", self.loc, self.index, self.message)
    }
}

impl std::error::Error for AssertionFailure {}

/// Any error cute-rs reports. The existing error types convert into it, so
/// `?` gathers them:
///
/// ```no_run
/// # use cute::prelude::*;
/// fn run(k: &Kernel, tc: &Toolchain) -> Result<(), cute::Error> {
///     let compiled = k.compile_with(tc, &CompileOptions::new().gpu_arch(Arch::Sm80))?;
///     let _ = compiled;
///     Ok(())
/// }
/// ```
///
/// `Display` prints the inner error's message; the alternate form (`{:#}`)
/// and [`render`](Error::render) add a code frame of the source line.
#[derive(Debug)]
pub enum Error {
    Validation(ValidationError),
    Sim(SimError),
    Compile(CompileError),
    Options(OptionError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
    Assertion(AssertionFailure),
}

impl Error {
    /// Where the error comes from, if known: the user's line for validation,
    /// simulation and assertion errors, the line `mlir-opt` names for an
    /// `mlir-opt` failure.
    pub fn loc(&self) -> Option<SourceLoc> {
        match self {
            Error::Validation(e) => e.loc.map(SourceLoc::from),
            Error::Sim(e) => e.loc.map(SourceLoc::from),
            Error::Compile(e) => e.loc(),
            Error::Assertion(a) => Some(a.loc.clone()),
            Error::Options(_) => None,
            #[cfg(feature = "cuda")]
            Error::Cuda(_) => None,
        }
    }

    /// The message, then the code frame when the source can be read.
    pub fn render(&self) -> String {
        let mut s = self.to_string();
        if let Some(frame) = self.loc().and_then(|l| l.code_frame()) {
            s.push('\n');
            s.push_str(&frame);
        }
        s
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            return f.write_str(&self.render());
        }
        match self {
            Error::Validation(e) => e.fmt(f),
            Error::Sim(e) => e.fmt(f),
            Error::Compile(e) => e.fmt(f),
            Error::Options(e) => e.fmt(f),
            #[cfg(feature = "cuda")]
            Error::Cuda(e) => e.fmt(f),
            Error::Assertion(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Validation(e) => Some(e),
            Error::Sim(e) => Some(e),
            Error::Compile(e) => Some(e),
            Error::Options(e) => Some(e),
            #[cfg(feature = "cuda")]
            Error::Cuda(e) => Some(e),
            Error::Assertion(e) => Some(e),
        }
    }
}

macro_rules! from_error {
    ($($t:ty => $v:ident),*) => {$(
        impl From<$t> for Error {
            fn from(e: $t) -> Error {
                Error::$v(e)
            }
        }
    )*};
}
from_error!(ValidationError => Validation, SimError => Sim, CompileError => Compile, OptionError => Options,
    AssertionFailure => Assertion);
#[cfg(feature = "cuda")]
from_error!(CudaError => Cuda);

/// How serious a [`Diagnostic`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Error,
    Warning,
    Remark,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Remark => "remark",
        })
    }
}

/// The diagnostic categories: `ptx` (the assembler's resource report),
/// `target` (PTX ISA raises and target fallbacks), `loop` (LLVM's loop
/// remarks, unavailable through `mlir-opt`), `nvvm` (CuTe DSL's NVVM
/// checks, which cute-rs's simulator does at run time instead), and `sim`
/// (the simulator's non-fatal findings).
pub const CATEGORIES: &[&str] = &["ptx", "target", "loop", "nvvm", "sim"];

/// A warning or remark from the compiler or the simulator (CuTe DSL's
/// compiler diagnostics), at a source location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub category: String,
    pub message: String,
    pub loc: Option<SourceLoc>,
}

/// `file:line:col: warning[ptx]: message`.
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(loc) = &self.loc {
            write!(f, "{loc}: ")?;
        }
        write!(f, "{}[{}]: {}", self.severity, self.category, self.message)
    }
}

impl Diagnostic {
    pub fn new(severity: Severity, category: &str, message: impl Into<String>, loc: Option<SourceLoc>) -> Diagnostic {
        Diagnostic { severity, category: category.to_string(), message: message.into(), loc }
    }

    /// Whether `filter` selects the diagnostic: `|`-separated alternatives,
    /// each a category or a substring of the message (CuTe DSL's filters are
    /// regular expressions; cute-rs, without a regex engine, matches text).
    pub fn matches(&self, filter: &str) -> bool {
        filter
            .split('|')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .any(|a| self.category.eq_ignore_ascii_case(a) || self.message.contains(a))
    }

    /// The diagnostic as a YAML document (CuTe DSL's `RemarkOutput`).
    pub fn to_yaml(&self) -> String {
        let q = |s: &str| format!("'{}'", s.replace('\'', "''").replace('\n', " "));
        let mut y = format!(
            "--- !{}\nCategory: {}\nMessage: {}\n",
            match self.severity {
                Severity::Error => "Failure",
                Severity::Warning => "Warning",
                Severity::Remark => "Remark",
            },
            q(&self.category),
            q(&self.message)
        );
        if let Some(l) = &self.loc {
            y.push_str(&format!("DebugLoc: {{ File: {}, Line: {}, Column: {} }}\n", q(&l.file), l.line, l.column));
        }
        y.push_str("...\n");
        y
    }
}

/// A kernel's resources: what `ptxas -v` or the JIT's verbose info log
/// reports ([`from_ptxas_log`](ResourceUsage::from_ptxas_log)), or what the
/// driver's `cuFuncGetAttribute` gives
/// ([`CudaFunction::resource_usage`](crate::cuda::CudaFunction::resource_usage)).
/// A field is `None` where the source does not say.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceUsage {
    pub registers: Option<u32>,
    /// The stack frame (per-thread local memory), in bytes.
    pub stack_bytes: Option<u32>,
    pub spill_store_bytes: Option<u32>,
    pub spill_load_bytes: Option<u32>,
    /// Static shared memory, in bytes.
    pub shared_bytes: Option<u32>,
    /// Constant memory (every bank), in bytes.
    pub constant_bytes: Option<u32>,
    pub max_threads_per_block: Option<u32>,
    /// The PTX and binary (SASS) versions, as the driver numbers them (`80`
    /// for 8.0).
    pub ptx_version: Option<u32>,
    pub binary_version: Option<u32>,
}

impl ResourceUsage {
    /// Whether registers were spilled to local memory.
    pub fn spills(&self) -> bool {
        self.spill_store_bytes.unwrap_or(0) > 0 || self.spill_load_bytes.unwrap_or(0) > 0
    }

    /// The report for the entry `kernel` in `ptxas -v` output:
    ///
    /// ```text
    /// ptxas info    : Compiling entry function 'k' for 'sm_89'
    /// ptxas info    : Function properties for k
    ///     8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads
    /// ptxas info    : Used 16 registers, used 0 barriers, 8 bytes cumulative stack size, 360 bytes cmem[0]
    /// ```
    ///
    /// The JIT's log has the same lines without the "Compiling entry
    /// function" one, and with the properties' second line prefixed
    /// `ptxas         .`.
    ///
    /// `None` if the log has no report for it.
    pub fn from_ptxas_log(log: &str, kernel: &str) -> Option<ResourceUsage> {
        let mut usage = ResourceUsage::default();
        let mut found = false;
        let mut lines = log.lines().peekable();
        let mut current = false;
        while let Some(line) = lines.next() {
            if let Some(rest) = line.split_once("Compiling entry function '").map(|(_, r)| r) {
                current = rest.split('\'').next() == Some(kernel);
                continue;
            }
            if let Some(name) = line.split_once("Function properties for ").map(|(_, r)| r.trim()) {
                // The JIT's log has no "Compiling entry function" line: the
                // properties name the function whose "Used" line follows.
                current = name == kernel;
                if name == kernel
                    && let Some(next) = lines.peek()
                {
                    found = true;
                    for (n, what) in numbers(next) {
                        match what.as_str() {
                            "bytes stack frame" => usage.stack_bytes = Some(n),
                            "bytes spill stores" => usage.spill_store_bytes = Some(n),
                            "bytes spill loads" => usage.spill_load_bytes = Some(n),
                            _ => {}
                        }
                    }
                }
                continue;
            }
            if current && line.contains("Used ") && line.contains("registers") {
                found = true;
                for (n, what) in numbers(line) {
                    match what.as_str() {
                        "registers" => usage.registers = Some(n),
                        w if w.starts_with("bytes smem") => {
                            usage.shared_bytes = Some(usage.shared_bytes.unwrap_or(0) + n)
                        }
                        w if w.starts_with("bytes cmem") => {
                            usage.constant_bytes = Some(usage.constant_bytes.unwrap_or(0) + n)
                        }
                        _ => {}
                    }
                }
                usage.shared_bytes.get_or_insert(0);
                current = false;
            }
        }
        found.then_some(usage)
    }

    /// [`from_ptxas_log`](ResourceUsage::from_ptxas_log) of the JIT's info
    /// log (`CU_JIT_INFO_LOG_BUFFER` with `CU_JIT_LOG_VERBOSE`), which the
    /// driver's assembler writes in the same form.
    pub fn from_jit_log(log: &str, kernel: &str) -> Option<ResourceUsage> {
        ResourceUsage::from_ptxas_log(log, kernel)
    }
}

/// `(number, what follows it up to the next comma)` for each number in
/// `line`: `[(16, "registers"), (360, "bytes cmem[0]")]`.
fn numbers(line: &str) -> Vec<(u32, String)> {
    line.split(',')
        .filter_map(|part| {
            // The number is the first word that is one (the JIT log prefixes
            // continuation lines with `ptxas         .`).
            let words: Vec<&str> = part.split_whitespace().collect();
            let at = words.iter().position(|w| w.parse::<u32>().is_ok())?;
            Some((words[at].parse().ok()?, words[at + 1..].join(" ")))
        })
        .collect()
}

/// Which diagnostics a compile reports: CuTe DSL's `warnings`,
/// `warnings{cat,...}`, `remarks`, `remarks{cat,...}`. `None` is off; an
/// empty list is every category.
pub(crate) fn wants(selection: &Option<Vec<String>>, category: &str) -> bool {
    selection.as_ref().is_some_and(|cats| cats.is_empty() || cats.iter().any(|c| c == category))
}

/// The categories in a `{a,b}` selector list, checked.
pub(crate) fn parse_categories(list: Option<&str>, shown: &str) -> Result<Vec<String>, OptionError> {
    let Some(list) = list else { return Ok(Vec::new()) };
    list.split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(|c| {
            if CATEGORIES.contains(&c) {
                Ok(c.to_string())
            } else {
                Err(OptionError::new(format!(
                    "`{shown}`: `{c}` is not a diagnostic category (they are {})",
                    CATEGORIES.join(", ")
                )))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_locations() {
        let out = "tests/diagnostics.rs:120:9: error: invalid constraint\nnote: see current operation";
        assert_eq!(
            tool_error_location(out),
            Some(SourceLoc { file: "tests/diagnostics.rs".into(), line: 120, column: 9 })
        );
        let quoted = "loc(\"src/x.rs\":3:4): error: bad";
        assert_eq!(tool_error_location(quoted).map(|l| l.line), Some(3));
        assert_eq!(tool_error_location("<stdin>:3:4: error: bad"), None);
        assert_eq!(tool_error_location("no error here"), None);
    }

    #[test]
    fn filters() {
        let d = Diagnostic::new(Severity::Warning, "ptx", "spilled 8 bytes", None);
        assert!(d.matches("ptx") && d.matches("Memory|spilled") && !d.matches("loop"));
        assert_eq!(d.to_string(), "warning[ptx]: spilled 8 bytes");
        assert!(d.to_yaml().starts_with("--- !Warning\nCategory: 'ptx'"));
        assert!(wants(&Some(vec![]), "ptx") && wants(&Some(vec!["ptx".into()]), "ptx"));
        assert!(!wants(&None, "ptx") && !wants(&Some(vec!["loop".into()]), "ptx"));
        assert_eq!(parse_categories(Some("ptx, loop"), "x").unwrap(), ["ptx", "loop"]);
        assert!(
            parse_categories(Some("bogus"), "remarks{bogus}").unwrap_err().to_string().contains("not a diagnostic")
        );
    }
}

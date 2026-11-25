//! The lowering pipeline: MLIR's `gpu-lower-to-nvvm-pipeline`, expanded into its passes and
//! split in two at the LLVM dialect.
//!
//! The pass list is not hard-coded for one LLVM version. It is what the installed `mlir-opt`
//! expands `gpu-lower-to-nvvm-pipeline{...}` to with `--dump-pass-pipeline` (LLVM 21, 22 and
//! 23 differ in their default pass options), parsed and checked for the passes the split
//! relies on. If the dump cannot be parsed, MLIR 21's list
//! (`mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp`) is used instead.
//!
//! The first stage ends with the kernel as `llvm.func` inside its `gpu.module`; the second
//! runs `gpu-to-llvm` and `gpu-module-to-binary`, which serialises the module to PTX. Stage
//! hooks run in between (see [`LoweredModule`](super::LoweredModule)).

use super::Arch;
use super::options::debug_log;

/// Lowered first: vector loads and stores keep their (proven) vector alignment, which the
/// pipeline's own lowering would reduce to element alignment, losing the 128-bit accesses.
const VECTOR_FIRST: &str = "gpu.module(convert-vector-to-llvm{use-vector-alignment=1})";

/// One element of a pass pipeline: a pass with its options, or passes nested on an op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Element {
    Pass { name: String, options: Option<String> },
    Nest { op: String, body: Vec<Element> },
}

impl Element {
    fn is_pass(&self, pass: &str) -> bool {
        matches!(self, Element::Pass { name, .. } if name == pass)
    }

    fn render(&self, out: &mut String) {
        match self {
            Element::Pass { name, options: None } => out.push_str(name),
            Element::Pass { name, options: Some(o) } => {
                out.push_str(name);
                out.push('{');
                out.push_str(o);
                out.push('}');
            }
            Element::Nest { op, body } => {
                out.push_str(op);
                out.push('(');
                out.push_str(&render(body));
                out.push(')');
            }
        }
    }

    fn names(&self, out: &mut Vec<String>) {
        match self {
            Element::Pass { name, .. } => out.push(name.clone()),
            Element::Nest { op, body } => {
                let mut inner = Vec::new();
                body.iter().for_each(|e| e.names(&mut inner));
                out.extend(inner.into_iter().map(|n| format!("{op}({n})")));
            }
        }
    }
}

fn render(elements: &[Element]) -> String {
    let mut out = String::new();
    for (i, e) in elements.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        e.render(&mut out);
    }
    out
}

/// The passes that compile a kernel to PTX, in two `mlir-opt` runs. Get one from
/// [`Toolchain::lowering_pipeline`](super::Toolchain::lowering_pipeline).
#[derive(Clone, Debug)]
pub struct LoweringPipeline {
    stage1: Vec<Element>,
    stage2: Vec<Element>,
    fallback: Option<String>,
}

impl LoweringPipeline {
    /// The first stage, as a `--pass-pipeline` value: up to the kernel as `llvm.func` inside
    /// its `gpu.module`.
    pub fn stage1(&self) -> String {
        format!("builtin.module({VECTOR_FIRST},{})", render(&self.stage1))
    }

    /// The second stage, as a `--pass-pipeline` value: `gpu-to-llvm`, `gpu-module-to-binary`
    /// (to PTX) and the host-side cleanups after it.
    pub fn stage2(&self) -> String {
        format!("builtin.module({})", render(&self.stage2))
    }

    /// The names of the passes of both stages, in order; nested passes are written
    /// `gpu.module(name)`.
    pub fn pass_names(&self) -> Vec<String> {
        let mut out = vec!["gpu.module(convert-vector-to-llvm)".to_string()];
        self.stage1.iter().chain(&self.stage2).for_each(|e| e.names(&mut out));
        out
    }

    /// This pipeline with `libraries` (LLVM bitcode files) linked into the GPU module:
    /// `nvvm-attach-target`'s `l=` option, so that `gpu-module-to-binary` links them before
    /// optimizing (calls of their functions can then be inlined). MLIR's convenience pipeline
    /// has no such option, which is why the expansion is used. Paths must not contain
    /// whitespace, commas, quotes or braces (the pass-option syntax); such a path is refused.
    pub fn with_link_libraries(&self, libraries: &[std::path::PathBuf]) -> Result<LoweringPipeline, String> {
        if libraries.is_empty() {
            return Ok(self.clone());
        }
        let mut list = Vec::new();
        for path in libraries {
            let text = path.to_str().ok_or_else(|| format!("bitcode path {} is not UTF-8", path.display()))?;
            if text.is_empty() || text.chars().any(|c| c.is_whitespace() || matches!(c, ',' | '"' | '{' | '}' | '\'')) {
                return Err(format!(
                    "bitcode path {text:?} cannot be passed to nvvm-attach-target: it must not contain whitespace, \
                     commas, quotes or braces"
                ));
            }
            list.push(text.to_string());
        }
        let mut out = self.clone();
        let attach = out
            .stage1
            .iter_mut()
            .find(|e| e.is_pass("nvvm-attach-target"))
            .ok_or("the lowering pipeline has no nvvm-attach-target pass")?;
        let Element::Pass { options, .. } = attach else { unreachable!("found as a pass") };
        let kept: Vec<String> = pass_options(options.as_deref().unwrap_or(""))
            .into_iter()
            .filter(|(k, _)| k != "l")
            .map(|(k, v)| if v.is_empty() && !k.is_empty() { format!("{k}=") } else { format!("{k}={v}") })
            .collect();
        *options = Some(format!("{} l={}", kept.join(" "), list.join(",")));
        Ok(out)
    }

    /// Why MLIR 21's built-in pass list is used instead of the installed `mlir-opt`'s own
    /// expansion, or `None` when the expansion is used.
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    /// MLIR 21's pass list for `arch`, PTX ISA version `ptx_version` and optimization level
    /// `opt_level` (0 to 3): what [`Toolchain::lowering_pipeline`](super::Toolchain::lowering_pipeline)
    /// falls back to when the installed `mlir-opt`'s own expansion cannot be parsed. It is
    /// public for the tests that run it through every installed `mlir-opt`
    /// ([`Toolchain::lower_with`](super::Toolchain::lower_with)), since a mistake in it would
    /// otherwise only show on a machine whose expansion fails to parse.
    #[doc(hidden)]
    pub fn fallback_for(arch: Arch, ptx_version: u32, opt_level: u32) -> LoweringPipeline {
        assert!(opt_level <= 3, "cute: optimization level {opt_level} is not 0 to 3");
        let reason = "requested with LoweringPipeline::fallback_for".to_string();
        fallback(arch.chip(), &ptx_features(arch, ptx_version), opt_level, reason)
    }
}

/// The LLVM feature selecting PTX ISA version `ptx_version`, or `arch`'s minimum if that is
/// newer.
pub(crate) fn ptx_features(arch: Arch, ptx_version: u32) -> String {
    format!("+ptx{}", ptx_version.max(arch.min_ptx()))
}

/// `gpu-lower-to-nvvm-pipeline` with the options cute-rs compiled with before the pipeline was
/// split.
pub(crate) fn nvvm_pipeline(chip: &str, features: &str, opt_level: u32) -> String {
    format!(
        "gpu-lower-to-nvvm-pipeline{{cubin-chip={chip} cubin-features={features} cubin-format=isa \
         kernel-bare-ptr-calling-convention=1 opt-level={opt_level}}}"
    )
}

/// The pipeline whose expansion is asked for.
pub(crate) fn expansion_request(chip: &str, features: &str, opt_level: u32) -> String {
    format!("builtin.module({})", nvvm_pipeline(chip, features, opt_level))
}

/// The pipeline from `mlir-opt --dump-pass-pipeline` output for [`expansion_request`], or the
/// fallback list if it cannot be parsed or lacks a pass the split relies on.
pub(crate) fn from_dump(dump: &str, chip: &str, features: &str, opt_level: u32) -> LoweringPipeline {
    match parse_dump(dump).and_then(|elements| split(elements, chip, features, opt_level)) {
        Ok((stage1, stage2)) => LoweringPipeline { stage1, stage2, fallback: None },
        Err(reason) => fallback(chip, features, opt_level, reason),
    }
}

/// MLIR 21's expansion of `gpu-lower-to-nvvm-pipeline`, with only the options cute-rs sets.
pub(crate) fn fallback(chip: &str, features: &str, opt_level: u32, reason: String) -> LoweringPipeline {
    debug_log(|| format!("using MLIR 21's lowering pass list: {reason}"));
    let pass = |name: &str| Element::Pass { name: name.to_string(), options: None };
    let with = |name: &str, options: String| Element::Pass { name: name.to_string(), options: Some(options) };
    let gpu = |e: Element| Element::Nest { op: "gpu.module".to_string(), body: vec![e] };
    let stage1 = vec![
        pass("convert-nvgpu-to-nvvm"),
        pass("gpu-kernel-outlining"),
        pass("convert-vector-to-scf"),
        pass("convert-scf-to-cf"),
        pass("convert-nvvm-to-llvm"),
        pass("convert-func-to-llvm"),
        pass("expand-strided-metadata"),
        with("nvvm-attach-target", format!("chip={chip} features={features} O={opt_level}")),
        pass("lower-affine"),
        pass("convert-arith-to-llvm"),
        pass("convert-index-to-llvm"),
        pass("canonicalize"),
        pass("cse"),
        gpu(with("convert-gpu-to-nvvm", "use-bare-ptr-memref-call-conv=1".to_string())),
        gpu(pass("canonicalize")),
        gpu(pass("cse")),
        gpu(pass("reconcile-unrealized-casts")),
    ];
    let stage2 = vec![
        with("gpu-to-llvm", "use-bare-pointers-for-kernels=1".to_string()),
        with("gpu-module-to-binary", "format=isa".to_string()),
        pass("convert-math-to-llvm"),
        pass("canonicalize"),
        pass("cse"),
        pass("reconcile-unrealized-casts"),
    ];
    LoweringPipeline { stage1, stage2, fallback: Some(reason) }
}

/// Split the expansion before `gpu-to-llvm`, after checking that it holds the passes the split
/// and the options rely on, in order: `nvvm-attach-target` (with the requested chip, features
/// and level), `gpu.module(convert-gpu-to-nvvm)`, `gpu-to-llvm` and `gpu-module-to-binary`.
fn split(
    mut elements: Vec<Element>,
    chip: &str,
    features: &str,
    opt_level: u32,
) -> Result<(Vec<Element>, Vec<Element>), String> {
    let top = |name: &str| {
        elements.iter().position(|e| e.is_pass(name)).ok_or_else(|| format!("the expansion has no {name} pass"))
    };
    let attach = top("nvvm-attach-target")?;
    let to_llvm = top("gpu-to-llvm")?;
    let binary = top("gpu-module-to-binary")?;
    let gpu_to_nvvm = elements
        .iter()
        .position(|e| {
            matches!(e, Element::Nest { op, body } if op == "gpu.module" && body.iter().any(|b| b.is_pass("convert-gpu-to-nvvm")))
        })
        .ok_or("the expansion has no gpu.module(convert-gpu-to-nvvm)")?;
    if !(attach < gpu_to_nvvm && gpu_to_nvvm < to_llvm && to_llvm < binary) {
        return Err("the expansion's nvvm-attach-target, convert-gpu-to-nvvm, gpu-to-llvm and gpu-module-to-binary \
                    are out of order"
            .to_string());
    }
    let Element::Pass { options, .. } = &elements[attach] else { unreachable!("found as a pass") };
    let options = pass_options(options.as_deref().unwrap_or(""));
    let level = opt_level.to_string();
    for (key, want) in [("chip", chip), ("features", features), ("O", level.as_str())] {
        let got = options.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str());
        if got != Some(want) {
            return Err(format!(
                "the expansion's nvvm-attach-target has {key}={}, not {want}",
                got.unwrap_or("nothing")
            ));
        }
    }
    let stage2 = elements.split_off(to_llvm);
    Ok((elements, stage2))
}

/// `key=value` pairs of a pass's options (values may be quoted or braced).
fn pass_options(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut quoted = false;
    let mut word = String::new();
    let mut flush = |word: &mut String| {
        if !word.is_empty() {
            let (k, v) = word.split_once('=').unwrap_or((word.as_str(), ""));
            out.push((k.to_string(), v.to_string()));
            word.clear();
        }
    };
    for c in text.chars() {
        match c {
            '"' => quoted = !quoted,
            '{' if !quoted => depth += 1,
            '}' if !quoted => depth = depth.saturating_sub(1),
            c if c.is_whitespace() && !quoted && depth == 0 => {
                flush(&mut word);
                continue;
            }
            _ => {}
        }
        word.push(c);
    }
    flush(&mut word);
    out
}

/// Parse the `builtin.module(...)` that `--dump-pass-pipeline` prints (after a "Pass Manager
/// with N passes:" line, one element per line).
pub(crate) fn parse_dump(dump: &str) -> Result<Vec<Element>, String> {
    const HEAD: &str = "builtin.module(";
    let start = dump.find(HEAD).ok_or("no builtin.module(...) in the --dump-pass-pipeline output")?;
    let mut cursor = Cursor { text: dump, pos: start + HEAD.len() };
    let elements = cursor.list()?;
    if elements.is_empty() {
        return Err("the dumped pipeline is empty".to_string());
    }
    Ok(elements)
}

struct Cursor<'a> {
    text: &'a str,
    pos: usize,
}

impl Cursor<'_> {
    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn skip_whitespace(&mut self) {
        let rest = &self.text[self.pos..];
        self.pos += rest.len() - rest.trim_start().len();
    }

    fn error(&self, what: &str) -> String {
        let context: String = self.text[self.pos..].chars().take(40).collect();
        format!("cannot parse the dumped pipeline: {what} at `{context}`")
    }

    /// Elements up to and including the closing `)`.
    fn list(&mut self) -> Result<Vec<Element>, String> {
        let mut elements = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(')') => {
                    self.pos += 1;
                    return Ok(elements);
                }
                None => return Err(self.error("unexpected end")),
                _ => {}
            }
            elements.push(self.element()?);
            self.skip_whitespace();
            match self.peek() {
                Some(',') => self.pos += 1,
                Some(')') => {
                    self.pos += 1;
                    return Ok(elements);
                }
                _ => return Err(self.error("expected `,` or `)`")),
            }
        }
    }

    fn element(&mut self) -> Result<Element, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let name = self.text[start..self.pos].to_string();
        if name.is_empty() {
            return Err(self.error("expected a pass name"));
        }
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                Ok(Element::Nest { op: name, body: self.list()? })
            }
            Some('{') => Ok(Element::Pass { name, options: Some(self.braces()?) }),
            _ => Ok(Element::Pass { name, options: None }),
        }
    }

    /// The contents of the braces at the cursor, which ends up after the closing brace.
    fn braces(&mut self) -> Result<String, String> {
        let open = self.pos;
        let mut depth = 0usize;
        let mut quoted = false;
        for (i, c) in self.text[open..].char_indices() {
            match c {
                '"' => quoted = !quoted,
                '{' if !quoted => depth += 1,
                '}' if !quoted => {
                    depth -= 1;
                    if depth == 0 {
                        self.pos = open + i + 1;
                        return Ok(self.text[open + 1..open + i].to_string());
                    }
                }
                _ => {}
            }
        }
        Err(self.error("unclosed `{`"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `mlir-opt --dump-pass-pipeline` of [`expansion_request`]`("sm_80", "+ptx70", 3)`, from
    /// LLVM 21.1.8, 22.1.8 and 23.1.4.
    pub(crate) const DUMP_21: &str = "Pass Manager with 23 passes:
builtin.module(
  convert-nvgpu-to-nvvm,
  gpu-kernel-outlining{data-layout-str=},
  convert-vector-to-scf{full-unroll=false lower-scalable=false lower-tensors=false target-rank=1},
  convert-scf-to-cf,
  convert-nvvm-to-llvm,
  convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
  expand-strided-metadata,
  nvvm-attach-target{O=3 chip=sm_80 fast=false features=+ptx70 ftz=false  module= ptxas-cmd-options= triple=nvptx64-nvidia-cuda},
  lower-affine,
  convert-arith-to-llvm{index-bitwidth=0},
  convert-index-to-llvm{index-bitwidth=64},
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  gpu.module(
    convert-gpu-to-nvvm{ has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=true}
  ),
  gpu.module(
    canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}
  ),
  gpu.module(
    cse
  ),
  gpu.module(
    reconcile-unrealized-casts
  ),
  gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=false use-bare-pointers-for-kernels=true},
  gpu-module-to-binary{format=isa  opts= section= toolkit=},
  convert-math-to-llvm{approximate-log1p=true},
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  reconcile-unrealized-casts
)
";

    pub(crate) const DUMP_22: &str = "Pass Manager with 23 passes:
builtin.module(
  convert-nvgpu-to-nvvm,
  gpu-kernel-outlining{data-layout-str=},
  convert-vector-to-scf{full-unroll=false lower-scalable=false lower-tensors=false target-rank=1},
  convert-scf-to-cf{allow-pattern-rollback=true},
  convert-nvvm-to-llvm,
  convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
  expand-strided-metadata,
  nvvm-attach-target{O=3 chip=sm_80 fast=false features=+ptx70 ftz=false  module= ptxas-cmd-options= triple=nvptx64-nvidia-cuda},
  lower-affine,
  convert-arith-to-llvm{index-bitwidth=0},
  convert-index-to-llvm{index-bitwidth=64},
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  gpu.module(
    convert-gpu-to-nvvm{allow-pattern-rollback=true  has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=true}
  ),
  gpu.module(
    canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}
  ),
  gpu.module(
    cse
  ),
  gpu.module(
    reconcile-unrealized-casts
  ),
  gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=false use-bare-pointers-for-kernels=true},
  gpu-module-to-binary{format=isa  opts= section= toolkit=},
  convert-math-to-llvm{approximate-log1p=true},
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  reconcile-unrealized-casts
)
";

    pub(crate) const DUMP_23: &str = "Pass Manager with 23 passes:
builtin.module(
  convert-nvgpu-to-nvvm,
  gpu-kernel-outlining{data-layout-str=},
  convert-vector-to-scf{full-unroll=false lower-scalable=false lower-tensors=false target-rank=1},
  convert-scf-to-cf{allow-pattern-rollback=true},
  convert-nvvm-to-llvm,
  convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
  expand-strided-metadata,
  nvvm-attach-target{O=3 chip=sm_80 collect-compiler-diagnostics=false fast=false features=+ptx70 ftz=false  module= ptxas-cmd-options= triple=nvptx64-nvidia-cuda verify-target-arch=true},
  lower-affine,
  convert-arith-to-llvm{index-bitwidth=0},
  convert-index-to-llvm{index-bitwidth=64},
  canonicalize{cse-between-iterations=false    max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  gpu.module(
    convert-gpu-to-nvvm{allow-pattern-rollback=true  has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=true}
  ),
  gpu.module(
    canonicalize{cse-between-iterations=false    max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}
  ),
  gpu.module(
    cse
  ),
  gpu.module(
    reconcile-unrealized-casts
  ),
  gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=false use-bare-pointers-for-kernels=true},
  gpu-module-to-binary{format=isa  opts= section= toolkit=},
  convert-math-to-llvm{approximate-log1p=true},
  canonicalize{cse-between-iterations=false    max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  reconcile-unrealized-casts
)
";

    const EXPECTED_NAMES: [&str; 24] = [
        "gpu.module(convert-vector-to-llvm)",
        "convert-nvgpu-to-nvvm",
        "gpu-kernel-outlining",
        "convert-vector-to-scf",
        "convert-scf-to-cf",
        "convert-nvvm-to-llvm",
        "convert-func-to-llvm",
        "expand-strided-metadata",
        "nvvm-attach-target",
        "lower-affine",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "canonicalize",
        "cse",
        "gpu.module(convert-gpu-to-nvvm)",
        "gpu.module(canonicalize)",
        "gpu.module(cse)",
        "gpu.module(reconcile-unrealized-casts)",
        "gpu-to-llvm",
        "gpu-module-to-binary",
        "convert-math-to-llvm",
        "canonicalize",
        "cse",
        "reconcile-unrealized-casts",
    ];

    #[test]
    fn dumps_of_llvm_21_22_and_23_parse() {
        for dump in [DUMP_21, DUMP_22, DUMP_23] {
            let p = from_dump(dump, "sm_80", "+ptx70", 3);
            assert_eq!(p.fallback_reason(), None);
            assert_eq!(p.pass_names(), EXPECTED_NAMES);
            let (s1, s2) = (p.stage1(), p.stage2());
            assert!(s1.starts_with("builtin.module(gpu.module(convert-vector-to-llvm{use-vector-alignment=1}),"));
            assert!(s1.contains(",nvvm-attach-target{O=3 chip=sm_80 "), "{s1}");
            assert!(s1.ends_with(",gpu.module(reconcile-unrealized-casts))"), "{s1}");
            assert!(s2.starts_with("builtin.module(gpu-to-llvm{"), "{s2}");
            assert!(s2.contains("gpu-module-to-binary{format=isa  opts= section= toolkit=}"), "{s2}");
            // The options are kept exactly as dumped, including empty ones.
            assert!(s1.contains("gpu-kernel-outlining{data-layout-str=}"), "{s1}");
        }
        let p23 = from_dump(DUMP_23, "sm_80", "+ptx70", 3);
        assert!(p23.stage1().contains("verify-target-arch=true"));
    }

    #[test]
    fn fallback_list() {
        let p = fallback("sm_90a", "+ptx80", 1, "test".to_string());
        assert_eq!(p.fallback_reason(), Some("test"));
        assert_eq!(p.pass_names(), EXPECTED_NAMES);
        assert_eq!(
            p.stage1(),
            "builtin.module(gpu.module(convert-vector-to-llvm{use-vector-alignment=1}),convert-nvgpu-to-nvvm,\
             gpu-kernel-outlining,convert-vector-to-scf,convert-scf-to-cf,convert-nvvm-to-llvm,convert-func-to-llvm,\
             expand-strided-metadata,nvvm-attach-target{chip=sm_90a features=+ptx80 O=1},lower-affine,\
             convert-arith-to-llvm,convert-index-to-llvm,canonicalize,cse,\
             gpu.module(convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv=1}),gpu.module(canonicalize),gpu.module(cse),\
             gpu.module(reconcile-unrealized-casts))"
        );
        assert_eq!(
            p.stage2(),
            "builtin.module(gpu-to-llvm{use-bare-pointers-for-kernels=1},gpu-module-to-binary{format=isa},\
             convert-math-to-llvm,canonicalize,cse,reconcile-unrealized-casts)"
        );
        // The public constructor raises the PTX version to the target's minimum, as
        // Toolchain::lowering_pipeline does.
        let q = LoweringPipeline::fallback_for(Arch::Sm90a, 70, 1);
        assert_eq!((q.stage1(), q.stage2()), (p.stage1(), p.stage2()));
        assert!(q.fallback_reason().is_some_and(|r| r.contains("fallback_for")));
        let r = LoweringPipeline::fallback_for(Arch::Sm80, 86, 0).stage1();
        assert!(r.contains(",nvvm-attach-target{chip=sm_80 features=+ptx86 O=0},"), "{r}");
    }

    #[test]
    fn unparsable_dumps_fall_back() {
        let missing_binary = DUMP_21.replace("  gpu-module-to-binary{format=isa  opts= section= toolkit=},\n", "");
        let swapped =
            DUMP_21.replace("gpu-to-llvm{", "gpu-to-llvmX{").replace("cse,\n  reconcile", "gpu-to-llvm,\n  r");
        for (dump, reason) in [
            ("", "no builtin.module"),
            ("mlir-opt: unknown pass gpu-lower-to-nvvm-pipeline", "no builtin.module"),
            ("builtin.module(", "unexpected end"),
            ("builtin.module(a, (b))", "expected a pass name"),
            ("builtin.module(a{x=1)", "unclosed `{`"),
            ("builtin.module(a b)", "expected `,` or `)`"),
            ("builtin.module()", "empty"),
            (DUMP_21.split("gpu-to-llvm").next().unwrap(), "unexpected end"),
            (&missing_binary, "no gpu-module-to-binary pass"),
            (&DUMP_21.replace("convert-gpu-to-nvvm", "convert-gpu-to-rocdl"), "no gpu.module(convert-gpu-to-nvvm)"),
            (&swapped, "out of order"),
            (&DUMP_21.replace("chip=sm_80", "chip=sm_90"), "has chip=sm_90, not sm_80"),
            (&DUMP_21.replace("{O=3 ", "{"), "has O=nothing, not 3"),
        ] {
            let p = from_dump(dump, "sm_80", "+ptx70", 3);
            let why = p.fallback_reason().unwrap_or_else(|| panic!("{dump:?} should fall back"));
            assert!(why.contains(reason), "{dump:?}: {why}");
            assert_eq!(p.pass_names(), EXPECTED_NAMES);
        }
    }

    #[test]
    fn pass_option_words() {
        let o = pass_options(r#"O=3  chip=sm_80 module= l={a b} name="x y" flag"#);
        let pairs: Vec<(&str, &str)> = o.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(
            pairs,
            [("O", "3"), ("chip", "sm_80"), ("module", ""), ("l", "{a b}"), ("name", "\"x y\""), ("flag", "")]
        );
    }
}

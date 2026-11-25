//! The kernel between the two lowering stages, and the helpers that hooks edit it with.
//!
//! The first stage lowers the kernel to the LLVM dialect inside its `gpu.module`
//! (`llvm.func`, `llvm.mlir.global`); the second serialises it to PTX. In between, a stage
//! hook ([`CompileOptions::stage_hook`](super::CompileOptions::stage_hook)) can change what
//! the MLIR dialects cannot express before that point, such as LLVM argument attributes
//! (`llvm.byval`, `nvvm.grid_constant`) or the alignment of a global (dynamic shared memory
//! is `@__cute_dynamic_smem`, as aligned as its most aligned buffer and at least to 16 bytes).

use std::ops::Range;

/// The kernel after the first lowering stage: MLIR text in the LLVM dialect, printed with
/// its locations (`--mlir-print-debuginfo`), which the second stage parses again.
#[derive(Clone, Debug)]
pub struct LoweredModule {
    kernel: String,
    text: String,
}

impl LoweredModule {
    pub(crate) fn new(kernel: &str, text: String) -> LoweredModule {
        LoweredModule { kernel: kernel.to_string(), text }
    }

    /// The kernel's name: its `llvm.func` is `@` followed by this name.
    pub fn kernel_name(&self) -> &str {
        &self.kernel
    }

    /// The MLIR text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Replace the MLIR text, for edits the helpers do not cover.
    pub fn set_text(&mut self, text: String) {
        self.text = text;
    }

    pub(crate) fn into_text(self) -> String {
        self.text
    }

    /// The attribute `name` of global `@global` (`llvm.mlir.global`), as MLIR text: `"16 :
    /// i64"` for `alignment`, `"unit"` for a unit attribute.
    pub fn global_attribute(&self, global: &str, name: &str) -> Option<String> {
        let dict = self.global_dict(global).ok()?;
        dict_get(&self.text, dict, name)
    }

    /// Set the attribute `name` of global `@global` to `value` (MLIR attribute syntax, such as
    /// `"1024 : i64"`; `"unit"` for a unit attribute), adding it if it is missing.
    pub fn set_global_attribute(&mut self, global: &str, name: &str, value: &str) -> Result<(), String> {
        match self.global_dict(global)? {
            DictSpot::Dict(range) => self.replace_dict(range, name, value),
            DictSpot::Insert(at) => self.text.insert_str(at, &format!(" {{{}}}", entry(name, value))),
        }
        Ok(())
    }

    /// The attribute `name` of argument `index` of function `@func` (`llvm.func`).
    pub fn argument_attribute(&self, func: &str, index: usize, name: &str) -> Option<String> {
        let dict = self.argument_dict(func, index).ok()?;
        dict_get(&self.text, dict, name)
    }

    /// Set the attribute `name` of argument `index` (from 0) of function `@func` to `value`,
    /// adding it if it is missing: `llvm.align` to `"64 : i64"`, `llvm.byval` to
    /// `"!llvm.array<128 x i8>"`, `nvvm.grid_constant` to `"unit"`.
    pub fn set_argument_attribute(&mut self, func: &str, index: usize, name: &str, value: &str) -> Result<(), String> {
        match self.argument_dict(func, index)? {
            DictSpot::Dict(range) => self.replace_dict(range, name, value),
            DictSpot::Insert(at) => self.text.insert_str(at, &format!(" {{{}}}", entry(name, value))),
        }
        Ok(())
    }

    /// Where the attribute dictionary of `@global` is, or where one would go.
    fn global_dict(&self, global: &str) -> Result<DictSpot, String> {
        let at = find_definition(&self.text, "llvm.mlir.global", global)
            .ok_or_else(|| format!("the lowered module has no llvm.mlir.global @{global}"))?;
        // `@name(initial value)`, then an optional `comdat(...)`, then the dictionary.
        let open = at + 1 + global.len();
        let mut pos = matching(&self.text, open).ok_or_else(|| format!("unbalanced parentheses after @{global}"))? + 1;
        let skip_spaces = |p: usize| p + self.text[p..].len() - self.text[p..].trim_start().len();
        pos = skip_spaces(pos);
        if self.text[pos..].starts_with("comdat(") {
            pos = skip_spaces(matching(&self.text, pos + "comdat".len()).ok_or("unbalanced comdat(...)")? + 1);
        }
        if self.text[pos..].starts_with('{') {
            let close = matching(&self.text, pos).ok_or_else(|| format!("unbalanced attributes of @{global}"))?;
            Ok(DictSpot::Dict(pos + 1..close))
        } else {
            let end = self.text[..pos].trim_end().len();
            Ok(DictSpot::Insert(end))
        }
    }

    /// Where the attribute dictionary of argument `index` of `@func` is, or where one would go.
    fn argument_dict(&self, func: &str, index: usize) -> Result<DictSpot, String> {
        let at = find_definition(&self.text, "llvm.func", func)
            .ok_or_else(|| format!("the lowered module has no llvm.func @{func}"))?;
        let open = at + 1 + func.len();
        let close = matching(&self.text, open).ok_or_else(|| format!("unbalanced argument list of @{func}"))?;
        let args = split_top_level(&self.text, open + 1..close, b',');
        let args: Vec<Range<usize>> = args.into_iter().filter(|r| !self.text[r.clone()].trim().is_empty()).collect();
        let arg = args.get(index).cloned().ok_or_else(|| {
            format!("@{func} has {} argument{}, so there is no argument {index}", args.len(), plural(args.len()))
        })?;
        // The dictionary is the argument's only top-level `{`; it goes before `loc(...)`.
        let mut depth = 0usize;
        let mut loc = None;
        let bytes = self.text.as_bytes();
        let mut i = arg.start;
        while i < arg.end {
            match bytes[i] {
                b'"' => i = skip_string(bytes, i),
                b'{' if depth == 0 => {
                    let close = matching(&self.text, i).ok_or("unbalanced argument attributes")?;
                    return Ok(DictSpot::Dict(i + 1..close));
                }
                b'(' | b'[' | b'{' | b'<' => depth += 1,
                b')' | b']' | b'}' => depth = depth.saturating_sub(1),
                b'>' if i == 0 || bytes[i - 1] != b'-' => depth = depth.saturating_sub(1),
                b'l' if depth == 0
                    && loc.is_none()
                    && bytes[i - 1].is_ascii_whitespace()
                    && self.text[i..arg.end].starts_with("loc(") =>
                {
                    loc = Some(i)
                }
                _ => {}
            }
            i += 1;
        }
        let end = loc.unwrap_or(arg.end);
        Ok(DictSpot::Insert(self.text[..end].trim_end().len()))
    }

    /// Set `name` to `value` in the dictionary whose contents span `range`.
    fn replace_dict(&mut self, range: Range<usize>, name: &str, value: &str) {
        let entries = split_top_level(&self.text, range.clone(), b',');
        for e in &entries {
            if entry_key(&self.text[e.clone()]) == name {
                let lead = self.text[e.clone()].len() - self.text[e.clone()].trim_start().len();
                let trail = self.text[e.clone()].len() - self.text[e.clone()].trim_end().len();
                self.text.replace_range(e.start + lead..e.end - trail, &entry(name, value));
                return;
            }
        }
        let new = if self.text[range.clone()].trim().is_empty() {
            entry(name, value)
        } else {
            format!(", {}", entry(name, value))
        };
        let end = range.start + self.text[range].trim_end().len();
        self.text.insert_str(end, &new);
    }
}

enum DictSpot {
    /// The contents of an existing dictionary.
    Dict(Range<usize>),
    /// Where to insert a new dictionary.
    Insert(usize),
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// `name = value`, or the bare name for a unit attribute (as MLIR prints it).
fn entry(name: &str, value: &str) -> String {
    if value.trim() == "unit" { name.to_string() } else { format!("{name} = {value}") }
}

fn entry_key(entry: &str) -> &str {
    let key = entry.split('=').next().unwrap_or("").trim();
    key.trim_matches('"')
}

fn dict_get(text: &str, spot: DictSpot, name: &str) -> Option<String> {
    let DictSpot::Dict(range) = spot else { return None };
    split_top_level(text, range, b',').into_iter().find_map(|e| {
        let e = text[e].trim();
        (entry_key(e) == name).then(|| match e.split_once('=') {
            Some((_, v)) => v.trim().to_string(),
            None => "unit".to_string(),
        })
    })
}

/// The offset of `@name` where op `op` defines it: on a line containing `op `, followed by
/// `(`. (Uses, such as `llvm.call @name(` or `llvm.mlir.addressof @name`, do not count.)
fn find_definition(text: &str, op: &str, name: &str) -> Option<usize> {
    let needle = format!("@{name}(");
    let mut from = 0;
    while let Some(found) = text[from..].find(&needle) {
        let at = from + found;
        let line_start = text[..at].rfind('\n').map_or(0, |p| p + 1);
        let head = &text[line_start..at];
        if head.split_whitespace().any(|w| w == op) {
            return Some(at);
        }
        from = at + needle.len();
    }
    None
}

/// The index of the quote that closes the string literal starting at `start` (a `"`).
fn skip_string(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// The index of the bracket that closes the one at `open` (any of `(`, `[`, `{`, `<`), skipping
/// string literals and the `>` of `->`.
fn matching(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut stack = Vec::new();
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => i = skip_string(bytes, i),
            b'(' => stack.push(b')'),
            b'[' => stack.push(b']'),
            b'{' => stack.push(b'}'),
            b'<' => stack.push(b'>'),
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            c @ (b')' | b']' | b'}' | b'>') => {
                if stack.pop() != Some(c) {
                    return None;
                }
                if stack.is_empty() {
                    return Some(i);
                }
            }
            _ => {}
        }
        if stack.is_empty() {
            return None;
        }
        i += 1;
    }
    None
}

/// Split `range` of `text` at the top-level occurrences of `sep`.
fn split_top_level(text: &str, range: Range<usize>, sep: u8) -> Vec<Range<usize>> {
    let bytes = text.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = range.start;
    let mut i = range.start;
    while i < range.end {
        match bytes[i] {
            b'"' => i = skip_string(bytes, i),
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            b')' | b']' | b'}' | b'>' => depth = depth.saturating_sub(1),
            c if c == sep && depth == 0 => {
                parts.push(start..i);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(start..range.end);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = r#"#loc4 = loc("<stdin>":3:37)
module attributes {gpu.container_module} {
  gpu.module @kernels [#nvvm.target<O = 3, chip = "sm_80", features = "+ptx70">] {
    llvm.mlir.global internal @__dynamic_shmem__0() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8> loc(#loc2)
    llvm.mlir.global internal constant @fmt("x = %d, y = (%f)\0A\00") {addr_space = 0 : i32} : !llvm.array<18 x i8> loc(#loc3)
    llvm.mlir.global private @bare(0 : i32) : i32
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32
    llvm.func @k(%arg0: !llvm.ptr {llvm.align = 16 : i64} loc("<stdin>":3:37), %arg1: f32 loc("a,b(\"c\")":3:86), %arg2: !llvm.struct<(i32, f32)>) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.addressof @__dynamic_shmem__0 : !llvm.ptr<3> loc(#loc6)
      %1 = llvm.call @vprintf(%0, %0) : (!llvm.ptr<3>, !llvm.ptr<3>) -> i32 loc(#loc7)
      llvm.return loc(#loc8)
    } loc(#loc9)
  } loc(#loc10)
} loc(#loc)
"#;

    fn module() -> LoweredModule {
        LoweredModule::new("k", TEXT.to_string())
    }

    #[test]
    fn global_attributes() {
        let mut m = module();
        assert_eq!(m.global_attribute("__dynamic_shmem__0", "alignment").as_deref(), Some("16 : i64"));
        assert_eq!(m.global_attribute("__dynamic_shmem__0", "addr_space").as_deref(), Some("3 : i32"));
        assert_eq!(m.global_attribute("__dynamic_shmem__0", "missing"), None);
        m.set_global_attribute("__dynamic_shmem__0", "alignment", "1024 : i64").unwrap();
        assert!(m.text().contains(
            "@__dynamic_shmem__0() {addr_space = 3 : i32, alignment = 1024 : i64} : !llvm.array<0 x i8> loc(#loc2)"
        ));
        // The initial value's parentheses and quotes do not confuse the search.
        m.set_global_attribute("fmt", "alignment", "8 : i64").unwrap();
        assert!(m.text().contains(r#"@fmt("x = %d, y = (%f)\0A\00") {addr_space = 0 : i32, alignment = 8 : i64} :"#));
        // A global without a dictionary gets one.
        m.set_global_attribute("bare", "alignment", "4 : i64").unwrap();
        assert!(m.text().contains("@bare(0 : i32) {alignment = 4 : i64} : i32"), "{}", m.text());
        assert_eq!(m.global_attribute("bare", "alignment").as_deref(), Some("4 : i64"));
        // The use of the global was left alone.
        assert!(m.text().contains("llvm.mlir.addressof @__dynamic_shmem__0 : !llvm.ptr<3>"));
        let e = m.set_global_attribute("nope", "alignment", "4 : i64").unwrap_err();
        assert!(e.contains("no llvm.mlir.global @nope"), "{e}");
    }

    #[test]
    fn argument_attributes() {
        let mut m = module();
        assert_eq!(m.argument_attribute("k", 0, "llvm.align").as_deref(), Some("16 : i64"));
        assert_eq!(m.argument_attribute("k", 1, "llvm.align"), None);
        m.set_argument_attribute("k", 0, "llvm.align", "64 : i64").unwrap();
        m.set_argument_attribute("k", 0, "nvvm.grid_constant", "unit").unwrap();
        assert!(
            m.text().contains(r#"%arg0: !llvm.ptr {llvm.align = 64 : i64, nvvm.grid_constant} loc("<stdin>":3:37)"#),
            "{}",
            m.text()
        );
        assert_eq!(m.argument_attribute("k", 0, "nvvm.grid_constant").as_deref(), Some("unit"));
        // A dictionary goes before the location, whose quotes hold a comma and parentheses.
        m.set_argument_attribute("k", 1, "llvm.noundef", "unit").unwrap();
        assert!(m.text().contains(r#"%arg1: f32 {llvm.noundef} loc("a,b(\"c\")":3:86)"#), "{}", m.text());
        // An argument whose type holds commas, and no location.
        m.set_argument_attribute("k", 2, "llvm.byval", "!llvm.struct<(i32, f32)>").unwrap();
        assert!(m.text().contains("%arg2: !llvm.struct<(i32, f32)> {llvm.byval = !llvm.struct<(i32, f32)>})"));
        let e = m.set_argument_attribute("k", 3, "llvm.align", "4 : i64").unwrap_err();
        assert!(e.contains("@k has 3 arguments, so there is no argument 3"), "{e}");
        // `llvm.call @vprintf(` is a use, and the declaration is found instead.
        let e = m.set_argument_attribute("vprintf", 0, "llvm.noundef", "unit");
        assert!(e.is_ok(), "{e:?}");
        assert!(m.text().contains("llvm.func @vprintf(!llvm.ptr {llvm.noundef}, !llvm.ptr) -> i32"), "{}", m.text());
        assert!(m.set_argument_attribute("missing", 0, "a", "unit").unwrap_err().contains("no llvm.func @missing"));
    }

    #[test]
    fn brackets() {
        let t = r#"f(a, "b)", g<x -> y>(c), {d, e}) tail"#;
        let close = matching(t, 1).unwrap();
        assert_eq!(&t[close..], ") tail");
        let parts: Vec<&str> = split_top_level(t, 2..close, b',').into_iter().map(|r| t[r].trim()).collect();
        assert_eq!(parts, ["a", "\"b)\"", "g<x -> y>(c)", "{d, e}"]);
        assert_eq!(matching("(]", 0), None);
        assert_eq!(matching("(()", 0), None);
    }
}

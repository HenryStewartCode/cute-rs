//! The export of a module of several kernels ([`CompiledModule::export_c`]): one image, and
//! in C one module struct with a launch function per kernel.
//!
//! The manifest is export format 2 ([`MODULE_EXPORT_FORMAT_VERSION`]): the stamp and image
//! lines of format 1, a `module <name>` and a `kernels <n>` line, then one `signature` section
//! per kernel. A module of one kernel is exported in format 1, exactly as
//! [`CompiledKernel::export_c`] exports the kernel, and [`ExportedModule::read`] reads both.
//!
//! The C files of a module `p` of kernels `a` and `b` declare `p_Module_t`, `p_Module_Load`,
//! `p_Module_Unload`, `p_Launch_Config_t`, and per kernel `p_a_Args_t` and `p_a_Launch`, with
//! the macros `P_A_KERNEL_NAME` and `P_A_INDEX` (the kernel's slot in `functions`).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::{
    AotError, CImage, CModuleNames, EXPORT_FORMAT_VERSION, ExportOptions, ExportReport, ExportStamp, ExportedKernel,
    ImageChoice, MAGIC, c_args_struct, c_comment, c_ident, c_images, c_launch, c_launch_config, c_load, c_string,
    c_unload, compile_object, field_names, is_file_name, upper,
};
use crate::backend::{Arch, CompiledModule};
use crate::signature::KernelSignature;

/// The export format of a module of several kernels.
pub const MODULE_EXPORT_FORMAT_VERSION: u32 = 2;

/// An exported module: its stamp, its kernels' signatures (each with the module's target and
/// PTX version) and its images, which every kernel is loaded from.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportedModule {
    pub stamp: ExportStamp,
    /// The module's name (the kernel's, for a module of one).
    pub name: String,
    pub signatures: Vec<KernelSignature>,
    pub ptx: Option<String>,
    pub cubin: Option<Vec<u8>>,
}

impl From<ExportedKernel> for ExportedModule {
    /// The module of one exported kernel (export format 1).
    fn from(k: ExportedKernel) -> ExportedModule {
        ExportedModule {
            stamp: k.stamp,
            name: k.signature.name.clone(),
            signatures: vec![k.signature],
            ptx: k.ptx,
            cubin: k.cubin,
        }
    }
}

impl ExportedModule {
    /// The export of `module`, stamped with this cute-rs: format 1 for one kernel, else 2.
    pub fn from_compiled(module: &CompiledModule) -> ExportedModule {
        let mut m = ExportedModule::from(ExportedKernel::from_compiled(&module.kernels()[0]));
        m.name = module.name().to_string();
        for k in &module.kernels()[1..] {
            m.signatures.push(ExportedKernel::from_compiled(k).signature);
        }
        if m.signatures.len() > 1 {
            m.stamp.format = MODULE_EXPORT_FORMAT_VERSION;
        }
        m
    }

    /// The kernel `name` as an [`ExportedKernel`] with the module's images. An exported kernel
    /// is export format 1 (what [`ExportedKernel::write_c`] writes and reads back), whatever
    /// the module's format.
    pub fn kernel(&self, name: &str) -> Option<ExportedKernel> {
        let signature = self.signatures.iter().find(|s| s.name == name)?.clone();
        let stamp = ExportStamp { format: EXPORT_FORMAT_VERSION, ..self.stamp.clone() };
        Some(ExportedKernel { stamp, signature, ptx: self.ptx.clone(), cubin: self.cubin.clone() })
    }

    /// The kernels' names, in order.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.signatures.iter().map(|s| s.name.as_str()).collect()
    }

    fn first(&self) -> ExportedKernel {
        self.kernel(&self.signatures[0].name).expect("a module has a kernel")
    }

    /// The module's target.
    pub fn target(&self) -> Arch {
        self.signatures[0].target.expect("validated")
    }

    /// The module's PTX ISA version.
    pub fn ptx_version(&self) -> u32 {
        self.signatures[0].ptx_version.expect("validated")
    }

    /// Which image a device loads ([`ExportedKernel::choose_image`]).
    pub fn choose_image(&self, cc: (u32, u32), driver_ptx: Option<u32>) -> Result<ImageChoice, AotError> {
        self.first().choose_image(cc, driver_ptx)
    }

    fn validate(&self) -> Result<(), AotError> {
        let Some(first) = self.signatures.first() else {
            return Err(AotError::new(format!("exported module `{}` has no kernels", self.name)));
        };
        self.first().validate()?;
        for (i, s) in self.signatures.iter().enumerate() {
            if (s.target, s.ptx_version) != (first.target, first.ptx_version) {
                return Err(AotError::new(format!(
                    "exported module `{}`: kernel `{}` has another target or PTX version than kernel `{}`",
                    self.name, s.name, first.name
                )));
            }
            if self.signatures[..i].iter().any(|o| o.name == s.name) {
                return Err(AotError::new(format!(
                    "exported module `{}` has two kernels named `{}`",
                    self.name, s.name
                )));
            }
        }
        if self.signatures.len() > 1 && self.stamp.format != MODULE_EXPORT_FORMAT_VERSION {
            return Err(AotError::new(format!(
                "exported module `{}` has {} kernels, which export format {} carries (not {})",
                self.name,
                self.signatures.len(),
                MODULE_EXPORT_FORMAT_VERSION,
                self.stamp.format
            )));
        }
        Ok(())
    }

    /// The manifest text, naming the image files beside it: format 1 for one kernel
    /// ([`ExportedKernel::manifest`]), else format 2.
    pub fn manifest(&self, ptx_file: Option<&str>, cubin_file: Option<&str>) -> String {
        if self.signatures.len() == 1 {
            return self.first().manifest(ptx_file, cubin_file);
        }
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
        out += &format!("module {}\nkernels {}\n", self.name, self.signatures.len());
        for s in &self.signatures {
            out += "signature\n";
            out += &s.to_text();
        }
        out
    }

    /// Read a manifest of either format: the stamp, the module's name, the signatures and the
    /// image files it names.
    pub fn parse_manifest(
        text: &str,
    ) -> Result<(ExportStamp, String, Vec<KernelSignature>, Option<String>, Option<String>), AotError> {
        let bad = |line: usize, m: String| AotError::new(format!("export manifest, line {}: {m}", line + 1));
        let first: Vec<&str> = text.lines().next().unwrap_or("").split_whitespace().collect();
        if first.len() == 2 && first[0] == MAGIC && first[1] == EXPORT_FORMAT_VERSION.to_string() {
            let (stamp, sig, ptx, cubin) = ExportedKernel::parse_manifest(text)?;
            return Ok((stamp, sig.name.clone(), vec![sig], ptx, cubin));
        }
        let format = match first.as_slice() {
            [m, v] if *m == MAGIC => v.parse::<u32>().map_err(|_| bad(0, format!("bad format version `{v}`")))?,
            _ => return Err(bad(0, format!("not a cute-rs export manifest (it starts with `{MAGIC} <version>`)"))),
        };
        if format != MODULE_EXPORT_FORMAT_VERSION {
            return Err(AotError::new(format!(
                "the export is format version {format}, and this cute-rs ({}) reads versions {EXPORT_FORMAT_VERSION} \
                 and {MODULE_EXPORT_FORMAT_VERSION}: export the module again with this cute-rs",
                crate::VERSION
            )));
        }
        let lines: Vec<&str> = text.lines().collect();
        let (mut crate_version, mut cubin_target, mut ptx, mut cubin, mut name, mut count) =
            (None, None, None, None, None, None);
        let mut at = 1;
        while at < lines.len() {
            let t: Vec<&str> = lines[at].split_whitespace().collect();
            match t.as_slice() {
                [] => {}
                ["signature"] => break,
                ["crate", v] if crate_version.is_none() => crate_version = Some(v.to_string()),
                ["cubin_target", c] if cubin_target.is_none() => {
                    cubin_target = Some(c.parse::<Arch>().map_err(|e| bad(at, e.to_string()))?)
                }
                ["image", _, f] if !is_file_name(f) => {
                    return Err(bad(at, format!("image file `{f}` is not a file name beside the manifest")));
                }
                ["image", "ptx", f] if ptx.is_none() => ptx = Some(f.to_string()),
                ["image", "cubin", f] if cubin.is_none() => cubin = Some(f.to_string()),
                ["module", n] if name.is_none() => name = Some(n.to_string()),
                ["kernels", n] if count.is_none() => {
                    count = Some(n.parse::<usize>().map_err(|_| bad(at, format!("bad kernel count `{n}`")))?)
                }
                _ => return Err(bad(at, format!("unexpected line `{}`", lines[at]))),
            }
            at += 1;
        }
        let crate_version = crate_version.ok_or_else(|| AotError::new("export manifest: no `crate` line"))?;
        let name = name.ok_or_else(|| AotError::new("export manifest: no `module` line"))?;
        let count = count.ok_or_else(|| AotError::new("export manifest: no `kernels` line"))?;
        let mut signatures = Vec::new();
        while at < lines.len() {
            // lines[at] is `signature`; the section runs to the next one.
            let start = at + 1;
            let mut end = start;
            while end < lines.len() && lines[end].trim() != "signature" {
                end += 1;
            }
            let sig = KernelSignature::from_text(&lines[start..end].join("\n"))
                .map_err(|e| AotError::new(format!("export manifest, line {}: {}", e.line + start, e.message)))?;
            signatures.push(sig);
            at = end;
        }
        if signatures.len() != count {
            return Err(AotError::new(format!(
                "export manifest: `kernels {count}`, but it has {} signatures",
                signatures.len()
            )));
        }
        Ok((ExportStamp { format, crate_version, cubin_target }, name, signatures, ptx, cubin))
    }

    /// An export from a manifest (of either format) and its images.
    pub fn from_parts(manifest: &str, ptx: Option<&str>, cubin: Option<&[u8]>) -> Result<ExportedModule, AotError> {
        let (stamp, name, signatures, _, _) = ExportedModule::parse_manifest(manifest)?;
        let m =
            ExportedModule { stamp, name, signatures, ptx: ptx.map(str::to_string), cubin: cubin.map(<[u8]>::to_vec) };
        m.validate()?;
        Ok(m)
    }

    /// Read an export (of a module, or of one kernel) from its manifest and the images it
    /// names, beside it.
    pub fn read(path: impl AsRef<Path>) -> Result<ExportedModule, AotError> {
        let path = path.as_ref();
        let io = |p: &Path, e: std::io::Error| AotError::new(format!("cannot read {}: {e}", p.display()));
        let text = std::fs::read_to_string(path).map_err(|e| io(path, e))?;
        let (stamp, name, signatures, ptx_file, cubin_file) = ExportedModule::parse_manifest(&text)?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let ptx = match ptx_file {
            Some(f) => Some(std::fs::read_to_string(dir.join(&f)).map_err(|e| io(&dir.join(&f), e))?),
            None => None,
        };
        let cubin = match cubin_file {
            Some(f) => Some(std::fs::read(dir.join(&f)).map_err(|e| io(&dir.join(&f), e))?),
            None => None,
        };
        let m = ExportedModule { stamp, name, signatures, ptx, cubin };
        m.validate()?;
        Ok(m)
    }

    /// The C header (`<name>.h`), with symbols starting `prefix` (`name` when empty).
    pub fn c_header(&self, name: &str, prefix: &str) -> String {
        if self.signatures.len() == 1 {
            return self.first().c_header(name, prefix);
        }
        module_c_header(self, name, &c_ident(if prefix.is_empty() { name } else { prefix }))
    }

    /// The C source (`<name>.c`), which includes `<name>.h`.
    pub fn c_source(&self, name: &str, prefix: &str) -> String {
        if self.signatures.len() == 1 {
            return self.first().c_source(name, prefix);
        }
        module_c_source(self, name, &c_ident(if prefix.is_empty() { name } else { prefix }))
    }

    /// Write the export to `dir`, as [`ExportedKernel::write_c`] does for one kernel.
    pub fn write_c(
        &self,
        dir: impl AsRef<Path>,
        name: &str,
        prefix: &str,
        options: &ExportOptions,
    ) -> Result<ExportReport, AotError> {
        if self.signatures.len() == 1 {
            return self.first().write_c(dir, name, prefix, options);
        }
        let dir = dir.as_ref();
        if !is_file_name(name) {
            return Err(AotError::new(format!(
                "`{name}` is not a file name for an export: use letters, digits, `_`, `-` and `.`"
            )));
        }
        if let Some(e) = crate::aot::unexportable_ptx(self.ptx.as_deref(), &self.name) {
            return Err(AotError::new(e));
        }
        for (i, s) in self.signatures.iter().enumerate() {
            if let Some(e) = s.text_error() {
                return Err(AotError::new(e));
            }
            let id = upper(&kernel_ident(s));
            if let Some(o) = self.signatures[..i].iter().find(|o| upper(&kernel_ident(o)) == id) {
                return Err(AotError::new(format!(
                    "kernels `{}` and `{}` of module `{}` have the same C identifier in the export's macros",
                    o.name, s.name, self.name
                )));
            }
        }
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
                notes.push("no cubin (the module was compiled without ptxas): the export carries the PTX only".into());
                None
            }
        };
        let manifest = write(
            &format!("{name}.sig"),
            self.manifest(self.ptx.as_ref().map(|_| ptx_file.as_str()), cubin.as_ref().map(|_| cubin_file.as_str()))
                .as_bytes(),
        )?;
        let header = write(&format!("{name}.h"), self.c_header(name, prefix).as_bytes())?;
        let source = write(&format!("{name}.c"), self.c_source(name, prefix).as_bytes())?;
        let object = if options.compile_object { compile_object(dir, name, options, &mut notes)? } else { None };
        Ok(ExportReport { header, source, manifest, ptx, cubin, object, notes })
    }
}

/// A kernel's identifier in C symbols and macros: its name as a C identifier.
fn kernel_ident(sig: &KernelSignature) -> String {
    c_ident(&sig.name)
}

fn module_c_header(m: &ExportedModule, name: &str, p: &str) -> String {
    let up = upper(p);
    let (target, v) = (m.target(), m.ptx_version());
    let kernels: Vec<String> = m.signatures.iter().map(|s| format!("`{}`", c_comment(&s.name))).collect();
    let first = kernel_ident(&m.signatures[0]);
    let mut h = String::new();
    let _ = writeln!(
        h,
        "/* {name}.h: the module `{}` of {} kernels ({}), exported by cute-rs {} (export format {}) for {}, PTX ISA \
         {}.{}.\n *\n * Generated code: do not edit. Build {name}.c with the CUDA toolkit's headers and link with \
         -lcuda:\n *\n *   {p}_Module_t module;\n *   {p}_Module_Load(&module);   (a CUresult; module.error says why \
         on failure)\n *   {p}_{first}_Args_t args = {{ ... }};   (one field per parameter of the kernel)\n *   \
         {p}_Launch_Config_t config = {{ {{grid}}, {{block}}, {{1, 1, 1}}, 0, stream }};\n *   \
         {p}_{first}_Launch(&module, &config, &args);   (and so for each kernel)\n *   \
         {p}_Module_Unload(&module);\n */",
        c_comment(&m.name),
        m.signatures.len(),
        kernels.join(", "),
        m.stamp.crate_version,
        m.stamp.format,
        target.chip(),
        v / 10,
        v % 10
    );
    let _ = writeln!(h, "#ifndef {up}_CUTE_RS_EXPORT_H\n#define {up}_CUTE_RS_EXPORT_H\n");
    h += "#include <stddef.h>\n#include <stdint.h>\n#include <cuda.h>\n\n";
    let _ = writeln!(h, "#define {up}_CUTE_RS_EXPORT_FORMAT {}", m.stamp.format);
    let _ = writeln!(h, "#define {up}_CUTE_RS_VERSION {}", c_string(&m.stamp.crate_version));
    let _ = writeln!(h, "#define {up}_MODULE_NAME {}", c_string(&m.name));
    let _ = writeln!(h, "#define {up}_TARGET {}", c_string(target.chip()));
    let _ = writeln!(h, "#define {up}_PTX_VERSION {v}");
    if let Some(t) = m.stamp.cubin_target {
        let _ = writeln!(h, "#define {up}_CUBIN_TARGET {}", c_string(t.chip()));
    }
    let _ = writeln!(h, "#define {up}_NUM_KERNELS {}", m.signatures.len());
    for (i, sig) in m.signatures.iter().enumerate() {
        let kup = format!("{up}_{}", upper(&kernel_ident(sig)));
        let _ = writeln!(h, "\n#define {kup}_KERNEL_NAME {}\n#define {kup}_INDEX {i}", c_string(&sig.name));
        if let Some([x, y, z]) = sig.block_dim {
            let _ = writeln!(
                h,
                "#define {kup}_BLOCK_DIM_X {x}\n#define {kup}_BLOCK_DIM_Y {y}\n#define {kup}_BLOCK_DIM_Z {z}"
            );
        }
        let _ = writeln!(h, "#define {kup}_NUM_PARAMS {}", sig.params.len());
    }
    h += "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
    for sig in &m.signatures {
        let comment = format!("The parameters of kernel `{}`, in order.", c_comment(&sig.name));
        h += &c_args_struct(sig, &field_names(sig), &comment, &format!("{p}_{}_Args_t", kernel_ident(sig)));
    }
    h += &c_launch_config(p);
    let _ = writeln!(
        h,
        "/* The loaded module: one function per kernel, at the kernel's _INDEX. `error` holds a message when a \
         function returns an error. */\ntypedef struct {{\n    CUcontext context;\n    CUmodule module;\n    \
         CUfunction functions[{}];\n    CUdevice device;\n    int retained_primary;\n    int from_cubin;\n    \
         char error[256];\n}} {p}_Module_t;\n",
        m.signatures.len()
    );
    let _ = writeln!(
        h,
        "/* Load the module into the current context (the first device's primary context when none is current) and \
         look up\n * every kernel: the cubin when the device runs it as a binary, else the PTX, which the driver's JIT \
         compiles. */\nCUresult {p}_Module_Load({p}_Module_t *module);\n\n\
         /* Unload the module (and release the primary context Load retained). */\n\
         CUresult {p}_Module_Unload({p}_Module_t *module);\n"
    );
    for sig in &m.signatures {
        let k = kernel_ident(sig);
        let _ = writeln!(
            h,
            "/* Check a launch of kernel `{}` (block size, clusters, pointer alignment and size, divisibility) and \
             queue it\n * (cuLaunchKernelEx), from a thread whose current context is the module's. Returns\n * \
             CUDA_ERROR_INVALID_VALUE, with module->error set, when a check fails. */\n\
             CUresult {p}_{k}_Launch({p}_Module_t *module, const {p}_Launch_Config_t *config, const {p}_{k}_Args_t \
             *args);\n",
            c_comment(&sig.name)
        );
    }
    let _ = writeln!(h, "#ifdef __cplusplus\n}}\n#endif\n\n#endif /* {up}_CUTE_RS_EXPORT_H */");
    h
}

fn module_c_source(m: &ExportedModule, name: &str, p: &str) -> String {
    let up = upper(p);
    let mut c = String::new();
    let _ = writeln!(
        c,
        "/* {name}.c: the module `{}` of {} kernels, exported by cute-rs {} for {}. Generated code: do not edit. */\n",
        c_comment(&m.name),
        m.signatures.len(),
        m.stamp.crate_version,
        m.target().chip()
    );
    let image = CImage {
        stamp: &m.stamp,
        ptx: m.ptx.as_deref(),
        cubin: m.cubin.as_deref(),
        target: m.target(),
        v: m.ptx_version(),
    };
    c += &c_images(&image, name, &up, p);
    let names = CModuleNames {
        load: format!("{p}_Module_Load"),
        unload: format!("{p}_Module_Unload"),
        module_t: format!("{p}_Module_t"),
        what: "module",
        name_macro: format!("{up}_MODULE_NAME"),
    };
    let unload = &names.unload;
    let mut lookup = String::new();
    for sig in &m.signatures {
        let kup = format!("{up}_{}", upper(&kernel_ident(sig)));
        let _ = writeln!(
            lookup,
            "    r = cuModuleGetFunction(&m->functions[{kup}_INDEX], m->module, {kup}_KERNEL_NAME);\n    if (r != \
             CUDA_SUCCESS) {{ {unload}(m); return {up}_FAIL(m, r, \"cuModuleGetFunction failed (%d) for `%s`\", (int)r, \
             {kup}_KERNEL_NAME); }}"
        );
    }
    lookup += "    m->error[0] = 0;\n    return CUDA_SUCCESS;\n}\n\n";
    c += &c_load(&image, p, &up, &names, &lookup);
    c += &c_unload(&names, "memset(m->functions, 0, sizeof(m->functions));");
    for sig in &m.signatures {
        let k = kernel_ident(sig);
        let kup = format!("{up}_{}", upper(&k));
        c += &c_launch(
            sig,
            &field_names(sig),
            p,
            &up,
            &format!("{p}_{k}_Launch"),
            &names.module_t,
            &format!("{p}_{k}_Args_t"),
            &format!("m->functions[{kup}_INDEX]"),
        );
        c += "\n";
    }
    c
}

impl CompiledModule {
    /// Export the module to C ([`ExportedModule::write_c`]): one header and source with the
    /// module struct and one launch function per kernel, the manifest and the images. A module
    /// of one kernel is exported as [`CompiledKernel::export_c`] exports it.
    ///
    /// [`CompiledKernel::export_c`]: crate::CompiledKernel::export_c
    pub fn export_c(&self, dir: impl AsRef<Path>, name: &str, prefix: &str) -> Result<ExportReport, AotError> {
        self.export_c_with(dir, name, prefix, &ExportOptions::default())
    }

    /// [`export_c`](Self::export_c) with options.
    pub fn export_c_with(
        &self,
        dir: impl AsRef<Path>,
        name: &str,
        prefix: &str,
        options: &ExportOptions,
    ) -> Result<ExportReport, AotError> {
        ExportedModule::from_compiled(self).write_c(dir, name, prefix, options)
    }

    /// The module as an [`ExportedModule`], without writing files.
    pub fn to_exported(&self) -> ExportedModule {
        ExportedModule::from_compiled(self)
    }
}

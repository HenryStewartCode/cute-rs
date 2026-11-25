//! `ptxas` and cubins, kept artifacts and debug mode, with fake and (when installed) real CUDA
//! tools. Skipped (with a note) without an `mlir-opt`; the real-`ptxas` tests also need one on
//! `PATH` (or in `/usr/local/cuda/bin`, or named by `CUTE_RS_PTXAS`):
//!
//! ```text
//! cargo test --test artifacts
//! ```

#[allow(dead_code)]
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
#[allow(dead_code)]
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;

use std::path::{Path, PathBuf};

use cute::prelude::*;
use cute::{Artifact, CudaTools, Tool};

fn toolchain() -> Option<Toolchain> {
    let t = Toolchain::discover();
    if t.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    t
}

/// A toolchain that sees no CUDA toolkit, whatever the machine has.
fn bare(t: Toolchain) -> Toolchain {
    t.with_cuda_tools(CudaTools::default())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cute-rs-artifacts-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> =
        std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}

/// A shell script standing in for `ptxas`: it logs its command line, writes a fake cubin and a
/// `-v` style report, and fails when the PTX file contains `FAIL`.
#[cfg(unix)]
fn fake_ptxas(dir: &Path, release: (u32, u32)) -> (Tool, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).unwrap();
    let (script, log) = (dir.join("ptxas"), dir.join("commands.log"));
    let text = format!(
        "#!/bin/sh\n\
         case \"$1\" in --version) echo \"Cuda compilation tools, release {}.{}, V{}.{}.1\"; exit 0;; esac\n\
         echo \"$@\" >> {}\n\
         for last in \"$@\"; do :; done\n\
         while [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then out=\"$2\"; fi; shift; done\n\
         if grep -q FAIL \"$last\"; then echo \"ptxas fatal   : refused\" >&2; exit 1; fi\n\
         printf 'FAKECUBIN' > \"$out\"\n\
         echo \"ptxas info    : Used 12 registers, 0 bytes smem\" >&2\n",
        release.0,
        release.1,
        release.0,
        release.1,
        log.display()
    );
    std::fs::write(&script, text).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    (Tool { path: script, release: Some(release) }, log)
}

#[test]
#[cfg(unix)]
fn ptxas_is_run_with_our_flags_then_the_users() {
    let Some(t) = toolchain() else { return };
    let dir = scratch("fake");
    let (tool, log) = fake_ptxas(&dir, (12, 8));
    let t = t.with_cuda_tools(CudaTools { ptxas: Some(tool), ..CudaTools::default() });
    let kernel = elementwise_add::naive();
    let base = CompileOptions::new().gpu_arch(Arch::Sm80);

    let compiled = kernel
        .compile_with(&t, &base.clone().ptxas_options("--maxrregcount=32 'x y'").opt_level(2).generate_line_info(true))
        .unwrap();
    assert_eq!(compiled.cubin(), Some(&b"FAKECUBIN"[..]));
    assert!(compiled.ptxas_log().unwrap().contains("Used 12 registers"));
    let command = compiled.ptxas_command().unwrap().to_string();
    let args: Vec<&str> = command.split_whitespace().collect();
    assert_eq!(args[1], "-arch=sm_80");
    assert_eq!(&args[2..6], ["-O2", "-lineinfo", "-v", "--maxrregcount=32"], "{command}");
    // One argument with a space, shown quoted as a shell would need it (no shell runs it).
    assert!(command.contains(" 'x y' -o "), "{command}");
    assert!(std::fs::read_to_string(&log).unwrap().contains(" x y -o "));

    // The user's own -O replaces ours.
    let compiled = kernel.compile_with(&t, &base.clone().ptxas_options("-O0")).unwrap();
    let command = compiled.ptxas_command().unwrap();
    assert_eq!(command.matches("-O").count(), 1, "{command}");
    assert!(command.contains(" -O0 "), "{command}");

    // The temporary directory is gone afterwards.
    let used = std::fs::read_to_string(&log).unwrap();
    let out = used.split_whitespace().skip_while(|w| *w != "-o").nth(1).unwrap();
    assert!(!Path::new(out).parent().unwrap().exists(), "{out} was left behind");

    // Nothing asks for ptxas by default; use_ptxas(false) and compile_ptx never run it.
    let before = std::fs::read_to_string(&log).unwrap().lines().count();
    assert!(kernel.compile_with(&t, &base).unwrap().cubin().is_none());
    assert!(kernel.compile_with(&t, &base.clone().use_ptxas(true)).unwrap().cubin().is_some());
    assert!(kernel.compile_with(&t, &base.clone().keep_cubin(true).dump_dir(&dir)).unwrap().cubin().is_some());
    assert_eq!(std::fs::read_to_string(&log).unwrap().lines().count(), before + 2);
    let before = std::fs::read_to_string(&log).unwrap().lines().count();
    assert!(kernel.compile_with(&t, &base.clone().use_ptxas(false)).unwrap().cubin().is_none());
    kernel.compile_ptx(&t, Arch::Sm80).unwrap();
    assert_eq!(std::fs::read_to_string(&log).unwrap().lines().count(), before);

    // One PTX assembled with two sets of flags is two variants, with a cubin each.
    let dump = scratch("fake-variants");
    let keep = base.clone().keep_cubin(true).dump_dir(&dump);
    let plain = kernel.compile_with(&t, &keep).unwrap();
    let capped = kernel.compile_with(&t, &keep.clone().ptxas_options("--maxrregcount=32")).unwrap();
    assert_eq!(plain.ptx(), capped.ptx());
    assert_eq!(files(&dump).len(), 2, "{:?}", files(&dump));
    let _ = std::fs::remove_dir_all(&dump);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[cfg(unix)]
fn ptxas_failures_and_old_toolkits_are_reported() {
    let Some(t) = toolchain() else { return };
    let dir = scratch("fake-old");
    let (tool, _) = fake_ptxas(&dir, (12, 0));
    let old = t.clone().with_cuda_tools(CudaTools { ptxas: Some(tool), ..CudaTools::default() });
    let kernel = elementwise_add::naive();
    // sm_100 needs PTX ISA 8.6; CUDA 12.0's ptxas assembles 8.0.
    let e = kernel.compile_with(&old, &CompileOptions::new().gpu_arch(Arch::Sm100).use_ptxas(true)).unwrap_err();
    let text = e.to_string();
    assert!(matches!(e, CompileError::Ptxas(_)), "{text}");
    assert!(text.contains("PTX ISA 8.6") && text.contains("release 12.0") && text.contains("8.0 at most"), "{text}");
    assert!(text.contains("use_ptxas(false)"), "{text}");
    // Turned off, the same compile goes through.
    assert!(kernel.compile_with(&old, &CompileOptions::new().gpu_arch(Arch::Sm100).use_ptxas(false)).is_ok());

    let _ = std::fs::remove_dir_all(&dir);

    // ptxas rejecting the PTX (the fake refuses PTX containing FAIL, which a hook can add).
    let dir = scratch("fake-fail");
    let (tool, _) = fake_ptxas(&dir, (12, 8));
    let t = t.with_cuda_tools(CudaTools { ptxas: Some(tool), ..CudaTools::default() });
    // The fake refuses PTX that contains FAIL, and a kernel named FAIL_kernel puts it there.
    let failing = Kernel::builder("FAIL_kernel").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major(32));
        a.store(thread_idx().x, 1.0f32);
    });
    let e = failing.compile_with(&t, &CompileOptions::new().gpu_arch(Arch::Sm80).use_ptxas(true)).unwrap_err();
    assert!(matches!(e, CompileError::Ptxas(_)), "{e}");
    let text = e.to_string();
    assert!(text.contains("ptxas fatal   : refused") && text.contains("-arch=sm_80"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn flags_without_ptxas_map_onto_the_jit_or_name_ptxas() {
    let Some(t) = toolchain() else { return };
    let t = bare(t);
    let kernel = elementwise_add::naive();
    let base = CompileOptions::new().gpu_arch(Arch::Sm80);
    let ok = kernel.compile_with(&t, &base.clone().ptxas_options("--maxrregcount=32 -v -O1")).unwrap();
    let jit = ok.jit_options();
    assert!(jit.contains(&JitOption::MaxRegisters(32)) && jit.contains(&JitOption::LogVerbose(true)), "{jit:?}");
    assert!(jit.contains(&JitOption::OptimizationLevel(1)), "{jit:?}");
    let e = kernel.compile_with(&t, &base.ptxas_options("--warn-on-spills")).unwrap_err().to_string();
    assert!(
        e.contains("ptxas") && e.contains("--warn-on-spills") && e.contains("no CUDA driver JIT equivalent"),
        "{e}"
    );
}

#[test]
fn kept_artifacts_are_written_and_missing_ones_reported() {
    let Some(t) = toolchain() else { return };
    let t = bare(t);
    let dir = scratch("keep");
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).keep_all(true).dump_dir(&dir);
    let compiled = elementwise_add::naive().compile_with(&t, &options).unwrap();
    let names = files(&dir);
    let stem = |ext: &str| {
        names
            .iter()
            .find(|n| n.ends_with(ext) && (ext != ".mlir" || !n.ends_with(".llvm.mlir")))
            .unwrap_or_else(|| panic!("no {ext} in {names:?}"))
    };
    let (ir, mlir, llvm, ptx) = (stem(".ir.txt"), stem(".mlir"), stem(".llvm.mlir"), stem(".ptx"));
    assert_eq!(names.len(), 4, "{names:?}");
    let prefix = "elementwise_add_naive_sm_80_";
    assert!([ir, mlir, llvm, ptx].iter().all(|n| n.starts_with(prefix)), "{names:?}");
    // One hash for the variant.
    let hash = |n: &str| n[prefix.len()..prefix.len() + 8].to_string();
    assert!(hash(ir) == hash(ptx) && hash(mlir) == hash(llvm) && hash(ir) == hash(mlir));
    let read = |n: &String| std::fs::read_to_string(dir.join(n)).unwrap();
    assert_eq!(read(ptx), compiled.ptx());
    assert_eq!(read(ir), compiled.ir());
    assert_eq!(read(mlir), compiled.mlir());
    assert!(read(mlir).contains("loc("), "the kept MLIR has locations");
    assert!(read(ir).starts_with("kernel elementwise_add_naive("));
    assert!(read(llvm).contains("llvm.func") && compiled.llvm().is_some());
    // The cubin and SASS cannot be made here, and the result says so.
    let missing: Vec<_> = compiled.artifacts().iter().filter(|n| n.unavailable.is_some()).collect();
    assert_eq!(missing.iter().map(|n| n.artifact).collect::<Vec<_>>(), [Artifact::Cubin, Artifact::Sass]);
    assert!(missing[0].unavailable.as_ref().unwrap().contains("ptxas"), "{missing:?}");
    assert!(missing.iter().all(|n| n.path.is_none()));
    assert_eq!(compiled.artifacts().iter().filter(|n| n.path.is_some()).count(), 4);
    assert!(compiled.cubin().is_none() && compiled.sass().is_none());
    let _ = std::fs::remove_dir_all(&dir);

    // Without a request nothing is written, and the text artifacts are still on the result.
    let compiled = elementwise_add::naive().compile_with(&t, &CompileOptions::new().gpu_arch(Arch::Sm80)).unwrap();
    assert!(compiled.artifacts().is_empty() && compiled.llvm().is_none());
    assert!(compiled.mlir().contains("gpu.module") && compiled.ir().contains("kernel "));
}

#[test]
fn variants_of_one_kernel_get_different_files() {
    let Some(t) = toolchain() else { return };
    let t = bare(t);
    let dir = scratch("variants");
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).keep_ptx(true).dump_dir(&dir);
    let a = tensorop_gemm::build(tensorop_gemm::GemmConfig::new(256, 128, 96));
    let b = tensorop_gemm::build(tensorop_gemm::GemmConfig::new(256, 128, 160));
    assert_eq!(a.name(), b.name());
    for k in [&a, &b] {
        k.compile_with(&t, &options).unwrap();
    }
    // The same one again overwrites its own file.
    a.compile_with(&t, &options).unwrap();
    assert_eq!(files(&dir).len(), 2, "{:?}", files(&dir));
    // And an optimization level makes another PTX.
    a.compile_with(&t, &options.clone().opt_level(0)).unwrap();
    assert_eq!(files(&dir).len(), 3, "{:?}", files(&dir));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn debug_mode_keeps_everything_with_locations() {
    let Some(t) = toolchain() else { return };
    let t = bare(t);
    let dir = scratch("debug");
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).debug(true).dump_dir(&dir);
    let compiled = elementwise_add::naive().compile_with(&t, &options).unwrap();
    assert!(compiled.line_info() && compiled.ptx().contains(".loc"));
    assert_eq!(files(&dir).len(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unwritable_dump_dir_is_reported_not_skipped() {
    let Some(t) = toolchain() else { return };
    let t = bare(t);
    let file = scratch("notadir");
    std::fs::write(&file, "x").unwrap();
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).keep_ptx(true).dump_dir(file.join("inside"));
    let compiled = elementwise_add::naive().compile_with(&t, &options).unwrap();
    let note = &compiled.artifacts()[0];
    assert!(note.path.is_none() && note.unavailable.as_ref().unwrap().contains("writing"), "{note:?}");
    let _ = std::fs::remove_file(&file);
}

// ---- with a real CUDA toolkit ----

fn real_tools(t: &Toolchain) -> Option<CudaTools> {
    let tools = t.cuda_tools().clone();
    if tools.ptxas.is_none() {
        eprintln!("skipping: no ptxas (CUTE_RS_PTXAS, CUDA_HOME, PATH, /usr/local/cuda/bin)");
        return None;
    }
    Some(tools)
}

/// A kernel with no launch bounds (a known block size becomes `.maxntid`, and ptxas then
/// ignores `--maxrregcount`) that keeps 80 loaded values live at once.
fn register_hog() -> Kernel {
    Kernel::build("register_hog", || {
        let xs = param_tensor::<f32>("xs", Layout::col_major(4096));
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let loaded: Vec<Float32> = (0..80).map(|i| xs.load(t + i * 40)).collect();
        let mut sum = loaded[79];
        for (i, v) in loaded.iter().enumerate().take(79) {
            sum = sum + *v * loaded[(i * 7 + 3) % 80];
        }
        out.store(t, sum);
    })
}

fn registers(log: &str) -> u32 {
    let rest = &log[log.find("Used ").expect("ptxas -v reports registers") + 5..];
    rest.split_whitespace().next().unwrap().parse().unwrap()
}

#[test]
fn real_ptxas_makes_a_cubin_and_obeys_maxrregcount() {
    let Some(t) = toolchain() else { return };
    let Some(_) = real_tools(&t) else { return };
    let kernel = register_hog();
    let base = CompileOptions::new().gpu_arch(Arch::Sm80);
    let base = base.use_ptxas(true);
    let plain = kernel.compile_with(&t, &base).unwrap();
    let cubin = plain.cubin().expect("ptxas ran");
    assert_eq!(&cubin[..4], b"\x7fELF");
    let (free, capped) = (
        registers(plain.ptxas_log().unwrap()),
        registers(
            kernel.compile_with(&t, &base.clone().ptxas_options("--maxrregcount=32")).unwrap().ptxas_log().unwrap(),
        ),
    );
    eprintln!("registers: {free} free, {capped} with --maxrregcount=32");
    assert!(capped <= 32, "{capped}");
    assert!(capped < free, "{capped} < {free}: pick a kernel that uses more than 32 registers");
}

#[test]
fn real_nvdisasm_shows_rust_lines() {
    let Some(t) = toolchain() else { return };
    let Some(tools) = real_tools(&t) else { return };
    if tools.nvdisasm.is_none() && tools.cuobjdump.is_none() {
        eprintln!("skipping: no nvdisasm or cuobjdump");
        return;
    }
    let dir = scratch("sass");
    let kernel = elementwise_add::naive();
    let options = CompileOptions::new().gpu_arch(Arch::Sm80).keep_cubin(true).keep_sass(true).dump_dir(&dir);
    let compiled = kernel.compile_with(&t, &options.clone().generate_line_info(true)).unwrap();
    let sass = compiled.sass().expect("SASS was made");
    assert!(sass.contains("elementwise_add_naive") && sass.contains("EXIT"), "{sass}");
    if tools.nvdisasm.is_some() {
        assert!(sass.contains("elementwise_add.rs"), "SASS with line info names the Rust file:\n{sass}");
    }
    assert!(compiled.artifacts().iter().all(|n| n.path.is_some()), "{:?}", compiled.artifacts());
    assert!(files(&dir).iter().any(|n| n.ends_with(".cubin")) && files(&dir).iter().any(|n| n.ends_with(".sass")));
    // Without line information the listing has no Rust file.
    let plain = kernel.compile_with(&t, &options).unwrap();
    if tools.nvdisasm.is_some() {
        assert!(!plain.sass().unwrap().contains("elementwise_add.rs"));
    }
    // And a failing disassembler options string is an artifact error, not a skip.
    let bad = kernel.compile_with(&t, &options.clone().nvdisasm_options("--no-such-flag")).unwrap();
    if tools.nvdisasm.is_some() {
        assert!(bad.sass().is_none());
        assert!(bad.artifacts().iter().any(|n| n.artifact == Artifact::Sass && n.unavailable.is_some()));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

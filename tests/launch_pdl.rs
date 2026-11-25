//! Programmatic dependent launch: `griddepcontrol_wait` and `griddepcontrol_launch_dependents`
//! (the validator, the MLIR and PTX, `ptxas` when it is on `PATH`), the simulator's two-kernel
//! PDL model ([`SimStream`]), the port of `programmatic_dependent_launch.py`, and the
//! cooperative and PDL fields of [`LaunchConfig`].

#[allow(dead_code)]
#[path = "../examples/programmatic_dependent_launch.rs"]
mod programmatic_dependent_launch;

use std::path::PathBuf;
use std::process::Command;

use cute::prelude::*;
use cute::validate::check_target;

const N: usize = 64;

fn cfg() -> LaunchConfig {
    LaunchConfig::new([2, 1, 1], [32, 1, 1])
}

/// `w[i] = u[i] + 1`, then a hint for the dependent.
fn producer() -> Kernel {
    Kernel::builder("producer").block_dim([32, 1, 1]).build(|| {
        let u = param_tensor::<f32>("u", Layout::col_major(N));
        let w = param_tensor::<f32>("w", Layout::col_major(N));
        let i = block_idx().x * 32 + thread_idx().x;
        w.store(i, u.load(i) + 1.0f32);
        griddepcontrol_launch_dependents();
    })
}

/// The dependent's three shapes: correct (`x` before the wait, `w` after), reading `w`
/// before the wait, and overwriting `u` (the producer's input) before the wait.
#[derive(Clone, Copy, PartialEq)]
enum Dependent {
    Correct,
    ReadsOutputEarly,
    OverwritesInputEarly,
}

/// `y[i] = w[i] + x[i]`, and in one shape a write to `u`.
fn dependent(shape: Dependent) -> Kernel {
    Kernel::builder("dependent").block_dim([32, 1, 1]).build(move || {
        let u = param_tensor::<f32>("u", Layout::col_major(N));
        let w = param_tensor::<f32>("w", Layout::col_major(N));
        let x = param_tensor::<f32>("x", Layout::col_major(N));
        let y = param_tensor::<f32>("y", Layout::col_major(N));
        let i = block_idx().x * 32 + thread_idx().x;
        let xi = x.load(i);
        match shape {
            Dependent::Correct => {
                griddepcontrol_wait();
                y.store(i, w.load(i) + xi);
            }
            Dependent::ReadsOutputEarly => {
                let wi = w.load(i);
                griddepcontrol_wait();
                y.store(i, wi + xi);
            }
            Dependent::OverwritesInputEarly => {
                u.store(i, 0.0f32);
                griddepcontrol_wait();
                y.store(i, w.load(i) + xi);
            }
        }
    })
}

/// Run the producer, then `dep` with `use_pdl`, on one stream.
fn run_pair(dep: &Kernel, use_pdl: bool) -> (SimStream, [SimBuffer; 4], Result<SimReport, SimError>) {
    let mut stream = SimStream::new();
    let u = stream.buffer(KernelArg::buffer(&(0..N).map(|i| i as f32).collect::<Vec<_>>()));
    let w = stream.buffer(KernelArg::buffer(&[0.0f32; N]));
    let x = stream.buffer(KernelArg::buffer(&[10.0f32; N]));
    let y = stream.buffer(KernelArg::buffer(&[0.0f32; N]));
    stream.launch(&producer(), cfg(), &[u.into(), w.into()]).unwrap_or_else(|e| panic!("{e}"));
    let r = stream.launch(dep, cfg().with_pdl(use_pdl), &[u.into(), w.into(), x.into(), y.into()]);
    (stream, [u, w, x, y], r)
}

#[test]
fn the_pdl_model_passes_a_correct_pair() {
    let (stream, [_, _, _, y], r) = run_pair(&dependent(Dependent::Correct), true);
    r.unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<f32> = (0..N).map(|i| i as f32 + 11.0).collect();
    assert_eq!(stream.get(y).to_vec::<f32>(), want);
}

#[test]
fn the_pdl_model_catches_a_read_of_the_producers_output_before_the_wait() {
    let (_, _, r) = run_pair(&dependent(Dependent::ReadsOutputEarly), true);
    let e = r.unwrap_err();
    assert!(
        e.message.contains("reads `w`[0] before griddepcontrol_wait()")
            && e.message.contains("the primary kernel `producer` writes it"),
        "{e}"
    );
    assert!(e.to_string().contains("tests/launch_pdl.rs"), "the access's line: {e}");
}

#[test]
fn the_pdl_model_catches_an_overwrite_of_the_producers_input_before_the_wait() {
    let (_, _, r) = run_pair(&dependent(Dependent::OverwritesInputEarly), true);
    let e = r.unwrap_err();
    assert!(
        e.message.contains("writes `u`[0] before griddepcontrol_wait()")
            && e.message.contains("the primary kernel `producer` reads it"),
        "{e}"
    );
}

/// Without `use_pdl` the dependent runs after the producer, so nothing is checked; a wait in
/// a kernel with no PDL primary (the first launch of a stream, or `Kernel::simulate`) is a
/// no-op.
#[test]
fn without_pdl_nothing_is_checked() {
    let (stream, [_, _, _, y], r) = run_pair(&dependent(Dependent::ReadsOutputEarly), false);
    r.unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(stream.get(y).to_vec::<f32>()[3], 14.0);
    let mut all: Vec<KernelArg> = [1.0f32, 0.0, 2.0, 0.0].iter().map(|&v| KernelArg::buffer(&[v; N])).collect();
    dependent(Dependent::ReadsOutputEarly).simulate(cfg().with_pdl(true), &mut all).unwrap_or_else(|e| panic!("{e}"));
    let mut stream = SimStream::new();
    let bufs: Vec<StreamArg> = all.iter().map(|a| stream.buffer(a.clone()).into()).collect();
    stream.launch(&dependent(Dependent::ReadsOutputEarly), cfg().with_pdl(true), &bufs).unwrap();
}

#[test]
fn griddepcontrol_needs_sm90() {
    let kernel = dependent(Dependent::Correct);
    let e = check_target(kernel.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("griddepcontrol.wait needs sm_90 or newer"), "{e}");
    assert!(e.to_string().contains("tests/launch_pdl.rs"), "the user's line: {e}");
    assert_eq!(check_target(kernel.ir(), Arch::Sm90).unwrap(), 78, "griddepcontrol needs PTX 7.8");
    let e = check_target(producer().ir(), Arch::Sm80).unwrap_err();
    assert!(e.message.contains("griddepcontrol.launch_dependents needs sm_90 or newer"), "{e}");
}

/// The MLIR is CuTe DSL's inline asm with side effects and a memory clobber.
#[test]
fn griddepcontrol_mlir_is_inline_asm_with_a_memory_clobber() {
    let mlir = producer().to_mlir();
    assert!(
        mlir.contains(r#"llvm.inline_asm has_side_effects "griddepcontrol.launch_dependents;", "~{memory}""#),
        "{mlir}"
    );
    let mlir = dependent(Dependent::Correct).to_mlir();
    assert!(mlir.contains(r#"llvm.inline_asm has_side_effects "griddepcontrol.wait;", "~{memory}""#), "{mlir}");
}

fn find_ptxas() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}

/// The sm_90 PTX of the tutorial's kernels holds both instructions, and `ptxas` (when on
/// `PATH`) assembles it.
#[test]
fn pdl_kernels_compile_for_sm90() {
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let options = CompileOptions::new().gpu_arch(Arch::Sm90);
    let first = programmatic_dependent_launch::build(true).compile_with(&toolchain, &options).unwrap();
    let second = programmatic_dependent_launch::build(false).compile_with(&toolchain, &options).unwrap();
    assert!(first.ptx().contains("griddepcontrol.launch_dependents;"), "{}", first.ptx());
    assert!(second.ptx().contains("griddepcontrol.wait;"), "{}", second.ptx());
    let e = programmatic_dependent_launch::build(true)
        .compile_with(&toolchain, &CompileOptions::new().gpu_arch(Arch::Sm89))
        .unwrap_err();
    assert!(e.to_string().contains("needs sm_90"), "{e}");
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping the assembly: no ptxas on PATH");
        return;
    };
    for k in [&first, &second] {
        let stem = std::env::temp_dir().join(format!("cute-pdl-{}-{}", std::process::id(), k.name()));
        let (source, cubin) = (stem.with_extension("ptx"), stem.with_extension("cubin"));
        std::fs::write(&source, k.ptx()).unwrap();
        let out = Command::new(&ptxas).arg("-arch=sm_90").arg(&source).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `programmatic_dependent_launch.py` on the simulator, through the PDL model.
#[test]
fn programmatic_dependent_launch_example() {
    let y = programmatic_dependent_launch::run(40, 200);
    assert_eq!(y.len(), 40 * 200);
}

#[test]
fn launch_config_fields() {
    let c = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    assert!(!c.use_pdl && !c.cooperative);
    let c = c.with_pdl(true).with_cooperative(true);
    assert!(c.use_pdl && c.cooperative);
}

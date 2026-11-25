// SPDX-License-Identifier: MIT OR Apache-2.0

//! A differential check of the simulator against a real GPU.
//!
//! With the environment variable `CUTE_RS_CROSSCHECK_GPU` set to a file path, every successful
//! [`Kernel::simulate`](crate::Kernel::simulate) or `simulate_with` also compiles the kernel for
//! GPU 0, runs it there on a copy of the arguments as they were before the simulation, and
//! appends one line to that file:
//!
//! * `SAME`: every buffer the device wrote equals the simulator's, bit for bit;
//! * `DIFF`: some buffer differs, with the argument, how many elements differ, the first one
//!   and, for floating-point buffers, the largest absolute and relative difference;
//! * `SKIP`: the kernel did not run on the device, with the reason (the device cannot run the
//!   architecture the kernel was built for, or the compile or the load failed);
//! * `FAIL`: the launch itself failed.
//!
//! A `RUN` line is written before each launch, so a kernel that never returns on the device
//! can be found from the last line of the log. Nothing here panics and the simulation's own
//! result is returned unchanged, so a whole simulator test suite can be run against a GPU:
//!
//! ```text
//! CUTE_RS_CROSSCHECK_GPU=/tmp/crosscheck.log cargo test --release --features cuda
//! ```
//!
//! `CUTE_RS_CROSSCHECK_DUMP=<directory>` also saves the buffers of every kernel that differs
//! (before the run, after the device and after the simulator), for looking at afterwards.
//!
//! `CUTE_RS_CROSSCHECK_REPLAY=<directory>` uses such a directory in place of a device, on
//! any machine: a simulated kernel that has a saved run with the same inputs is compared with
//! what the device gave then (`SAME` or `DIFF`; `NONE` when its saved runs have other inputs).
//!
//! The kernel is compiled without floating-point contraction, as the simulator rounds every
//! operation. Results that depend on how threads interleave (atomics, a racy reduction) and
//! tensor-core arithmetic on values that are not exactly representable can still differ.

use std::cell::RefCell;
use std::io::Write;
use std::panic::Location;
use std::sync::{Mutex, OnceLock};

use super::CudaContext;
use crate::ir::DType;
use crate::sim::{KernelArg, LaunchConfig};
use crate::{CompileOptions, Kernel, Toolchain};

/// The environment variable naming the log file; the check is off without it.
pub const ENV: &str = "CUTE_RS_CROSSCHECK_GPU";

fn log_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| std::env::var(ENV).ok().filter(|s| !s.is_empty())).as_deref()
}

/// The arguments as they are now, if the check is on.
pub(crate) fn snapshot(args: &[KernelArg]) -> Option<Vec<KernelArg>> {
    log_path().map(|_| args.to_vec())
}

fn log(line: &str) {
    static LOCK: Mutex<()> = Mutex::new(());
    let Some(path) = log_path() else { return };
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

fn toolchain() -> Option<&'static Toolchain> {
    static TOOLCHAIN: OnceLock<Option<Toolchain>> = OnceLock::new();
    TOOLCHAIN.get_or_init(Toolchain::discover).as_ref()
}

thread_local! {
    /// This thread's handle on device 0's primary context.
    static CONTEXT: RefCell<Option<Result<CudaContext, String>>> = const { RefCell::new(None) };
}

/// Run `kernel` on the device with `before` (the arguments before the simulation) and log how
/// its buffers compare with `sim` (the arguments after it).
pub(crate) fn run(kernel: &Kernel, cfg: LaunchConfig, before: Vec<KernelArg>, sim: &[KernelArg], at: &Location<'_>) {
    let thread = std::thread::current();
    let tag = format!("{} [{}] {}:{}", kernel.name(), thread.name().unwrap_or("?"), at.file(), at.line());
    if let Some(dir) = replay_dir() {
        return replay(dir, kernel.name(), &tag, &before, sim);
    }
    let Some(toolchain) = toolchain() else { return log(&format!("SKIP {tag}: no mlir-opt")) };
    CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        let ctx = match slot.get_or_insert_with(|| CudaContext::new(0).map_err(|e| e.to_string())) {
            Ok(ctx) => ctx,
            Err(e) => return log(&format!("SKIP {tag}: no device: {e}")),
        };
        let Some(device) = ctx.arch() else { return log(&format!("SKIP {tag}: the device is older than sm_80")) };
        // A kernel with a target of its own is compiled for it, and the driver compiles that
        // PTX for this device at load time. An older baseline target runs on a newer device;
        // an architecture-specific one (`sm_90a`) runs on its own chip only.
        let arch = match kernel.target() {
            Some(target) if target == device || (!target.is_arch_specific() && target.sm() <= device.sm()) => target,
            Some(target) => return log(&format!("SKIP {tag}: built for {target}, the device is {device}")),
            None => device,
        };
        let options = CompileOptions::new().gpu_arch(arch).fp_contract(false);
        let compiled = match kernel.compile_with(toolchain, &options) {
            Ok(compiled) => compiled,
            Err(e) => return log(&format!("SKIP {tag}: compile for {arch}: {}", first_line(&e.to_string()))),
        };
        let function = match ctx.load(&compiled) {
            Ok(function) => function,
            Err(e) => return log(&format!("SKIP {tag}: load: {}", first_line(&e.to_string()))),
        };
        let inputs = dump_dir().map(|_| before.clone());
        let mut gpu = before;
        log(&format!("RUN  {tag}"));
        if let Err(e) = ctx.launch(kernel, &function, cfg, &mut gpu) {
            return log(&format!("FAIL {tag}: launch: {}", first_line(&e.to_string())));
        }
        match compare(&gpu, sim) {
            None => log(&format!("SAME {tag}")),
            Some(diff) => {
                let saved = inputs.and_then(|inputs| dump(kernel.name(), &inputs, &gpu, sim));
                log(&format!("DIFF {tag}: {diff}{}", saved.map(|s| format!(" [saved {s}]")).unwrap_or_default()));
            }
        }
    });
}

/// `CUTE_RS_CROSSCHECK_DUMP`: a directory that receives, for each kernel that differs, the
/// buffers before the run (`in`), after the device (`gpu`) and after the simulator (`sim`), as
/// raw little-endian files `<n>_<kernel>.arg<i>.<in|gpu|sim>.bin`.
fn dump_dir() -> Option<&'static str> {
    static DIR: OnceLock<Option<String>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var("CUTE_RS_CROSSCHECK_DUMP").ok().filter(|s| !s.is_empty())).as_deref()
}

fn dump(kernel: &str, inputs: &[KernelArg], gpu: &[KernelArg], sim: &[KernelArg]) -> Option<String> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = dump_dir()?;
    std::fs::create_dir_all(dir).ok()?;
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = format!("{}_{n}_{kernel}", std::process::id());
    for (i, ((before, g), s)) in inputs.iter().zip(gpu).zip(sim).enumerate() {
        if s.buffer_type().is_none() {
            continue;
        }
        for (what, arg) in [("in", before), ("gpu", g), ("sim", s)] {
            std::fs::write(format!("{dir}/{stem}.arg{i}.{what}.bin"), arg.to_bytes()).ok()?;
        }
    }
    Some(stem)
}

/// `CUTE_RS_CROSSCHECK_REPLAY`: a directory of buffers saved by `CUTE_RS_CROSSCHECK_DUMP` on a
/// machine with a GPU. With it set no device is used: a simulated kernel that has a saved run
/// with the same inputs is compared with what the device gave then.
fn replay_dir() -> Option<&'static str> {
    static DIR: OnceLock<Option<String>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var("CUTE_RS_CROSSCHECK_REPLAY").ok().filter(|s| !s.is_empty())).as_deref()
}

/// The saved runs in `dir`, by kernel name: the stems `<pid>_<n>_<kernel>` of their files.
fn recorded(dir: &'static str) -> &'static std::collections::HashMap<String, Vec<String>> {
    static INDEX: OnceLock<std::collections::HashMap<String, Vec<String>>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut index: std::collections::HashMap<String, Vec<String>> = Default::default();
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let file = entry.file_name().to_string_lossy().into_owned();
            let Some((stem, rest)) = file.split_once(".arg") else { continue };
            let mut parts = stem.splitn(3, '_');
            let (Some(_), Some(_), Some(kernel)) = (parts.next(), parts.next(), parts.next()) else { continue };
            if rest.ends_with(".gpu.bin") {
                let stems = index.entry(kernel.to_string()).or_default();
                if !stems.iter().any(|s| s == stem) {
                    stems.push(stem.to_string());
                }
            }
        }
        index
    })
}

/// Compare `sim` with the device's saved buffers of a run of `kernel` on the inputs `before`.
fn replay(dir: &'static str, kernel: &str, tag: &str, before: &[KernelArg], sim: &[KernelArg]) {
    let Some(stems) = recorded(dir).get(kernel) else { return };
    let read = |stem: &str, i: usize, what: &str| std::fs::read(format!("{dir}/{stem}.arg{i}.{what}.bin")).ok();
    let buffers: Vec<usize> = (0..sim.len()).filter(|&i| sim[i].buffer_type().is_some()).collect();
    let same_inputs =
        |stem: &str| buffers.iter().all(|&i| read(stem, i, "in").is_some_and(|b| b == before[i].to_bytes()));
    let Some(stem) = stems.iter().find(|stem| same_inputs(stem)) else {
        return log(&format!("NONE {tag}: {} saved runs, none with these inputs", stems.len()));
    };
    let mut gpu = sim.to_vec();
    for &i in &buffers {
        match read(stem, i, "gpu") {
            Some(bytes) if bytes.len() == sim[i].to_bytes().len() => gpu[i].set_from_bytes(&bytes),
            _ => return log(&format!("NONE {tag}: the saved run {stem} lacks argument {i}")),
        }
    }
    match compare(&gpu, sim) {
        None => log(&format!("SAME {tag} (saved {stem})")),
        Some(diff) => log(&format!("DIFF {tag} (saved {stem}): {diff}")),
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").chars().take(300).collect()
}

/// A description of the first buffer that differs, or `None` if every buffer is the same.
fn compare(gpu: &[KernelArg], sim: &[KernelArg]) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    for (i, (g, s)) in gpu.iter().zip(sim).enumerate() {
        let Some((elem, len)) = s.buffer_type() else { continue };
        let (gb, sb) = (g.to_bytes(), s.to_bytes());
        if gb == sb {
            continue;
        }
        let width = (sb.len() / len.max(1)).max(1);
        let differing: Vec<usize> =
            (0..len).filter(|&e| gb.get(e * width..(e + 1) * width) != sb.get(e * width..(e + 1) * width)).collect();
        let first = differing.first().copied().unwrap_or(0);
        let hex = |b: &[u8]| b.iter().rev().map(|x| format!("{x:02x}")).collect::<String>();
        let slice = |b: &[u8]| b.get(first * width..(first + 1) * width).map(hex).unwrap_or_default();
        let mut text = format!(
            "arg {i} ({elem}): {} of {len} differ, first at {first}: gpu 0x{} sim 0x{}",
            differing.len(),
            slice(&gb),
            slice(&sb)
        );
        if let Some((abs, rel)) = float_gap(elem, &gb, &sb, &differing, width) {
            text.push_str(&format!(", max abs {abs:.3e} rel {rel:.3e}"));
        }
        out.push(text);
    }
    (!out.is_empty()).then(|| out.join("; "))
}

/// The largest absolute and relative difference over `differing`, for the float types read here.
fn float_gap(elem: DType, gpu: &[u8], sim: &[u8], differing: &[usize], width: usize) -> Option<(f64, f64)> {
    let read: fn(&[u8]) -> f64 = match (elem, width) {
        (DType::F32, 4) => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
        (DType::F64, 8) => |b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        (DType::F16, 2) => |b| crate::F16(u16::from_le_bytes([b[0], b[1]])).to_f32() as f64,
        (DType::BF16, 2) => |b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64,
        _ => return None,
    };
    let (mut abs, mut rel) = (0.0f64, 0.0f64);
    for &e in differing {
        let (g, s) = (read(&gpu[e * width..(e + 1) * width]), read(&sim[e * width..(e + 1) * width]));
        let d = (g - s).abs();
        if d.is_nan() {
            // One side is NaN and the other is not (two NaNs with different payloads count too).
            return Some((f64::NAN, f64::NAN));
        }
        abs = abs.max(d);
        rel = rel.max(d / s.abs().max(f64::MIN_POSITIVE));
    }
    Some((abs, rel))
}

//! The logger: output at each level, nothing formatted when it is off, and the lines the
//! pipeline writes. One test function, because the logger is process-wide.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use cute::log::{self, Level};
use cute::prelude::*;

/// Counts how often it is formatted.
struct Probe(&'static AtomicUsize);

impl fmt::Display for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fetch_add(1, Ordering::SeqCst);
        f.write_str("probe")
    }
}

static FORMATTED: AtomicUsize = AtomicUsize::new(0);

fn file(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("cute-rs-log-test-{}-{name}.log", std::process::id()))
}

#[test]
fn the_logger() {
    // Off by default: nothing is formatted, whatever the level.
    log::log_to_console(false);
    log::log_to_file(None).unwrap();
    for level in [Level::Error, Level::Info, Level::Trace] {
        cute::cute_log!(level, "never {}", Probe(&FORMATTED));
        assert!(!log::enabled(level));
    }
    assert_eq!(FORMATTED.load(Ordering::SeqCst), 0, "a disabled log call formatted its arguments");

    // Each level prints itself and the more important ones.
    let all = [Level::Error, Level::Warn, Level::Info, Level::Debug, Level::Trace];
    for (i, set) in all.into_iter().enumerate() {
        let path = file(set.name());
        let _ = std::fs::remove_file(&path);
        log::log_to_file(Some(&path)).unwrap();
        log::set_level(set);
        for level in all {
            cute::cute_log!(level, "{} line {}", level.name(), Probe(&FORMATTED));
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), i + 1, "{set:?}: {text}");
        for (line, level) in lines.iter().zip(all) {
            assert_eq!(*line, format!("cute-rs: {0}: {0} line probe", level.name()));
        }
        // Only the lines that were written were formatted.
        assert_eq!(FORMATTED.swap(0, Ordering::SeqCst), i + 1);
        let _ = std::fs::remove_file(&path);
    }

    // Switching the sink off again silences it, and the level check is what decides.
    log::log_to_file(None).unwrap();
    cute::cute_log!(Level::Error, "{}", Probe(&FORMATTED));
    assert_eq!(FORMATTED.load(Ordering::SeqCst), 0);

    // The pipeline logs the build, its stages with their times, and the compile.
    let Some(toolchain) = Toolchain::discover() else {
        eprintln!("skipping the pipeline lines: no mlir-opt");
        return;
    };
    let path = file("pipeline");
    let _ = std::fs::remove_file(&path);
    log::log_to_file(Some(&path)).unwrap();
    log::set_level(Level::Info);
    let kernel = Kernel::builder("logged_kernel").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major(32));
        a.store(thread_idx().x, 1.0f32);
    });
    kernel.compile_ptx(&toolchain, Arch::Sm80).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    for want in [
        "building kernel `logged_kernel`",
        "built kernel `logged_kernel` in ",
        "compiling kernel `logged_kernel` for sm_80",
        "lowering stage 1",
        "lowering stage 2",
        "compiled in ",
    ] {
        assert!(text.contains(want), "missing {want:?} in\n{text}");
    }
    // Debug and trace lines stay out at info level.
    assert!(!text.contains(": debug: "));
    log::log_to_file(None).unwrap();
    let _ = std::fs::remove_file(&path);
}

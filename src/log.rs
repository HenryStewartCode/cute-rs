//! A dependency-free logger (CuTe DSL's `CUTE_DSL_LOG_*`): lines go to stderr, to a file, or
//! both, at one of five levels.
//!
//! The environment decides, once, the first time anything logs:
//!
//! * `CUTE_RS_LOG_TO_CONSOLE`: `1` (also `true`, `yes`, `on`) writes to stderr;
//! * `CUTE_RS_LOG_TO_FILE`: a path to append to (created if missing);
//! * `CUTE_RS_LOG_LEVEL`: `error`, `warn`, `info`, `debug` or `trace`, or one of Python's
//!   numbers as CuTe DSL takes them (up to 10 is `debug`, 20 `info`, 30 `warn`, above that
//!   `error`). Without it, `debug`: with a sink on, everything but `trace`.
//!
//! With neither sink the logger is off, and logging costs one relaxed atomic load: the level is
//! checked before anything is formatted. Programs can configure it instead of the environment
//! with [`set_level`], [`log_to_console`] and [`log_to_file`].
//!
//! cute-rs logs the kernel build (`info`), each pipeline stage with its time (`info`), the
//! `ptxas` and `nvdisasm` command lines (`debug`), module loads and launches with their grid,
//! block, shared memory and attributes (`debug`), and artifacts that were asked for and could
//! not be produced (`warn`).

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// How important a log line is; a logger set to a level prints that level and the more
/// important ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    /// The lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }

    /// A level from its name (any case) or from one of Python's logging numbers.
    pub fn parse(text: &str) -> Option<Level> {
        let t = text.trim().to_ascii_lowercase();
        match t.as_str() {
            "error" | "critical" | "fatal" => Some(Level::Error),
            "warn" | "warning" => Some(Level::Warn),
            "info" => Some(Level::Info),
            "debug" => Some(Level::Debug),
            "trace" => Some(Level::Trace),
            _ => t.parse::<i64>().ok().map(|n| match n {
                ..=10 => Level::Debug,
                11..=20 => Level::Info,
                21..=30 => Level::Warn,
                _ => Level::Error,
            }),
        }
    }
}

const UNINITIALISED: u8 = u8::MAX;
const OFF: u8 = 0;

/// The most detailed level that is printed, or `OFF`: the one thing the fast path reads.
static MAX_LEVEL: AtomicU8 = AtomicU8::new(UNINITIALISED);

struct Sinks {
    console: bool,
    file: Option<File>,
    level: Level,
}

fn sinks() -> &'static Mutex<Sinks> {
    static SINKS: OnceLock<Mutex<Sinks>> = OnceLock::new();
    SINKS.get_or_init(|| Mutex::new(Sinks { console: false, file: None, level: Level::Debug }))
}

fn lock() -> std::sync::MutexGuard<'static, Sinks> {
    sinks().lock().unwrap_or_else(|e| e.into_inner())
}

fn publish(s: &Sinks) {
    let on = s.console || s.file.is_some();
    MAX_LEVEL.store(if on { s.level as u8 } else { OFF }, Ordering::Relaxed);
}

/// Read the environment (once; programs that configured the logger themselves keep their
/// settings) and return the level in force.
#[cold]
fn init_from_env() -> u8 {
    let mut s = lock();
    if MAX_LEVEL.load(Ordering::Relaxed) == UNINITIALISED {
        s.console =
            std::env::var("CUTE_RS_LOG_TO_CONSOLE").ok().and_then(|v| crate::backend::parse_bool(&v)).unwrap_or(false);
        if let Some(path) = std::env::var_os("CUTE_RS_LOG_TO_FILE").filter(|p| !p.is_empty()) {
            match open_log(Path::new(&path)) {
                Ok(f) => s.file = Some(f),
                Err(e) => eprintln!("cute-rs: warning: CUTE_RS_LOG_TO_FILE={}: {e}", Path::new(&path).display()),
            }
        }
        if let Some(text) = std::env::var("CUTE_RS_LOG_LEVEL").ok().filter(|t| !t.trim().is_empty()) {
            match Level::parse(&text) {
                Some(l) => s.level = l,
                None => eprintln!(
                    "cute-rs: warning: CUTE_RS_LOG_LEVEL={text:?} is not error, warn, info, debug, trace or a number"
                ),
            }
        }
        publish(&s);
    }
    MAX_LEVEL.load(Ordering::Relaxed)
}

fn open_log(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

/// Whether a line at `level` would be written. This is the check every log call makes before
/// it formats anything.
#[inline]
pub fn enabled(level: Level) -> bool {
    let mut max = MAX_LEVEL.load(Ordering::Relaxed);
    if max == UNINITIALISED {
        max = init_from_env();
    }
    level as u8 <= max
}

/// Set the level, whatever the environment says.
pub fn set_level(level: Level) {
    let mut s = lock();
    ensure_env_read();
    s.level = level;
    publish(&s);
}

/// Switch writing to stderr on or off.
pub fn log_to_console(on: bool) {
    let mut s = lock();
    ensure_env_read();
    s.console = on;
    publish(&s);
}

/// Append to the file at `path` (`None` stops writing to a file).
pub fn log_to_file(path: Option<&Path>) -> std::io::Result<()> {
    let mut s = lock();
    ensure_env_read();
    s.file = path.map(open_log).transpose()?;
    publish(&s);
    Ok(())
}

/// A programmatic setting replaces the environment's, so the environment must not be read
/// afterwards on top of it.
fn ensure_env_read() {
    let _ = MAX_LEVEL.compare_exchange(UNINITIALISED, OFF, Ordering::Relaxed, Ordering::Relaxed);
}

/// Write one line at `level` if it is enabled. Prefer the formatting entry point
/// [`log`], which does not build the message when the level is off.
pub fn log(level: Level, message: fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }
    let mut s = lock();
    let line = format!("cute-rs: {}: {message}\n", level.name());
    if s.console {
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    if let Some(f) = s.file.as_mut() {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Log with `format!`-style arguments at a level, formatting nothing when it is off:
/// `cute_log!(Level::Info, "built {name}")`.
#[macro_export]
macro_rules! cute_log {
    ($level:expr, $($arg:tt)*) => {
        if $crate::log::enabled($level) {
            $crate::log::log($level, format_args!($($arg)*));
        }
    };
}

/// A message built by a closure, for lines that are costly to make.
pub(crate) fn log_with(level: Level, message: impl FnOnce() -> String) {
    if enabled(level) {
        log(level, format_args!("{}", message()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse() {
        assert_eq!(Level::parse("INFO"), Some(Level::Info));
        assert_eq!(Level::parse(" trace "), Some(Level::Trace));
        assert_eq!(Level::parse("warning"), Some(Level::Warn));
        assert_eq!(Level::parse("1"), Some(Level::Debug));
        assert_eq!(Level::parse("10"), Some(Level::Debug));
        assert_eq!(Level::parse("20"), Some(Level::Info));
        assert_eq!(Level::parse("30"), Some(Level::Warn));
        assert_eq!(Level::parse("40"), Some(Level::Error));
        assert_eq!(Level::parse("loud"), None);
        assert!(Level::Error < Level::Trace);
    }
}

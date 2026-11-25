//! Run one test of the current test binary in a child process and capture
//! its stdout. The CUDA driver writes device `printf` output (and failed
//! device assertions) to the process's stdout when the host synchronises,
//! (a failed device assertion goes to stderr), so a test that checks it runs
//! the kernel in a child: the parent re-runs
//! the test binary on the one test, with [`CHILD_ENV`] set to select the
//! child role, and reads the child's stdout between [`BEGIN`] and [`END`].

use std::process::Command;

/// Set in the child: the test runs the kernel instead of spawning.
pub const CHILD_ENV: &str = "CUTE_RS_TEST_CHILD";
/// Printed by the child just before the device output starts.
pub const BEGIN: &str = "@@cute-rs-child-begin@@";
/// Printed by the child once the device output is flushed.
pub const END: &str = "@@cute-rs-child-end@@";

/// Whether this process is a child started by [`run_child`].
pub fn is_child() -> bool {
    std::env::var_os(CHILD_ENV).is_some()
}

/// What a child printed and how it ended.
#[derive(Debug)]
pub struct ChildOutput {
    /// The lines of stdout between [`BEGIN`] and [`END`] (to the end of the
    /// output if the child printed no [`END`]), each with its newline
    /// removed.
    pub lines: Vec<String>,
    /// The whole of stdout and stderr, for messages.
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

/// Re-run this test binary on the test `name` alone (its full path, such as
/// `printf_on_gpu`), in a child process with [`CHILD_ENV`] set.
pub fn run_child(name: &str) -> ChildOutput {
    run_child_with(name, &[])
}

/// [`run_child`] with extra environment variables (read by the CUDA driver
/// when it starts, such as `CUDA_CACHE_DISABLE`).
pub fn run_child_with(name: &str, env: &[(&str, &str)]) -> ChildOutput {
    let exe = std::env::current_exe().expect("the test binary's path");
    let out = Command::new(exe)
        .args([name, "--exact", "--nocapture", "--test-threads=1"])
        .envs(env.iter().copied())
        .env(CHILD_ENV, "1")
        .output()
        .expect("the child test process starts");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let lines = match stdout.split_once(BEGIN) {
        Some((_, rest)) => {
            let body = rest.split_once(END).map_or(rest, |(b, _)| b);
            body.strip_prefix('\n').unwrap_or(body).lines().map(str::to_string).collect()
        }
        None => vec![],
    };
    ChildOutput { lines, stdout, stderr, success: out.status.success() }
}

unsafe extern "C" {
    fn fflush(stream: *mut std::ffi::c_void) -> i32;
}

/// In the child: mark the start of the device output.
pub fn begin() {
    use std::io::Write;
    println!("{BEGIN}");
    std::io::stdout().flush().unwrap();
}

/// In the child: flush C's stdio buffers (where the driver's output waits
/// when stdout is a pipe) and mark the end of the device output.
pub fn end() {
    use std::io::Write;
    // SAFETY: fflush(NULL) flushes every open C stream; it takes no Rust
    // state.
    unsafe { fflush(std::ptr::null_mut()) };
    println!("{END}");
    std::io::stdout().flush().unwrap();
}

/// `lines` sorted, since the order in which threads print varies.
pub fn sorted(lines: &[String]) -> Vec<String> {
    let mut v = lines.to_vec();
    v.sort();
    v
}

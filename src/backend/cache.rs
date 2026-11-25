//! Compile caches (CuTe DSL's JIT cache): a process-wide in-memory cache of compiled kernels
//! and an on-disk cache of their PTX and cubins, both keyed on everything that decides the
//! output, and [`KernelCache`], a memo of built and compiled kernels on keys of the user's.
//!
//! The key ([`Kernel::compile_with`](crate::Kernel::compile_with) builds it) is the MLIR text
//! (with source locations only when line information is on, so a change that only moves
//! code keeps the key), the kernel's signature, the target, PTX version, optimization level
//! and the other options that change the output (`CompileOptions::cache_key`), the lowering
//! pipeline, the toolchain (the `mlir-opt` path, size, modification time and LLVM version,
//! the `ptxas` path, size, modification time and CUDA release when it runs) and the crate
//! version. The on-disk key adds the running
//! executable's path, size and modification time, which stand in for CuTe DSL's hash of its
//! own sources: a rebuilt program never reads entries a different build wrote.
//!
//! * In memory: thread-safe, bounded LRU-style by `CUTE_RS_JIT_CACHE_MAX_ELEMS` (default
//!   1024 kernels).
//! * On disk (on by default, as CuTe DSL's file cache is): in `CUTE_RS_CACHE_DIR`, or
//!   `std::env::temp_dir()/cute-rs-<user>`. The directory is created private to the user
//!   (mode 0700) and, before every read or write, must be a real directory owned by the user
//!   that no other user can write; otherwise the disk cache is skipped with a warning, since
//!   whoever can write entries chooses the code that runs on the GPU. Each entry is
//!   named by a SHA-256 of its key, holds the full key (compared on read, so a collision can
//!   never return the wrong kernel), the PTX, the cubin when `ptxas` ran and its log, and a
//!   SHA-256 checksum of the whole; it is written to a temporary file and renamed, so
//!   concurrent processes never read a partial entry. The directory is bounded by
//!   `CUTE_RS_FILE_CACHE_MAX_ENTRIES` (default 256) and `CUTE_RS_FILE_CACHE_MAX_BYTES`
//!   (default 64 MiB), evicting the least recently used first. `CUTE_RS_DISABLE_FILE_CACHING`
//!   turns it off.
//!
//! [`CompileOptions::no_cache`](crate::CompileOptions::no_cache) and `CUTE_RS_NO_CACHE` bypass
//! both, as do stage hooks and kept artifacts (CuTe DSL's keep paths force `no_cache`). Hits
//! and misses are logged at `info` and counted ([`cache_stats`]).

use std::collections::HashMap;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::{CompileError, CompiledKernel};
use crate::builder::Kernel;
use crate::log::Level;

/// `CUTE_RS_JIT_CACHE_MAX_ELEMS`: the most kernels the in-memory cache and each context's
/// module cache keep.
pub const ENV_JIT_CACHE_MAX_ELEMS: &str = "CUTE_RS_JIT_CACHE_MAX_ELEMS";
/// `CUTE_RS_NO_CACHE`: bypass every compile and module cache.
pub const ENV_NO_CACHE: &str = "CUTE_RS_NO_CACHE";
/// `CUTE_RS_CACHE_DIR`: the on-disk cache's directory.
pub const ENV_CACHE_DIR: &str = "CUTE_RS_CACHE_DIR";
/// `CUTE_RS_DISABLE_FILE_CACHING`: turn the on-disk cache off.
pub const ENV_DISABLE_FILE_CACHING: &str = "CUTE_RS_DISABLE_FILE_CACHING";
/// `CUTE_RS_FILE_CACHE_MAX_ENTRIES`: the most entries the on-disk cache keeps.
pub const ENV_FILE_CACHE_MAX_ENTRIES: &str = "CUTE_RS_FILE_CACHE_MAX_ENTRIES";
/// `CUTE_RS_FILE_CACHE_MAX_BYTES`: the most bytes the on-disk cache keeps.
pub const ENV_FILE_CACHE_MAX_BYTES: &str = "CUTE_RS_FILE_CACHE_MAX_BYTES";

const MAGIC: &[u8] = b"CUTE-RS-JIT-CACHE\x01";
const EXTENSION: &str = "cute-cache";

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Whether the environment turns every cache off (`CUTE_RS_NO_CACHE`).
pub fn no_cache_env() -> bool {
    env_flag(ENV_NO_CACHE)
}

/// The most entries of the in-memory caches (`CUTE_RS_JIT_CACHE_MAX_ELEMS`, at least 1).
pub fn max_elems() -> usize {
    env_number(ENV_JIT_CACHE_MAX_ELEMS, 1024).max(1) as usize
}

/// What the compile caches have done in this process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
    /// Compiles that bypassed the caches.
    pub bypassed: u64,
    /// Corrupt or truncated disk entries found (and removed).
    pub corrupt: u64,
    /// Lowerings of a kernel by `mlir-opt` (each one or two `mlir-opt` runs).
    pub mlir_opt_runs: u64,
}

static MEMORY_HITS: AtomicU64 = AtomicU64::new(0);
static DISK_HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static BYPASSED: AtomicU64 = AtomicU64::new(0);
static CORRUPT: AtomicU64 = AtomicU64::new(0);
pub(crate) static MLIR_OPT_RUNS: AtomicU64 = AtomicU64::new(0);

/// The counts so far.
pub fn cache_stats() -> CacheStats {
    CacheStats {
        memory_hits: MEMORY_HITS.load(Ordering::Relaxed),
        disk_hits: DISK_HITS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        bypassed: BYPASSED.load(Ordering::Relaxed),
        corrupt: CORRUPT.load(Ordering::Relaxed),
        mlir_opt_runs: MLIR_OPT_RUNS.load(Ordering::Relaxed),
    }
}

pub(crate) fn count_bypass() {
    BYPASSED.fetch_add(1, Ordering::Relaxed);
}

/// A least-recently-used map bounded by [`max_elems`].
pub(crate) struct Lru<K, V> {
    map: HashMap<K, (u64, V)>,
    tick: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> Lru<K, V> {
    pub(crate) fn new() -> Self {
        Lru { map: HashMap::new(), tick: 0 }
    }

    pub(crate) fn get(&mut self, k: &K) -> Option<V> {
        self.tick += 1;
        let tick = self.tick;
        self.map.get_mut(k).map(|e| {
            e.0 = tick;
            e.1.clone()
        })
    }

    /// Insert, returning what was evicted to stay within `cap`.
    pub(crate) fn put(&mut self, k: K, v: V, cap: usize) -> Vec<V> {
        self.tick += 1;
        self.map.insert(k, (self.tick, v));
        let mut evicted = Vec::new();
        while self.map.len() > cap {
            let oldest = self.map.iter().min_by_key(|(_, (t, _))| *t).map(|(k, _)| k.clone()).expect("non-empty");
            evicted.extend(self.map.remove(&oldest).map(|(_, v)| v));
        }
        evicted
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    pub(crate) fn drain(&mut self) -> Vec<V> {
        self.map.drain().map(|(_, (_, v))| v).collect()
    }
}

fn memory() -> &'static Mutex<Lru<String, Arc<CompiledKernel>>> {
    static MEMORY: OnceLock<Mutex<Lru<String, Arc<CompiledKernel>>>> = OnceLock::new();
    MEMORY.get_or_init(|| Mutex::new(Lru::new()))
}

/// Empty the in-memory compile cache.
pub fn clear_memory_cache() {
    memory().lock().unwrap_or_else(|e| e.into_inner()).drain();
}

/// The number of kernels in the in-memory compile cache.
pub fn memory_cache_len() -> usize {
    memory().lock().unwrap_or_else(|e| e.into_inner()).len()
}

pub(crate) fn memory_get(key: &str) -> Option<Arc<CompiledKernel>> {
    memory().lock().unwrap_or_else(|e| e.into_inner()).get(&key.to_string())
}

pub(crate) fn memory_put(key: &str, kernel: Arc<CompiledKernel>) {
    memory().lock().unwrap_or_else(|e| e.into_inner()).put(key.to_string(), kernel, max_elems());
}

pub(crate) fn note_hit(disk: bool, name: &str) {
    if disk {
        DISK_HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        MEMORY_HITS.fetch_add(1, Ordering::Relaxed);
    }
    crate::cute_log!(Level::Info, "kernel `{name}`: JIT cache hit ({})", if disk { "on disk" } else { "in memory" });
}

pub(crate) fn note_miss(name: &str) {
    MISSES.fetch_add(1, Ordering::Relaxed);
    crate::cute_log!(Level::Info, "kernel `{name}`: JIT cache miss");
}

// ---------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), for stable entry names and checksums.

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

/// The SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K256[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// Lower-case hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------------------
// The on-disk cache.

/// What an on-disk entry holds besides its key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskEntry {
    pub ptx: String,
    pub cubin: Option<Vec<u8>>,
    pub ptxas_log: Option<String>,
    pub ptxas_command: Option<String>,
}

/// The on-disk cache's directory, or `None` when it is off (`CUTE_RS_DISABLE_FILE_CACHING`).
pub fn cache_dir() -> Option<PathBuf> {
    if env_flag(ENV_DISABLE_FILE_CACHING) {
        return None;
    }
    if let Some(dir) = std::env::var_os(ENV_CACHE_DIR).filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let user = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_else(|_| "user".into());
    let user: String =
        user.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    Some(std::env::temp_dir().join(format!("cute-rs-{user}")))
}

/// A file's size and modification time (a tool's, in the key: a toolkit or LLVM replaced in
/// place keys differently), or `?` if it cannot be read.
pub fn file_identity(path: &Path) -> String {
    match std::fs::metadata(path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            format!("{}b@{mtime}", m.len())
        }
        Err(_) => "?".into(),
    }
}

/// What identifies the running program's build: its path, size and modification time.
pub fn build_identity() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let exe = std::env::current_exe().ok();
        let meta = exe.as_ref().and_then(|e| std::fs::metadata(e).ok());
        let mtime = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        format!("{:?} {} {mtime}", exe, meta.map_or(0, |m| m.len()))
    })
}

/// Make sure `dir` is a cache directory only this user can change, creating it (and its
/// missing parents) private to this user if it does not exist: on Unix it must then be a
/// real directory (not a symbolic link) owned by this user that no one else can write, or
/// it is refused. Anyone who could write entries could have cute-rs load their code onto the
/// GPU: the checksum catches damage, not an entry written on purpose.
pub fn ensure_private_dir(dir: &Path) -> Result<(), String> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    if std::fs::symlink_metadata(dir).is_err() {
        builder.create(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let meta = std::fs::symlink_metadata(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    if !meta.is_dir() {
        return Err(format!("{} is not a directory (or is a symbolic link)", dir.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        unsafe extern "C" {
            safe fn geteuid() -> u32;
        }
        let me = geteuid();
        if meta.uid() != me {
            return Err(format!("{} belongs to user id {}, not this user ({me})", dir.display(), meta.uid()));
        }
        if meta.mode() & 0o022 != 0 {
            return Err(format!(
                "{} can be written by other users (mode {:o}): make it private (chmod 700)",
                dir.display(),
                meta.mode() & 0o777
            ));
        }
    }
    Ok(())
}

/// [`ensure_private_dir`], logging a refusal (which turns the disk cache off for the call).
fn usable(dir: &Path) -> bool {
    match ensure_private_dir(dir) {
        Ok(()) => true,
        Err(why) => {
            crate::cute_log!(Level::Warn, "JIT cache directory refused, not caching on disk: {why}");
            false
        }
    }
}

fn entry_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{}.{EXTENSION}", &hex(&sha256(key.as_bytes()))[..32]))
}

fn push_field(out: &mut Vec<u8>, field: Option<&[u8]>) {
    match field {
        Some(b) => {
            out.extend_from_slice(&(b.len() as u64).to_le_bytes());
            out.extend_from_slice(b);
        }
        None => out.extend_from_slice(&u64::MAX.to_le_bytes()),
    }
}

/// An entry's bytes: the magic, the key, the PTX, the cubin, the `ptxas` log and command,
/// then the SHA-256 of everything before.
pub(crate) fn encode(key: &str, e: &DiskEntry) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    push_field(&mut out, Some(key.as_bytes()));
    push_field(&mut out, Some(e.ptx.as_bytes()));
    push_field(&mut out, e.cubin.as_deref());
    push_field(&mut out, e.ptxas_log.as_ref().map(|s| s.as_bytes()));
    push_field(&mut out, e.ptxas_command.as_ref().map(|s| s.as_bytes()));
    let sum = sha256(&out);
    out.extend_from_slice(&sum);
    out
}

/// Parse an entry: `Err` when it is corrupt or truncated, `Ok(None)` when it holds another
/// key (a collision of names).
pub(crate) fn decode(key: &str, bytes: &[u8]) -> Result<Option<DiskEntry>, String> {
    if bytes.len() < MAGIC.len() + 32 || !bytes.starts_with(MAGIC) {
        return Err("truncated, or not a cute-rs cache entry".into());
    }
    let (body, sum) = bytes.split_at(bytes.len() - 32);
    if sha256(body) != sum {
        return Err("the checksum does not match".into());
    }
    let mut at = MAGIC.len();
    let mut field = || -> Result<Option<Vec<u8>>, String> {
        let len = body.get(at..at + 8).ok_or("truncated field length")?;
        let len = u64::from_le_bytes(len.try_into().unwrap());
        at += 8;
        if len == u64::MAX {
            return Ok(None);
        }
        let end = at.checked_add(len as usize).filter(|&e| e <= body.len()).ok_or("truncated field")?;
        let v = body[at..end].to_vec();
        at = end;
        Ok(Some(v))
    };
    let text = |v: Option<Vec<u8>>| v.map(|b| String::from_utf8_lossy(&b).into_owned());
    let stored = text(field()?).ok_or("no key")?;
    let ptx = text(field()?).ok_or("no PTX")?;
    let cubin = field()?;
    let ptxas_log = text(field()?);
    let ptxas_command = text(field()?);
    if stored != key {
        return Ok(None);
    }
    Ok(Some(DiskEntry { ptx, cubin, ptxas_log, ptxas_command }))
}

/// Read the entry for `key` from `dir`. A corrupt or truncated one is removed and reported
/// as a miss.
pub fn disk_get(dir: &Path, key: &str) -> Option<DiskEntry> {
    if !usable(dir) {
        return None;
    }
    let path = entry_path(dir, key);
    let bytes = std::fs::read(&path).ok()?;
    match decode(key, &bytes) {
        Ok(Some(e)) => {
            // Recently used: evicted last.
            if let Ok(f) = std::fs::File::options().append(true).open(&path) {
                let _ = f.set_modified(std::time::SystemTime::now());
            }
            Some(e)
        }
        Ok(None) => None,
        Err(why) => {
            CORRUPT.fetch_add(1, Ordering::Relaxed);
            crate::cute_log!(Level::Warn, "JIT cache entry {} is unusable ({why}): recompiling", path.display());
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

/// Write the entry for `key` into `dir` (through a temporary file renamed into place), then
/// bound the directory.
pub fn disk_put(dir: &Path, key: &str, entry: &DiskEntry) {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if !usable(dir) {
        return;
    }
    let path = entry_path(dir, key);
    let tmp = dir.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("entry"),
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let written = std::fs::write(&tmp, encode(key, entry)).and_then(|_| std::fs::rename(&tmp, &path));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        crate::cute_log!(Level::Warn, "cannot write the JIT cache entry {}: {e}", path.display());
        return;
    }
    evict(dir);
}

/// Remove the least recently used entries until the directory is within its bounds.
pub fn evict(dir: &Path) {
    let max_entries = env_number(ENV_FILE_CACHE_MAX_ENTRIES, 256) as usize;
    let max_bytes = env_number(ENV_FILE_CACHE_MAX_BYTES, 64 << 20);
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<(std::time::SystemTime, u64, PathBuf)> = read
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == EXTENSION))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    entries.sort();
    let mut total: u64 = entries.iter().map(|e| e.1).sum();
    let mut count = entries.len();
    for (_, len, path) in entries {
        if count <= max_entries && total <= max_bytes {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            crate::cute_log!(Level::Info, "JIT cache: evicted {}", path.display());
        }
        count -= 1;
        total -= len;
    }
}

// ---------------------------------------------------------------------------------------
// KernelCache.

/// A kernel built and compiled for one key of a [`KernelCache`].
#[derive(Debug)]
pub struct CachedKernel {
    pub kernel: Kernel,
    pub compiled: CompiledKernel,
}

/// A memo of built and compiled kernels on keys of the user's (CuTe DSL's caching of
/// compiled functions per signature): build the key from [`TensorSpec::cache_key`](crate::TensorSpec::cache_key),
/// [`PointerSpec::cache_key`](crate::PointerSpec::cache_key), [`SymInt::cache_key`](crate::SymInt::cache_key),
/// [`Arch`](crate::Arch) and [`CompileOptions::cache_key`](crate::CompileOptions::cache_key),
/// and a second call with an equal key returns the first one's kernel without building or
/// compiling again. The key must hold everything the build depends on: for several tensors
/// whose specs may share a [`SymInt`](crate::SymInt), use
/// [`TensorSpec::cache_keys`](crate::TensorSpec::cache_keys), which numbers them together.
///
/// ```no_run
/// # use cute::prelude::*;
/// # fn demo(tc: &Toolchain, spec: TensorSpec) -> Result<(), CompileError> {
/// let cache: KernelCache<(TensorSpecKey, Arch)> = KernelCache::new();
/// let hit = cache.get_or_compile((spec.cache_key(), Arch::Sm89), || {
///     let k = Kernel::builder("scale").block_dim([32, 1, 1]).build(|| {
///         let _x = param_tensor::<f32>("x", &spec);
///     });
///     let c = k.compile_with(tc, &CompileOptions::new().gpu_arch(Arch::Sm89))?;
///     Ok((k, c))
/// })?;
/// # let _ = hit; Ok(()) }
/// ```
pub struct KernelCache<K> {
    map: Mutex<HashMap<K, Arc<CachedKernel>>>,
}

impl<K: Hash + Eq> Default for KernelCache<K> {
    fn default() -> Self {
        KernelCache { map: Mutex::new(HashMap::new()) }
    }
}

impl<K: Hash + Eq> KernelCache<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The kernel for `key`, built and compiled by `build` the first time.
    pub fn get_or_compile(
        &self,
        key: K,
        build: impl FnOnce() -> Result<(Kernel, CompiledKernel), CompileError>,
    ) -> Result<Arc<CachedKernel>, CompileError> {
        if let Some(hit) = self.map.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(hit.clone());
        }
        let (kernel, compiled) = build()?;
        let entry = Arc::new(CachedKernel { kernel, compiled });
        Ok(self.map.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_insert(entry).clone())
    }

    /// The kernel for `key`, if it was compiled.
    pub fn get(&self, key: &K) -> Option<Arc<CachedKernel>> {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned()
    }

    pub fn len(&self) -> usize {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-2's test vectors.
    #[test]
    fn sha256_test_vectors() {
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(
            hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let million = vec![b'a'; 1_000_000];
        assert_eq!(hex(&sha256(&million)), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn entries_round_trip_and_detect_damage() {
        let e = DiskEntry {
            ptx: ".version 8.0".into(),
            cubin: Some(vec![1, 2, 3]),
            ptxas_log: None,
            ptxas_command: Some("ptxas".into()),
        };
        let bytes = encode("k", &e);
        assert_eq!(decode("k", &bytes), Ok(Some(e.clone())));
        assert_eq!(decode("other", &bytes), Ok(None), "a different key is never returned");
        assert!(decode("k", &bytes[..bytes.len() - 1]).is_err(), "truncated");
        let mut bad = bytes.clone();
        bad[MAGIC.len() + 10] ^= 1;
        assert!(decode("k", &bad).unwrap_err().contains("checksum"));
        assert!(decode("k", b"short").is_err());
    }

    /// The cache directory is created private, and one others could write (or a symbolic
    /// link) is refused, so no other user can plant entries.
    #[cfg(unix)]
    #[test]
    fn the_cache_directory_must_be_private() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("cute-rs-cache-dir-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("a").join("cache");
        ensure_private_dir(&dir).unwrap();
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(ensure_private_dir(&dir).unwrap_err().contains("other users"));
        let e = DiskEntry { ptx: "x".into(), cubin: None, ptxas_log: None, ptxas_command: None };
        disk_put(&dir, "k", &e);
        assert!(disk_get(&dir, "k").is_none(), "nothing is written to or read from a shared directory");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink(&dir, &link).unwrap();
        assert!(ensure_private_dir(&link).unwrap_err().contains("symbolic link"));
        disk_put(&dir, "k", &e);
        assert_eq!(disk_get(&dir, "k"), Some(e));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn lru_evicts_the_least_recently_used() {
        let mut l: Lru<u32, u32> = Lru::new();
        l.put(1, 10, 2);
        l.put(2, 20, 2);
        assert_eq!(l.get(&1), Some(10));
        assert_eq!(l.put(3, 30, 2), vec![20]);
        assert_eq!((l.get(&1), l.get(&2), l.get(&3)), (Some(10), None, Some(30)));
    }
}

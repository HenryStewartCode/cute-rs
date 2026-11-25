//! The compile caches: in memory (hits skip `mlir-opt`; options, targets and toolchains
//! miss; `no_cache` and `CUTE_RS_NO_CACHE` bypass; moving code keeps the key unless line
//! information is on), on disk (a fresh process hits; corrupt and truncated entries are
//! recompiled; concurrent writers never produce a bad read; the size bound evicts) and
//! `KernelCache`. The tests change the environment, so they run one at a time, each with
//! its own cache directory, which it deletes. Skipped (with a note) without an `mlir-opt`.

use std::path::PathBuf;
use std::sync::Mutex;

use cute::backend::cache::{self, DiskEntry};
use cute::prelude::*;

static SERIAL: Mutex<()> = Mutex::new(());

/// A cache directory of this test's own, deleted on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("cute-rs-jit-cache-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn entries(&self) -> Vec<PathBuf> {
        std::fs::read_dir(&self.0)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "cute-cache"))
            .collect()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn env(name: &str, value: Option<&str>) {
    // SAFETY: the tests of this file run one at a time (SERIAL) and spawn no threads that read
    // the environment while it changes.
    unsafe {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
}

fn reset_env() {
    for v in [
        cache::ENV_NO_CACHE,
        cache::ENV_CACHE_DIR,
        cache::ENV_DISABLE_FILE_CACHING,
        cache::ENV_FILE_CACHE_MAX_ENTRIES,
        cache::ENV_JIT_CACHE_MAX_ELEMS,
    ] {
        env(v, None);
    }
    env(cache::ENV_DISABLE_FILE_CACHING, Some("1"));
}

fn toolchain() -> Option<Toolchain> {
    let t = Toolchain::discover();
    if t.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    t
}

fn kernel(name: &str) -> Kernel {
    Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let i = thread_idx().x;
        x.store(i, x.load(i) * 3.0f32 + 1.0f32);
    })
}

fn opts() -> CompileOptions {
    CompileOptions::new().gpu_arch(Arch::Sm89).use_ptxas(false)
}

/// Compile and return how many `mlir-opt` runs it took.
fn runs(tc: &Toolchain, k: &Kernel, o: &CompileOptions) -> (u64, String) {
    let before = cache_stats().mlir_opt_runs;
    let c = k.compile_with(tc, o).unwrap_or_else(|e| panic!("{e}"));
    (cache_stats().mlir_opt_runs - before, c.ptx().to_string())
}

#[test]
fn memory_cache_hits_and_misses() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    reset_env();
    let Some(tc) = toolchain() else { return };
    cache::clear_memory_cache();
    let k = kernel("jit_cache_memory");
    let (first, ptx) = runs(&tc, &k, &opts());
    assert!(first > 0);
    let hits = cache_stats().memory_hits;
    let (second, again) = runs(&tc, &k, &opts());
    assert_eq!((second, cache_stats().memory_hits), (0, hits + 1), "the second compile hits");
    assert_eq!(ptx, again);
    // A different option, target or toolchain misses.
    assert!(runs(&tc, &k, &opts().opt_level(1)).0 > 0, "another opt level");
    assert!(runs(&tc, &k, &opts().gpu_arch(Arch::Sm80)).0 > 0, "another target");
    let dir = TempDir::new("toolchain");
    let link = dir.0.join("mlir-opt");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&tc.mlir_opt, &link).unwrap();
    #[cfg(unix)]
    assert!(runs(&Toolchain::new(link), &k, &opts()).0 > 0, "another mlir-opt");
    // no_cache and CUTE_RS_NO_CACHE bypass.
    assert!(runs(&tc, &k, &opts().no_cache(true)).0 > 0);
    env(cache::ENV_NO_CACHE, Some("1"));
    assert!(runs(&tc, &k, &opts()).0 > 0);
    env(cache::ENV_NO_CACHE, None);
    assert_eq!(runs(&tc, &k, &opts()).0, 0);
    // Keeping an artifact bypasses the cache.
    let keep = TempDir::new("keep");
    assert!(runs(&tc, &k, &opts().keep(Artifact::Ptx, true).dump_dir(&keep.0)).0 > 0);
    // The bound evicts the least recently used.
    env(cache::ENV_JIT_CACHE_MAX_ELEMS, Some("1"));
    let other = kernel("jit_cache_other");
    runs(&tc, &other, &opts());
    assert_eq!(cache::memory_cache_len(), 1);
    assert!(runs(&tc, &k, &opts()).0 > 0, "evicted");
    reset_env();
}

/// The same body built at two places: without line information the MLIR (and the key) is
/// the same; with it, the locations differ and the second compile misses.
#[test]
fn moving_code_keeps_the_key_unless_line_info_is_on() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    reset_env();
    let Some(tc) = toolchain() else { return };
    cache::clear_memory_cache();
    let a = Kernel::builder("jit_cache_moved").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        x.store(thread_idx().x, 2.0f32);
    });

    let b = Kernel::builder("jit_cache_moved").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        x.store(thread_idx().x, 2.0f32);
    });
    assert!(runs(&tc, &a, &opts()).0 > 0);
    assert_eq!(runs(&tc, &b, &opts()).0, 0, "only the locations moved");
    let lines = opts().generate_line_info(true);
    assert!(runs(&tc, &a, &lines).0 > 0);
    assert!(runs(&tc, &b, &lines).0 > 0, "with line information the locations are part of the key");
}

/// The child half of `a_fresh_process_hits_the_disk_cache`: does nothing unless run by it.
#[test]
fn disk_cache_child() {
    if std::env::var("CUTE_RS_JIT_CACHE_CHILD").is_err() {
        return;
    }
    let Some(tc) = toolchain() else { return };
    let (n, _) = runs(&tc, &kernel("jit_cache_disk"), &opts());
    assert_eq!((n, cache_stats().disk_hits), (0, 1), "the fresh process hits the disk cache");
}

#[test]
fn a_fresh_process_hits_the_disk_cache() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    reset_env();
    let Some(tc) = toolchain() else { return };
    let dir = TempDir::new("disk");
    env(cache::ENV_DISABLE_FILE_CACHING, None);
    env(cache::ENV_CACHE_DIR, Some(dir.0.to_str().unwrap()));
    cache::clear_memory_cache();
    let k = kernel("jit_cache_disk");
    assert!(runs(&tc, &k, &opts()).0 > 0);
    assert_eq!(dir.entries().len(), 1);
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "disk_cache_child", "--nocapture"])
        .env("CUTE_RS_JIT_CACHE_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stdout).contains("1 passed"),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // In this process a memory miss now reads the disk.
    cache::clear_memory_cache();
    let disk = cache_stats().disk_hits;
    assert_eq!(runs(&tc, &k, &opts()).0, 0);
    assert_eq!(cache_stats().disk_hits, disk + 1);
    // Corrupt and truncated entries are detected, removed and recompiled.
    let entry = dir.entries()[0].clone();
    for damage in [0, 1] {
        let mut bytes = std::fs::read(&entry).unwrap();
        if damage == 0 {
            let mid = bytes.len() / 2;
            bytes[mid] ^= 0x40;
        } else {
            bytes.truncate(bytes.len() - 7);
        }
        std::fs::write(&entry, bytes).unwrap();
        cache::clear_memory_cache();
        let corrupt = cache_stats().corrupt;
        assert!(runs(&tc, &k, &opts()).0 > 0, "damage {damage}: recompiled");
        assert_eq!(cache_stats().corrupt, corrupt + 1);
        cache::clear_memory_cache();
        assert_eq!(runs(&tc, &k, &opts()).0, 0, "the rewritten entry is good");
    }
    reset_env();
}

/// Writers replacing one entry while readers read it: every read is a whole entry of one
/// writer or nothing.
#[test]
fn concurrent_writers_never_produce_a_bad_read() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = TempDir::new("concurrent");
    let entry = |w: usize| DiskEntry {
        ptx: format!("w{w};").repeat(20000),
        cubin: Some(vec![w as u8; 50000]),
        ptxas_log: None,
        ptxas_command: None,
    };
    std::thread::scope(|s| {
        for w in 0..4 {
            let (dir, e) = (&dir.0, entry(w));
            s.spawn(move || (0..20).for_each(|_| cache::disk_put(dir, "shared", &e)));
        }
        for _ in 0..4 {
            let dir = &dir.0;
            s.spawn(move || {
                for _ in 0..200 {
                    if let Some(e) = cache::disk_get(dir, "shared") {
                        let w = e.cubin.as_ref().unwrap()[0] as usize;
                        assert_eq!(e, entry(w), "a whole entry of writer {w}");
                    }
                }
            });
        }
    });
    assert_eq!(dir.entries().len(), 1);
    let leftovers: Vec<_> = std::fs::read_dir(&dir.0).unwrap().filter_map(Result::ok).collect();
    assert_eq!(leftovers.len(), 1, "no temporary files left");
}

#[test]
fn the_size_bound_evicts_the_oldest() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = TempDir::new("bound");
    env(cache::ENV_FILE_CACHE_MAX_ENTRIES, Some("3"));
    let e = DiskEntry { ptx: "x".into(), cubin: None, ptxas_log: None, ptxas_command: None };
    for i in 0..5 {
        cache::disk_put(&dir.0, &format!("key{i}"), &e);
        // Some file systems (the test pod's network volume) keep whole-second times.
        std::thread::sleep(std::time::Duration::from_millis(1100));
    }
    assert_eq!(dir.entries().len(), 3);
    assert!(cache::disk_get(&dir.0, "key0").is_none() && cache::disk_get(&dir.0, "key1").is_none());
    assert!(cache::disk_get(&dir.0, "key4").is_some());
    env(cache::ENV_FILE_CACHE_MAX_ENTRIES, None);
}

#[test]
fn kernel_cache_memoizes_on_user_keys() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    reset_env();
    let Some(tc) = toolchain() else { return };
    let cache: KernelCache<(TensorSpecKey, Arch, String)> = KernelCache::new();
    let spec = make_fake_compact_tensor::<f32>((sym_int32(1, None),), None, false);
    let mut builds = 0;
    for _ in 0..3 {
        let key = (spec.cache_key(), Arch::Sm89, opts().cache_key());
        let hit = cache
            .get_or_compile(key, || {
                builds += 1;
                let k = Kernel::builder("kernel_cache").block_dim([32, 1, 1]).build(|| {
                    let x = param_tensor::<f32>("x", &spec);
                    x.store(thread_idx().x, 1.0f32);
                });
                let c = k.compile_with(&tc, &opts())?;
                Ok((k, c))
            })
            .unwrap();
        assert!(hit.compiled.ptx().contains("kernel_cache"));
    }
    assert_eq!((builds, cache.len()), (1, 1));
    assert_ne!(opts().cache_key(), opts().opt_level(1).cache_key());
}

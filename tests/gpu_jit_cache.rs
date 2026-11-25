//! The per-context module cache and the library unload callbacks on a real GPU. Needs the
//! `cuda` feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_jit_cache
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cute::cuda::{CudaContext, register_library_unload_callback, unregister_library_unload_callback};
use cute::prelude::*;

/// The tests change `CUTE_RS_JIT_CACHE_MAX_ELEMS`: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn kernel(name: &str, k: f32) -> Kernel {
    Kernel::builder(name).block_dim([64, 1, 1]).build(move || {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let i = thread_idx().x;
        x.store(i, x.load(i) + k);
    })
}

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [64, 1, 1])
}

/// `run_cuda` twice compiles once (the in-memory cache) and loads the module once (the
/// context's module cache).
#[test]
fn run_cuda_twice_loads_the_module_once() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = kernel("module_cache_twice", 1.5);
    let loads = ctx.modules_loaded();
    let mut args = vec![KernelArg::f32(vec![0.0; 64])];
    k.run_cuda(&ctx, &tc, cfg(), &mut args).unwrap();
    let lowerings = cache_stats().mlir_opt_runs;
    k.run_cuda(&ctx, &tc, cfg(), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<f32>(), vec![3.0; 64]);
    assert_eq!(ctx.modules_loaded() - loads, 1, "one cuModuleLoadDataEx");
    assert_eq!(cache_stats().mlir_opt_runs, lowerings, "no second lowering");
    // no_cache loads again.
    let options = CompileOptions::new().with_env().unwrap().no_cache(true);
    k.run_cuda_with(&ctx, &tc, &options, cfg(), &mut args).unwrap();
    assert_eq!(ctx.modules_loaded() - loads, 2);
}

/// Unload callbacks fire exactly once per module, when its last holder goes: on eviction from
/// the module cache and when the cache is dropped; a panicking callback is logged and the
/// others still run.
#[test]
fn unload_callbacks_fire_once_on_eviction_and_drop() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let panics = Arc::new(AtomicUsize::new(0));
    let s2 = seen.clone();
    let p2 = panics.clone();
    let bad = register_library_unload_callback(move |_| {
        p2.fetch_add(1, Ordering::SeqCst);
        panic!("a misbehaving runtime");
    });
    let good = register_library_unload_callback(move |m| s2.lock().unwrap().push(m as usize));
    let arch = ctx.best_target(&tc).unwrap();
    let compile = |k: &Kernel| k.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap();
    let (a, b) = (compile(&kernel("unload_a", 1.0)), compile(&kernel("unload_b", 2.0)));
    let count = |m: usize| seen.lock().unwrap().iter().filter(|&&x| x == m).count();

    // Dropping the function leaves the module in the cache; clearing the cache unloads it.
    let fa = ctx.load(&a).unwrap();
    let ma = fa.as_raw().1 as usize;
    drop(fa);
    assert_eq!(count(ma), 0, "still cached");
    ctx.clear_module_cache();
    assert_eq!(count(ma), 1, "unloaded once, with the cache dropped");

    // Eviction: with room for one module, loading b evicts a (no function holds it).
    // SAFETY: SERIAL keeps the other tests of this binary from reading the environment.
    unsafe { std::env::set_var("CUTE_RS_JIT_CACHE_MAX_ELEMS", "1") };
    let fa = ctx.load(&a).unwrap();
    let ma = fa.as_raw().1 as usize;
    drop(fa);
    let fb = ctx.load(&b).unwrap();
    let mb = fb.as_raw().1 as usize;
    assert_eq!(count(ma), 1, "evicted and unloaded once");
    // A function keeps an evicted module alive until it is dropped.
    let fa = ctx.load(&a).unwrap();
    let ma2 = fa.as_raw().1 as usize;
    assert_eq!(count(mb), 0, "fb still holds b");
    drop(fb);
    assert_eq!(count(mb), 1);
    drop(fa);
    ctx.clear_module_cache();
    assert_eq!(count(ma2), 1);
    // SAFETY: as above.
    unsafe { std::env::remove_var("CUTE_RS_JIT_CACHE_MAX_ELEMS") };
    assert!(panics.load(Ordering::SeqCst) >= 3, "the panicking callback ran each time and was skipped");
    assert!(unregister_library_unload_callback(bad) && unregister_library_unload_callback(good));
    assert!(!unregister_library_unload_callback(good));
    // Unregistered callbacks no longer run.
    let before = seen.lock().unwrap().len();
    let f = ctx.load(&b).unwrap();
    drop(f);
    ctx.clear_module_cache();
    assert_eq!(seen.lock().unwrap().len(), before);
}

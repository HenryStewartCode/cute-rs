//! Loaded modules, shared between the [`CudaFunction`](super::CudaFunction)s made from them and
//! each context's module cache, and the library unload callbacks (CuTe DSL's
//! `cutlass.register_library_unload_callback`).
//!
//! A module is unloaded when its last holder goes: the last function loaded from it, or its
//! entry in the context's cache (evicted, bounded by `CUTE_RS_JIT_CACHE_MAX_ELEMS`, or dropped
//! with the context). Just before, every registered unload callback is called with the raw
//! `CUmodule`, so a runtime holding references into it (NVSHMEM, say) can drop them.
//! Callbacks must be idempotent; one that panics is logged and skipped, and the others still
//! run.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::{CuContext, CuModule, CuResult};
use crate::log::Level;

type Callback = Arc<dyn Fn(*mut c_void) + Send + Sync>;

static CALLBACKS: Mutex<Vec<(u64, Callback)>> = Mutex::new(Vec::new());
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// The handle of a registered unload callback, for
/// [`unregister_library_unload_callback`] (Rust closures cannot be compared).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnloadCallbackToken(u64);

/// Call `callback` with the raw `CUmodule` just before cute-rs unloads a module it loaded
/// (CuTe DSL's `cutlass.register_library_unload_callback`, which passes the `CUlibrary`).
/// The callback must be idempotent; a panic in it is logged and ignored.
pub fn register_library_unload_callback(callback: impl Fn(*mut c_void) + Send + Sync + 'static) -> UnloadCallbackToken {
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).push((token, Arc::new(callback)));
    UnloadCallbackToken(token)
}

/// Stop calling the callback `token` registered; whether it was registered.
pub fn unregister_library_unload_callback(token: UnloadCallbackToken) -> bool {
    let mut list = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner());
    let before = list.len();
    list.retain(|(t, _)| *t != token.0);
    list.len() != before
}

/// Run every unload callback for `module`, outside the registry's lock.
fn run_callbacks(module: CuModule) {
    let callbacks: Vec<(u64, Callback)> = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for (token, cb) in callbacks {
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| cb(module))) {
            let why = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "a panic".into());
            eprintln!("cute-rs: warning: library unload callback {token} panicked ({why}); continuing");
            crate::cute_log!(Level::Warn, "library unload callback {token} panicked ({why}); continuing");
        }
    }
}

/// A loaded `CUmodule`, unloaded (after the unload callbacks) when the last holder drops it.
pub(super) struct LoadedModule {
    pub(super) module: CuModule,
    ctx: CuContext,
    set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    unload: unsafe extern "C" fn(CuModule) -> CuResult,
}

impl LoadedModule {
    pub(super) fn new(
        module: CuModule,
        ctx: CuContext,
        set_current: unsafe extern "C" fn(CuContext) -> CuResult,
        unload: unsafe extern "C" fn(CuModule) -> CuResult,
    ) -> LoadedModule {
        LoadedModule { module, ctx, set_current, unload }
    }
}

impl Drop for LoadedModule {
    fn drop(&mut self) {
        run_callbacks(self.module);
        crate::cute_log!(Level::Debug, "unloading module {:?}", self.module);
        // SAFETY: the module was loaded in `ctx`, which is still retained (a context drops
        // its module cache before releasing itself, and functions borrow the context), and is
        // unloaded once.
        unsafe {
            (self.set_current)(self.ctx);
            (self.unload)(self.module);
        }
    }
}

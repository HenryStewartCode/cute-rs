//! The kernels of the multi-kernel module tests (tests/multi_kernel.rs on any machine,
//! tests/gpu_multi_kernel.rs on a GPU): a producer that writes a `__device__` global and a
//! consumer, in the same module, that reads it.

use cute::prelude::*;

pub const N: usize = 128;
pub const BIAS: i32 = 7;

/// `mk_handoff[i] = 3 * x[N - 1 - i] + 1`, reversed through shared memory, and `mk_calls`
/// counts the launches. 128 threads (`.reqntid`).
pub fn producer() -> Kernel {
    Kernel::builder("mk_producer").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major((N,)));
        let handoff = module_global::<i32>(Some("mk_handoff"), N, AddressSpace::Gmem, None, false);
        let calls = module_global::<i32>(Some("mk_calls"), 1, AddressSpace::Gmem, None, false);
        let tile = make_smem_tensor::<i32>(Layout::col_major((N,)));
        let i = thread_idx().x;
        tile.store(i, x.load(i) * 3 + 1);
        sync_threads();
        handoff.store(i, tile.load(i * -1 + (N as i32 - 1)));
        if_(i.eq(0), || calls.store(0, calls.load(0) + 1));
    })
}

/// `out[i] = (mk_handoff[i] + bias) * mk_calls`, through shared memory of its own; at most 256
/// threads (`.maxntid`), the first N active.
pub fn consumer() -> Kernel {
    consumer_with(1)
}

/// [`consumer`] scaled by `scale` (another kernel, for the cache tests).
pub fn consumer_with(scale: i32) -> Kernel {
    Kernel::builder("mk_consumer").max_threads([256, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((N,)));
        let bias = param_scalar::<i32>("bias");
        let handoff = module_global::<i32>(Some("mk_handoff"), N, AddressSpace::Gmem, None, false);
        let calls = module_global::<i32>(Some("mk_calls"), 1, AddressSpace::Gmem, None, false);
        let tile = make_smem_tensor::<i32>(Layout::col_major((N,)));
        let i = thread_idx().x;
        if_(i.lt(N as i32), || tile.store(i, handoff.load(i) + bias));
        sync_threads();
        if_(i.lt(N as i32), || out.store(i, tile.load(i) * calls.load(0) * scale));
    })
}

/// The module of both.
pub fn module() -> KernelModule {
    KernelModule::new("mk_pair", [producer(), consumer()]).unwrap_or_else(|e| panic!("{e}"))
}

pub fn input() -> Vec<i32> {
    (0..N as i32).map(|i| i - 50).collect()
}

/// What the consumer writes after `launches` producer launches.
pub fn expected(launches: i32) -> Vec<i32> {
    let x = input();
    (0..N).map(|i| (3 * x[N - 1 - i] + 1 + BIAS) * launches).collect()
}

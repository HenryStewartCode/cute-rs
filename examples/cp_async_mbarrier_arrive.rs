// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/cp_async_mbarrier_arrive_cta.py
//   (the kernel `kernel`, the launch of `host` and the host data of `run`)
//   and examples/python/CuTeDSL/experimental/primitives/cp_async_mbarrier_arrive_cluster.py
//   (the kernel `kernel` without its peer shared-memory read, and the launch
//   of `host`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! `cp.async` completion through an mbarrier, ported from CuTe DSL's
//! `experimental/primitives/cp_async_mbarrier_arrive_cta.py` and
//! `cp_async_mbarrier_arrive_cluster.py`.
//!
//! [`cta`]: a producer warp issues one 16-byte `cp.async` (`.cg`) per lane
//! (`prims.cp_async_shared_global`) and `cp_async_mbarrier_arrive_noinc`,
//! so each lane's arrival lands once its copy has; the consumer warp waits
//! on the mbarrier (initialised with 32 arrivals) and copies the tile to
//! global memory, with no cp.async groups to commit or wait for.
//!
//! [`cluster`]: a cluster of 2 CTAs of 96 threads, a producer, a forwarder
//! and a consumer warp each. Each producer copies its CTA's tile and arrives
//! (`noinc`) on its local mbarrier; each forwarder waits for that and its
//! elected lane arrives remotely, at cluster scope, on CTA 0's leader
//! mbarrier (`mapa`, here `MbarrierOptions::peer_cta_rank_in_cluster`); CTA
//! 0's consumer waits for both forwarded completions and releases both
//! CTAs' empty mbarriers (its own locally, CTA 1's remotely at cluster
//! scope). `cluster_arrive_relaxed` and `cluster_wait` follow the inits and
//! `fence.mbarrier_init`. The example's consumer then reads CTA 1's tile
//! through distributed shared memory, which cute-rs has no loads for yet
//! (m8): instead each CTA writes its own tile to global memory once its
//! empty mbarrier completes, which only the chain of mbarriers orders after
//! the copies of its own producer. It needs SM90: the simulator,
//! `mlir-opt` for sm_90a and `ptxas` check it.
//!
//! Run with `cargo run --release --example cp_async_mbarrier_arrive`.

use cute::prelude::*;

/// Float32 elements per lane: one 16-byte copy (`_ELEMS_PER_THREAD`).
const ELEMS_PER_THREAD: usize = 4;
/// A CTA's tile (`_DEFAULT_N`, `_TILE_ELEMS_PER_CTA`).
pub const TILE: usize = 32 * ELEMS_PER_THREAD;
/// The cluster example's CTAs per cluster (`_CLUSTER_SIZE`).
pub const CLUSTER_SIZE: u32 = 2;
/// The seeds [`run`] simulates under, besides the built-in schedules.
pub const SEEDS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Whether `arch` has `mbarrier.try_wait` and `fence.mbarrier_init` (SM90).
pub fn has_try_wait(arch: Arch) -> bool {
    !matches!(arch, Arch::Sm80 | Arch::Sm86 | Arch::Sm89)
}

/// `while not mbarrier_try_wait_parity(bar, parity): pass`, or a
/// `mbarrier_test_wait` spin before SM90.
fn wait_parity(bar: &Ptr<u64>, parity: i32, sm90: bool) {
    if sm90 {
        while_carry((), |()| !mbarrier_try_wait(bar, parity), |()| ());
    } else {
        while_carry((), |()| !mbarrier_test_wait(bar, parity), |()| ());
    }
}

/// The example's 16-byte `.cg` global-to-shared transfer: one
/// `cp.async.cg` of a lane's four elements.
fn cp_async_16(dst: &Ptr<f32>, src: &Ptr<f32>) {
    let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
    let layout = Layout::col_major(ELEMS_PER_THREAD);
    copy(&atom, &make_tensor(src, layout.clone()), &make_tensor(dst, layout));
}

/// `cp_async_mbarrier_arrive_cta.py`'s kernel, for `arch`: warp 0
/// produces, warp 1 consumes.
pub fn cta(arch: Arch) -> Kernel {
    let sm90 = has_try_wait(arch);
    Kernel::builder("cp_async_mbarrier_arrive_cta").block_dim([64, 1, 1]).build(move || {
        let src = param_tensor::<f32>("src", Layout::col_major(TILE));
        let dst = param_tensor::<f32>("dst", Layout::col_major(TILE));
        let tidx = thread_idx().x;
        let lane = tidx % 32;
        let warp_idx = warp_idx();
        let smem = alloc_smem::<f32>(TILE, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        // With noinc each producer lane's arrival is one of the count.
        if_(tidx.eq(0), || mbarrier_init(&mbar, 32));
        if sm90 {
            mbarrier_init_fence();
        }
        sync_threads();
        let offset = lane * ELEMS_PER_THREAD as i32;
        if_(warp_idx.eq(0), || {
            cp_async_16(&smem.offset(offset), &src.ptr().offset(offset));
            // No commit or wait group: the arrive ties the copy to mbar.
            cp_async_mbarrier_arrive_noinc(&mbar);
        });
        if_(warp_idx.eq(1), || {
            wait_parity(&mbar, 0, sm90);
            let values = load_vector::<f32, 4>(&smem.offset(offset), LoadOptions::default());
            store_vector(&dst.ptr().offset(offset), values, StoreOptions::default());
        });
    })
}

/// `cp_async_mbarrier_arrive_cluster.py`'s kernel (SM90), each CTA writing
/// its own tile to `dst` instead of CTA 0 reading CTA 1's.
pub fn cluster() -> Kernel {
    Kernel::builder("cp_async_mbarrier_arrive_cluster").block_dim([96, 1, 1]).build(|| {
        let src = param_tensor::<f32>("src", Layout::col_major(CLUSTER_SIZE as usize * TILE));
        let dst = param_tensor::<f32>("dst", Layout::col_major(CLUSTER_SIZE as usize * TILE));
        let tidx = thread_idx().x;
        let lane = tidx % 32;
        let warp_idx = warp_idx();
        let cta_rank = block_idx_in_cluster();
        let smem = alloc_smem::<f32>(TILE, 128);
        let local_mbar = alloc_smem::<u64>(1, 8);
        let leader_mbar = alloc_smem::<u64>(1, 8);
        let empty_mbar = alloc_smem::<u64>(1, 8);
        let cluster_scope = |peer: i32| MbarrierOptions {
            peer_cta_rank_in_cluster: Some(peer.into_value()),
            scope: Some(MemScopeKind::Cluster),
            ..Default::default()
        };

        if_(tidx.eq(0), || {
            mbarrier_init(&local_mbar, 32);
            mbarrier_init(&empty_mbar, 1);
            if_(cta_rank.eq(0), || mbarrier_init(&leader_mbar, CLUSTER_SIZE as i32));
        });
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();

        let offset = lane * ELEMS_PER_THREAD as i32;
        // Producer: this CTA's tile, tracked by the local mbarrier.
        if_(warp_idx.eq(0), || {
            let cta_base = cta_rank * TILE as i32;
            cp_async_16(&smem.offset(offset), &src.ptr().offset(cta_base + offset));
            cp_async_mbarrier_arrive_noinc(&local_mbar);
        });
        // Forwarder: the local completion fans in to CTA 0's leader barrier.
        if_(warp_idx.eq(1), || {
            wait_parity(&local_mbar, 0, true);
            if_(elect_one(), || mbarrier_arrive_with(&leader_mbar, 1, cluster_scope(0)));
        });
        // Consumer: CTA 0 waits for both CTAs and releases both empty
        // barriers; every CTA then writes its tile.
        if_(warp_idx.eq(2), || {
            if_(cta_rank.eq(0), || {
                wait_parity(&leader_mbar, 0, true);
                sync_warp();
                if_(elect_one(), || {
                    mbarrier_arrive(&empty_mbar);
                    mbarrier_arrive_with(&empty_mbar, 1, cluster_scope(1));
                });
            });
            wait_parity(&empty_mbar, 0, true);
            let values = load_vector::<f32, 4>(&smem.offset(offset), LoadOptions::default());
            store_vector(&dst.ptr().offset(cta_rank * TILE as i32 + offset), values, StoreOptions::default());
        });
    })
}

/// The CTA example's launch: one block of two warps.
pub fn cta_launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [64, 1, 1])
}

/// The cluster example's launch: one cluster of two CTAs of three warps.
pub fn cluster_launch() -> LaunchConfig {
    LaunchConfig::new([CLUSTER_SIZE, 1, 1], [96, 1, 1]).with_cluster([CLUSTER_SIZE, 1, 1])
}

/// `run`'s data for `ctas` CTAs: `src` is `0, 1, 2, ...`, `dst` zeros.
pub fn args(ctas: usize) -> Vec<KernelArg> {
    let n = ctas * TILE;
    vec![KernelArg::f32((0..n).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; n])]
}

/// `verify`'s check: `dst == src`.
pub fn check(args: &[KernelArg]) {
    assert_eq!(args[1].as_f32(), args[0].as_f32());
}

/// Simulate both kernels (the CTA one for SM89 and SM90) under every
/// schedule and [`SEEDS`], and check.
pub fn run() {
    for kernel in [cta(Arch::Sm89), cta(Arch::Sm90)] {
        let mut args = args(1);
        kernel
            .simulate_schedules(cta_launch(), &mut args, &SimOptions::default(), &SEEDS)
            .unwrap_or_else(|e| panic!("{e}"));
        check(&args);
    }
    let mut args = args(CLUSTER_SIZE as usize);
    cluster()
        .simulate_schedules(cluster_launch(), &mut args, &SimOptions::default(), &SEEDS)
        .unwrap_or_else(|e| panic!("{e}"));
    check(&args);
}

#[allow(dead_code)]
fn main() {
    run();
    println!("cp_async_mbarrier_arrive_cta and _cluster: PASS");
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file follow the design of NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/experimental/primitives/mapa.py (a ring of CTAs reading and
//   writing each other's shared memory through mapa) and
//   examples/python/CuTeDSL/experimental/primitives/reduction/cluster_smem_reduce.py (a
//   cluster reduction with red on the root CTA's shared memory),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Distributed shared memory (roadmap part m8-dsmem), held to plain Rust in clusters of
//! two and four CTAs: a ring in which each CTA writes the next one's shared memory
//! through `map_dsmem_ptr` and reads its own after a cluster barrier, then reads the
//! next one's with `ld.shared::cluster` (`cute::load` with `SharedSpace::Cluster`); a
//! cluster reduction into the root CTA's shared memory with `red` and an atomic ticket
//! counter there; a hand-off of a tile with `CopyR2SOp` (relaxed, cluster) into the
//! peer, a release store of the peer's flag, an acquire spin on the own flag and
//! `CopyS2ROp` reads (weak of the own tile, acquire of the peer's); and
//! `CopyDsmemStoreOp` (`st.async`) of 32, 64 and 128 bits, and the widest the copy finds,
//! completing on the peer's mbarrier. They need sm_90 and run in the simulator.

use cute::prelude::*;

use super::Case;

/// Threads of the DSMEM kernels.
const TH: usize = 128;

fn cluster_cfg(n: usize, clusters: usize) -> LaunchConfig {
    LaunchConfig::new([(n * clusters) as u32, 1, 1], [TH as u32, 1, 1]).with_cluster([n as u32, 1, 1])
}

/// The ring, after `mapa.py`, in `clusters` clusters of `n`: CTA `r` writes its inputs
/// into CTA `r + 1`'s `buf` through a DSMEM tensor, a cluster barrier publishes them,
/// and each CTA writes its own `buf` to `out` (what CTA `r - 1` sent); then each reads
/// CTA `r + 1`'s `buf` (its own inputs) with `cute::load(ss=Cluster)` into `back`, and a
/// last cluster barrier keeps every CTA until its peer has read.
pub fn dsmem_ring(n: usize, clusters: usize) -> Case {
    let name: &'static str = Box::leak(format!("dsmem_ring_{n}x1").into_boxed_str());
    let kernel = Kernel::builder(name).block_dim([TH as u32, 1, 1]).build(move || {
        let total = n * clusters * TH;
        let x = param_tensor::<i32>("x", Layout::col_major(total));
        let out = param_tensor::<i32>("out", Layout::col_major(total));
        let back = param_tensor::<i32>("back", Layout::col_major(total));
        let buf = make_smem_tensor_aligned::<i32>(Layout::col_major(TH), 16);
        let (t, b) = (thread_idx().x, block_idx().x);
        let rank = block_idx_in_cluster();
        let peer = (rank + 1) % n as i32;
        // The peers have started.
        cluster_arrive_relaxed();
        cluster_wait();
        let remote = make_tensor(map_dsmem_ptr(&buf.ptr(), peer), Layout::col_major(TH));
        remote.store(t, x.load(b * TH as i32 + t));
        cluster_arrive();
        cluster_wait();
        out.store(b * TH as i32 + t, buf.load(t));
        let cluster = LoadOptions { ss: Some(SharedSpace::Cluster), ..Default::default() };
        back.store(b * TH as i32 + t, load(&map_dsmem_ptr(&buf.ptr().offset(t), peer), cluster));
        cluster_arrive();
        cluster_wait();
    });
    fn check4(args: &[KernelArg]) {
        check_ring(args, 4)
    }
    fn check2(args: &[KernelArg]) {
        check_ring(args, 2)
    }
    Case { name, kernel, cfg: cluster_cfg(n, clusters), small_only: false, check: if n == 4 { check4 } else { check2 } }
}

fn check_ring(args: &[KernelArg], n: usize) {
    let (x, out, back) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
    let blocks = x.len() / TH;
    for b in 0..blocks {
        let (base, r) = (b - b % n, b % n);
        let prev = base + (r + n - 1) % n;
        for t in 0..TH {
            assert_eq!(out[b * TH + t], x[prev * TH + t], "CTA {b}, thread {t}: what CTA {prev} sent");
            assert_eq!(back[b * TH + t], x[b * TH + t], "CTA {b}, thread {t}: read back from its peer");
        }
    }
}

/// A cluster reduction after `cluster_smem_reduce.py`: every CTA zeroes its `acc`, a
/// cluster barrier, then each thread adds its input into the root CTA's `acc` with
/// `red.relaxed.cluster.shared::cluster.add.s32`, and thread 0 of each CTA takes a ticket
/// from the root's counter (`atom.add`); after a cluster barrier the root writes the
/// sums, the counter and the set of tickets taken (a bit mask: which CTA got which
/// ticket depends on the schedule).
pub fn dsmem_cluster_reduce(n: usize) -> Case {
    let name: &'static str = Box::leak(format!("dsmem_cluster_reduce_{n}x1").into_boxed_str());
    let kernel = Kernel::builder(name).block_dim([TH as u32, 1, 1]).build(move || {
        let x = param_tensor::<i32>("x", Layout::col_major(n * TH));
        let out = param_tensor::<i32>("out", Layout::col_major(TH + 2));
        let acc = make_smem_tensor_aligned::<i32>(Layout::col_major(TH), 16);
        let counter = alloc_smem::<i32>(1, 4);
        let tickets = alloc_smem::<i32>(n, 4);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        acc.store(t, 0);
        if_(t.eq(0), || counter.store(0));
        // The zeros are published to the cluster before anyone adds.
        cluster_arrive();
        cluster_wait();
        let relaxed = AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Cluster);
        let root_acc = map_dsmem_ptr(&acc.ptr(), 0);
        red(&root_acc.offset(t), x.load(rank * TH as i32 + t), RedOp::Add, DType::I32, relaxed);
        if_(t.eq(0), || {
            let ticket = atomic_add_with(&map_dsmem_ptr(&counter, 0), 1, relaxed);
            // Stored in the root's ticket slot of this CTA.
            store(
                &map_dsmem_ptr(&tickets.offset(rank), 0),
                ticket,
                StoreOptions { ss: Some(SharedSpace::Cluster), ..Default::default() },
            );
        });
        cluster_arrive();
        cluster_wait();
        if_(rank.eq(0), || {
            out.store(t, acc.load(t));
            if_(t.eq(0), || out.store(TH as i32, counter.load()));
            if_(t.eq(0), || {
                let mask = (0..n).fold(Int32::from(0), |m, r| m | (Int32::from(1) << tickets.offset(r).load()));
                out.store(TH as i32 + 1, mask);
            });
        });
    });
    fn check_n(args: &[KernelArg], n: usize) {
        let (x, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..TH {
            let want = (0..n).fold(0i32, |a, r| a.wrapping_add(x[r * TH + t]));
            assert_eq!(out[t], want, "sum {t}");
        }
        assert_eq!(out[TH], n as i32, "the counter");
        assert_eq!(out[TH + 1], (1 << n) - 1, "each CTA took one ticket");
    }
    fn check2(args: &[KernelArg]) {
        check_n(args, 2)
    }
    fn check4(args: &[KernelArg]) {
        check_n(args, 4)
    }
    Case { name, kernel, cfg: cluster_cfg(n, 1), small_only: false, check: if n == 4 { check4 } else { check2 } }
}

/// Values per thread of [`dsmem_s2r_r2s`]: one 128-bit access.
const HV: usize = 4;

/// A hand-off between neighbours with `CopyR2SOp` and `CopyS2ROp`: each CTA copies its
/// `HV` values per thread into CTA `r + 1`'s `inbox` with relaxed cluster-scope 128-bit
/// stores (`st.relaxed.cluster.shared::cluster.v4`), a barrier, thread 0 sets the peer's
/// flag with a release store at cluster scope; every thread spins on its own flag with
/// acquire loads, then reads its own inbox with a weak `CopyS2ROp` (what CTA `r - 1`
/// sent) into `got`. After a cluster barrier each reads CTA `r + 1`'s inbox with an
/// acquire `CopyS2ROp` on the cluster (`num_bits_per_copy = 0`: the widest vector) into
/// `peer`; a last barrier precedes the exit.
pub fn dsmem_s2r_r2s(n: usize) -> Case {
    let name: &'static str = Box::leak(format!("dsmem_s2r_r2s_{n}x1").into_boxed_str());
    let kernel = Kernel::builder(name).block_dim([TH as u32, 1, 1]).build(move || {
        let x = param_tensor::<i32>("x", Layout::col_major(n * TH * HV));
        let got = param_tensor::<i32>("got", Layout::col_major(n * TH * HV));
        let peer_out = param_tensor::<i32>("peer", Layout::col_major(n * TH * HV));
        let inbox = make_smem_tensor_aligned::<i32>(Layout::col_major(TH * HV), 16);
        let flag = alloc_smem::<i32>(1, 4);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        let peer = (rank + 1) % n as i32;
        if_(t.eq(0), || flag.store(0));
        cluster_arrive();
        cluster_wait();
        let at = |r: Int32| (r * TH as i32 + t) * HV as i32;
        let regs = make_fragment::<i32>(HV);
        for v in 0..HV {
            regs.store(v, x.load(at(rank) + v as i32));
        }
        let cluster_relaxed = R2SAttrs {
            memory_order: MemoryOrder::Relaxed,
            memory_scope: MemoryScope::Cluster,
            shared_space: SharedSpace::Cluster,
        };
        let r2s = make_copy_atom::<i32>(CopyOp::R2S { bits: 128, attrs: cluster_relaxed });
        let mine = |p: &Ptr<i32>| make_tensor(p.offset(t * HV as i32), Layout::col_major(HV));
        copy(&r2s, &regs, &mine(&map_dsmem_ptr(&inbox.ptr(), peer)));
        sync_threads();
        if_(t.eq(0), || {
            let release = StoreOptions {
                sem: MemoryOrder::Release,
                scope: Some(MemoryScope::Cluster),
                ss: Some(SharedSpace::Cluster),
                ..Default::default()
            };
            store(&map_dsmem_ptr(&flag, peer), 1, release);
        });
        let acquire =
            LoadOptions { sem: MemoryOrder::Acquire, scope: Some(MemoryScope::Cluster), ..Default::default() };
        while_carry((), |()| load(&flag, acquire).eq(0), |()| ());
        let s2r = make_copy_atom::<i32>(CopyOp::S2R { bits: 128, attrs: S2RAttrs::default() });
        let mine_back = make_fragment::<i32>(HV);
        copy(&s2r, &mine(&inbox.ptr()), &mine_back);
        for v in 0..HV {
            got.store(at(rank) + v as i32, mine_back.load(v));
        }
        cluster_arrive();
        cluster_wait();
        let cluster_acquire = S2RAttrs {
            memory_order: MemoryOrder::Acquire,
            memory_scope: MemoryScope::Cluster,
            shared_space: SharedSpace::Cluster,
        };
        let s2r_peer = make_copy_atom::<i32>(CopyOp::S2R { bits: 0, attrs: cluster_acquire });
        let from_peer = make_fragment::<i32>(HV);
        copy(&s2r_peer, &mine(&map_dsmem_ptr(&inbox.ptr(), peer)), &from_peer);
        for v in 0..HV {
            peer_out.store(at(rank) + v as i32, from_peer.load(v));
        }
        cluster_arrive();
        cluster_wait();
    });
    fn check_n(args: &[KernelArg], n: usize) {
        let (x, got, peer) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let per = TH * HV;
        for r in 0..n {
            let prev = (r + n - 1) % n;
            assert_eq!(got[r * per..(r + 1) * per], x[prev * per..(prev + 1) * per], "CTA {r}: what CTA {prev} sent");
            assert_eq!(peer[r * per..(r + 1) * per], x[r * per..(r + 1) * per], "CTA {r}: its own, from its peer");
        }
    }
    fn check2(args: &[KernelArg]) {
        check_n(args, 2)
    }
    fn check4(args: &[KernelArg]) {
        check_n(args, 4)
    }
    Case { name, kernel, cfg: cluster_cfg(n, 1), small_only: false, check: if n == 4 { check4 } else { check2 } }
}

/// `CopyDsmemStoreOp` of `bits` bits (0: the widest the copy finds) on `T` elements, 16
/// bytes per thread, from registers into CTA `r + 1`'s `slots`, counted down the
/// peer's mbarrier (mapped with `map_dsmem_ptr` too), which expects all of them; each CTA
/// waits on its own mbarrier and writes its slots out.
fn dsmem_store_atom<T: Numeric>(name: &'static str, bits: usize, n: usize) -> Kernel {
    let e = 128 / T::DTYPE.bits();
    Kernel::builder(name).block_dim([TH as u32, 1, 1]).build(move || {
        let x = param_tensor::<T>("x", Layout::col_major(n * TH * e));
        let out = param_tensor::<T>("out", Layout::col_major(n * TH * e));
        let slots = make_smem_tensor_aligned::<T>(Layout::col_major(TH * e), 16);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        let peer = (rank + 1) % n as i32;
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();
        if_(t.eq(0), || mbarrier_arrive_and_expect_tx(&mbar, (TH * 16) as i32));
        let regs = make_fragment::<T>(e);
        let at = (rank * TH as i32 + t) * e as i32;
        for v in 0..e {
            regs.store(v, x.load(at + v as i32));
        }
        let atom = make_copy_atom::<T>(CopyOp::DsmemStore { bits });
        let dst = make_tensor(map_dsmem_ptr(&slots.ptr().offset(t * e as i32), peer), Layout::col_major(e));
        copy_with(&atom, &regs, &dst, CopyArgs::new().mbar_ptr(&map_dsmem_ptr(&mbar, peer)));
        mbarrier_wait(&mbar, 0);
        for v in 0..e {
            out.store(at + v as i32, slots.load(t * e as i32 + v as i32));
        }
        cluster_arrive();
        cluster_wait();
    })
}

fn check_store_atom<T: Numeric + PartialEq + std::fmt::Debug>(args: &[KernelArg], n: usize) {
    let (x, out) = (args[0].to_vec::<T>(), args[1].to_vec::<T>());
    let per = x.len() / n;
    for r in 0..n {
        let prev = (r + n - 1) % n;
        assert!(out[r * per..(r + 1) * per] == x[prev * per..(prev + 1) * per], "CTA {r}: what CTA {prev} stored");
    }
}

/// The `CopyDsmemStoreOp` cases: 32 bits of Float32, 64 of Float16, 128 of Uint8 and
/// the widest (128) of Int32, in clusters of 2 and 4.
pub fn dsmem_store_atom_cases() -> Vec<Case> {
    fn f32_2(a: &[KernelArg]) {
        check_store_atom::<f32>(a, 2)
    }
    fn f16_4(a: &[KernelArg]) {
        check_store_atom::<F16>(a, 4)
    }
    fn u8_2(a: &[KernelArg]) {
        check_store_atom::<u8>(a, 2)
    }
    fn i32_4(a: &[KernelArg]) {
        check_store_atom::<i32>(a, 4)
    }
    vec![
        Case {
            name: "dsmem_store_f32_b32_2x1",
            kernel: dsmem_store_atom::<f32>("dsmem_store_f32_b32_2x1", 32, 2),
            cfg: cluster_cfg(2, 1),
            small_only: false,
            check: f32_2,
        },
        Case {
            name: "dsmem_store_f16_b64_4x1",
            kernel: dsmem_store_atom::<F16>("dsmem_store_f16_b64_4x1", 64, 4),
            cfg: cluster_cfg(4, 1),
            small_only: false,
            check: f16_4,
        },
        Case {
            name: "dsmem_store_u8_b128_2x1",
            kernel: dsmem_store_atom::<u8>("dsmem_store_u8_b128_2x1", 128, 2),
            cfg: cluster_cfg(2, 1),
            small_only: false,
            check: u8_2,
        },
        Case {
            name: "dsmem_store_i32_auto_4x1",
            kernel: dsmem_store_atom::<i32>("dsmem_store_i32_auto_4x1", 0, 4),
            cfg: cluster_cfg(4, 1),
            small_only: false,
            check: i32_4,
        },
    ]
}

/// Every DSMEM case, which needs sm_90.
pub fn dsmem_cases_sm90() -> Vec<Case> {
    let mut cases = vec![
        dsmem_ring(2, 2),
        dsmem_ring(4, 1),
        dsmem_cluster_reduce(2),
        dsmem_cluster_reduce(4),
        dsmem_s2r_r2s(2),
        dsmem_s2r_r2s(4),
    ];
    cases.extend(dsmem_store_atom_cases());
    cases
}

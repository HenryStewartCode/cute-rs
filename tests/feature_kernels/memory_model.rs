// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! Dynamic shared-memory regions, cluster ranks, memory-ordered hand-offs, cache hints, prefetches and copy atoms with memory attributes.

use super::*;
/// Kernels the simulator cannot run (their inline PTX or hand-written MLIR
/// follows pointers, their inline PTX reads the carry flag across kinds, or
/// they read hardware state through hand-written MLIR): compiled, and held
/// to plain Rust on the GPU.
pub fn gpu_only() -> Vec<Case> {
    vec![inline_ptx_pointers(), mlir_nsmid(), mixed_carry_flag(), mlir_pointers()]
}

/// The bytes [`dynamic_region`] gives its dynamic shared-memory region: not
/// a multiple of 128, so the shared-memory size registers round.
pub const DYNAMIC_REGION_BYTES: u32 = 4108;

/// Where [`dynamic_region_kernel`]'s dynamic region starts: after its
/// static buffers' 100 + 128 bytes, at a multiple of 16.
pub const DYNAMIC_REGION_START: u32 = 240;

/// The plain-Rust result of [`dynamic_region_kernel`] with a region of
/// `region_bytes`: `dynamic_smem_size` and `get_dyn_smem_size` read the
/// region's bytes, and `total_smem_size` the block's shared memory (the
/// static buffers live in the dynamic window too, in front of the region)
/// rounded up to 128 bytes, as an SM89 GPU reports it.
pub fn check_dynamic_region(args: &[KernelArg], region_bytes: u32) {
    let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
    let window = DYNAMIC_REGION_START + region_bytes;
    for t in 0..T {
        let o = (t + 32) % 64;
        let want = [
            xs[o].wrapping_mul(3),
            xs[o].wrapping_add(7),
            xs[o % 25] ^ 0x5a5a_5a5a,
            xs[o] & 0xffff,
            xs[o].wrapping_mul(3) & 0xff,
            region_bytes,
            region_bytes,
            window.next_multiple_of(128),
        ];
        assert_eq!(out[t * 8..t * 8 + 8], want, "thread {t} (a region of {region_bytes} bytes)");
    }
}

/// [`dynamic_region_kernel`] with a region of [`DYNAMIC_REGION_BYTES`].
pub fn dynamic_region() -> Case {
    let cfg = launch(1).with_dynamic_smem(DYNAMIC_REGION_BYTES);
    let check = |args: &[KernelArg]| check_dynamic_region(args, DYNAMIC_REGION_BYTES);
    Case { name: "dynamic_region", kernel: dynamic_region_kernel(), cfg, small_only: false, check }
}

/// Blocks [`cluster_ranks`] launches at most.
pub(super) const CLUSTER_BLOCKS: usize = 8;

/// The plain-Rust result of [`cluster_ranks_kernel`] on a `grid` of blocks in clusters of
/// `cluster`: 128 bytes of shared memory, plus the 1 KB the system reserves for
/// `%aggr_smem_size` (from the PTX ISA and the CUDA guide; not checked on an SM90 GPU).
pub(super) fn check_cluster_ranks(args: &[KernelArg], grid: [usize; 3], cluster: [usize; 3]) {
    let out = args[0].to_vec::<i32>();
    for (z, y, x) in (0..grid[2]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[0]).map(move |x| (z, y, x)))) {
        let block = x + y * grid[0] + z * grid[0] * grid[1];
        let rank = (x % cluster[0] + (y % cluster[1]) * cluster[0] + (z % cluster[2]) * cluster[0] * cluster[1]) as i32;
        let size = (cluster[0] * cluster[1] * cluster[2]) as i32;
        for t in 0..32 {
            let row = block * 32 + t;
            let want = [rank, size, rank, size, rank * 100 + (t as i32 + 1) % 32, 128, 128 + 1024];
            assert_eq!(out[row * 7..row * 7 + 7], want, "block ({x},{y},{z}) thread {t}");
        }
    }
}

/// [`cluster_ranks_kernel`] on 4 blocks in clusters of 2.
pub fn cluster_ranks() -> Case {
    let cfg = LaunchConfig::new([4, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    let check = |args: &[KernelArg]| check_cluster_ranks(args, [4, 1, 1], [2, 1, 1]);
    Case { name: "cluster_ranks", kernel: cluster_ranks_kernel(), cfg, small_only: false, check }
}

/// [`cluster_ranks_kernel`] on 2 x 4 blocks in clusters of 2 x 2.
pub fn cluster_ranks_2x2() -> Case {
    let cfg = LaunchConfig::new([2, 4, 1], [32, 1, 1]).with_cluster([2, 2, 1]);
    let check = |args: &[KernelArg]| check_cluster_ranks(args, [2, 4, 1], [2, 2, 1]);
    Case { name: "cluster_ranks_2x2", kernel: cluster_ranks_kernel(), cfg, small_only: false, check }
}

// ---------------------------------------------------------------------------
// Memory-ordered and cache-hinted loads and stores, prefetch, the G2R and R2G
// copy atoms with memory attributes, make_cotiled_copy.
// ---------------------------------------------------------------------------

/// What lane `l` of exchange `k` of [`ordered_handoffs`] hands over, from
/// the input `x` of its thread.
pub fn handoff_value(k: u32, x: u32) -> u32 {
    x.wrapping_mul(2 * k + 3).wrapping_add(k << 20) ^ (k * 0x1111_1111)
}

/// Seven hand-offs of shared-memory data between the four warps of a block,
/// each through a flag that the producer sets after writing its data and the
/// consumer spins on before reading it, in every ordered form: a release
/// store and an acquire load on shared memory (CTA scope) and on global
/// memory (GPU scope), `fence; st.relaxed` and `ld.relaxed; fence`, seq_cst
/// accesses, volatile ones between fences, a vector release and acquire, and
/// a release and acquire with L1 eviction priorities (inline PTX). Each
/// producer comes first in program order; the simulator's race checker
/// must accept every access.
pub fn ordered_handoffs() -> Case {
    const N: usize = 128;
    let kernel = Kernel::builder("ordered_handoffs").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(2 * N));
        let gflags = param_tensor::<u32>("gflags", Layout::col_major(2 * 8));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * N, 2)));
        let t = thread_idx().x;
        let b = block_idx().x;
        let row = b * N as i32 + t;
        let x = xs.load(row);
        let warp = warp_idx();
        let lane = t % 32;
        let data = make_smem_tensor::<u32>(Layout::col_major(7 * 32));
        let sflags = alloc_smem::<u32>(4, 16);
        let sflag = |k: usize| sflags.offset(k);
        let gflag = |k: usize| gflags.ptr().offset(IndexExpr::from(b) * 8 + k);
        let (cta, gpu) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu));
        let ld = |sem, scope| LoadOptions { sem, scope, ..Default::default() };
        let st = |sem, scope| StoreOptions { sem, scope, ..Default::default() };
        let spin = |flag: &Ptr<u32>, opts: LoadOptions| {
            while_carry((), |()| load(flag, opts).eq(0u32), |()| ());
        };
        let value = |k: u32| (x * (2 * k + 3) + (k << 20)) ^ (k * 0x1111_1111);
        // Every flag starts at zero.
        if_(t.lt(4), || sflags.offset(t).store(0u32));
        if_(t.lt(8), || gflags.ptr().offset(b * 8 + t).store(0u32));
        sync_threads();
        // A release and an acquire on shared memory, warp 0 to warp 1.
        if_(warp.eq(0), || {
            data.store(lane, value(0));
            store(&sflag(0), 1u32, st(MemoryOrder::Release, cta));
        });
        if_(warp.eq(1), || {
            spin(&sflag(0), ld(MemoryOrder::Acquire, cta));
            out.store((row, 0), data.load(lane));
        });
        // A release and an acquire on global memory, warp 2 to warp 3.
        if_(warp.eq(2), || {
            data.store(lane + 32, value(1));
            store(&gflag(0), 1u32, st(MemoryOrder::Release, gpu));
        });
        if_(warp.eq(3), || {
            spin(&gflag(0), ld(MemoryOrder::Acquire, gpu));
            out.store((row, 0), data.load(lane + 32));
        });
        // A fence and a relaxed store; a relaxed load and a fence. Warp 1 to 2.
        if_(warp.eq(1), || {
            data.store(lane + 64, value(2));
            fence_acq_rel_cta();
            store(&sflag(1), 1u32, st(MemoryOrder::Relaxed, cta));
        });
        if_(warp.eq(2), || {
            spin(&sflag(1), ld(MemoryOrder::Relaxed, cta));
            fence_acq_rel_cta();
            out.store((row, 0), data.load(lane + 64));
        });
        // seq_cst accesses on global memory, warp 3 to warp 0.
        if_(warp.eq(3), || {
            data.store(lane + 96, value(3));
            store(&gflag(1), 1u32, st(MemoryOrder::Sc, gpu));
        });
        if_(warp.eq(0), || {
            spin(&gflag(1), ld(MemoryOrder::Sc, gpu));
            out.store((row, 0), data.load(lane + 96));
        });
        // Volatile accesses between fences on shared memory, warp 0 to 2.
        if_(warp.eq(0), || {
            data.store(lane + 128, value(4));
            fence_acq_rel_cta();
            store(&sflag(2), 1u32, st(MemoryOrder::Volatile, None));
        });
        if_(warp.eq(2), || {
            spin(&sflag(2), ld(MemoryOrder::Volatile, None));
            fence_acq_rel_cta();
            out.store((row, 1), data.load(lane + 128));
        });
        // A vector release and acquire on global memory, warp 2 to warp 1.
        if_(warp.eq(2), || {
            data.store(lane + 160, value(5));
            let one = Uint32::from(1u32);
            store_vector(&gflag(2), [one, one], st(MemoryOrder::Release, gpu));
        });
        if_(warp.eq(1), || {
            let unset = |()| {
                let [p, q] = load_vector::<u32, 2>(&gflag(2), ld(MemoryOrder::Acquire, gpu));
                p.eq(0u32) | q.eq(0u32)
            };
            while_carry((), unset, |()| ());
            out.store((row, 1), data.load(lane + 160));
        });
        // A release and an acquire with eviction priorities, warp 1 to 3.
        if_(warp.eq(1), || {
            data.store(lane + 192, value(6));
            let hint = Some(CacheEvictionPriority::NoAllocate);
            store(&gflag(4), 1u32, StoreOptions { level1_eviction_priority: hint, ..st(MemoryOrder::Release, gpu) });
        });
        if_(warp.eq(3), || {
            let hint = Some(CacheEvictionPriority::EvictLast);
            spin(&gflag(4), LoadOptions { level1_eviction_priority: hint, ..ld(MemoryOrder::Acquire, gpu) });
            out.store((row, 1), data.load(lane + 192));
        });
        if_(warp.eq(0), || out.store((row, 1), 0u32));
    });
    fn check(args: &[KernelArg]) {
        const N: usize = 128;
        let (xs, gflags, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for b in 0..2 {
            for t in 0..N {
                let row = b * N + t;
                let from = |warp: usize, k: u32| handoff_value(k, xs[b * N + warp * 32 + t % 32]);
                let want = match t / 32 {
                    0 => [from(3, 3), 0],
                    1 => [from(0, 0), from(2, 5)],
                    2 => [from(1, 2), from(0, 4)],
                    _ => [from(2, 1), from(1, 6)],
                };
                assert_eq!(out[row * 2..row * 2 + 2], want, "block {b} thread {t}");
            }
            assert_eq!(gflags[b * 8..b * 8 + 8], [1, 1, 1, 1, 1, 0, 0, 0], "block {b}'s flags");
        }
    }
    Case { name: "ordered_handoffs", kernel, cfg: launch_n(2, N), small_only: false, check }
}

/// The u32 loads and stores of [`cache_hints`], paired: element `j` of a
/// thread is loaded with `.0` and stored, xor `j * 0x01010101`, with `.1`.
fn hinted_pairs() -> Vec<(LoadOptions, StoreOptions)> {
    use CacheEvictionPriority::*;
    use CacheOperator::*;
    use L2PrefetchSize::*;
    let (cta, gpu, sys) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu), Some(MemoryScope::Sys));
    let (l, s) = (LoadOptions::default(), StoreOptions::default());
    let lc = |c| LoadOptions { cop: Some(c), ..l };
    let le = |e| LoadOptions { level1_eviction_priority: Some(e), ..l };
    let lp = |p| LoadOptions { level_prefetch_size: Some(p), ..l };
    let lo = |sem, scope| LoadOptions { sem, scope, ..l };
    let sc = |c| StoreOptions { cop: Some(c), ..s };
    let se = |e| StoreOptions { level1_eviction_priority: Some(e), ..s };
    let so = |sem, scope| StoreOptions { sem, scope, ..s };
    vec![
        (lc(Ca), sc(Wb)),
        (lc(Cg), sc(Cg)),
        (lc(Cs), sc(Cs)),
        (lc(Lu), sc(Wt)),
        (lc(Cv), se(EvictNormal)),
        (le(EvictNormal), se(EvictFirst)),
        (le(EvictFirst), se(EvictLast)),
        (le(EvictLast), se(EvictUnchanged)),
        (le(EvictUnchanged), se(NoAllocate)),
        (le(NoAllocate), so(MemoryOrder::Weak, None)),
        (lp(Size64B), so(MemoryOrder::Release, gpu)),
        (lp(Size128B), StoreOptions { level1_eviction_priority: Some(NoAllocate), ..so(MemoryOrder::Relaxed, sys) }),
        (lp(Size256B), so(MemoryOrder::Mmio, sys)),
        (
            LoadOptions { cop: Some(Cs), ..lp(Size128B) },
            StoreOptions { level1_eviction_priority: Some(EvictFirst), ..so(MemoryOrder::Sc, gpu) },
        ),
        (lo(MemoryOrder::Weak, None), so(MemoryOrder::Relaxed, cta)),
        (LoadOptions { level_prefetch_size: Some(Size64B), ..le(EvictLast) }, so(MemoryOrder::Volatile, None)),
        (
            LoadOptions {
                level1_eviction_priority: Some(EvictFirst),
                level_prefetch_size: Some(Size256B),
                ..lo(MemoryOrder::Relaxed, gpu)
            },
            so(MemoryOrder::Release, sys),
        ),
        (lo(MemoryOrder::Acquire, sys), so(MemoryOrder::Sc, cta)),
        (lo(MemoryOrder::Mmio, sys), se(EvictLast)),
        (LoadOptions { level1_eviction_priority: Some(EvictLast), ..lo(MemoryOrder::Sc, gpu) }, sc(Cg)),
        (lo(MemoryOrder::Volatile, None), so(MemoryOrder::Relaxed, gpu)),
        (lo(MemoryOrder::Relaxed, cta), so(MemoryOrder::Release, cta)),
        (lo(MemoryOrder::Acquire, gpu), sc(Wt)),
        (LoadOptions { level_prefetch_size: Some(Size128B), ..lo(MemoryOrder::Volatile, None) }, sc(Cs)),
    ]
}

/// Which pairs of [`hinted_pairs`] also take the kernel's L2 cache policy:
/// `(load, store)`.
fn hinted_pair_policy(j: usize) -> (bool, bool) {
    (matches!(j, 14 | 15 | 17), matches!(j, 9 | 10))
}

/// Every cache hint of `ld` and `st` on global memory (each cache operator,
/// L1 eviction priority and L2 prefetch size, an L2 cache policy from
/// `createpolicy`, and their combinations with each other and with relaxed,
/// acquire, release, seq_cst, volatile and MMIO accesses), as scalars and as
/// `.v2` and `.v4` vectors, on every element type (8- and 16-bit elements
/// packed into 32-bit words by the inline PTX), scalar constant
/// (non-coherent, `ld.global.nc`) loads, and cache operators and volatile
/// accesses on shared memory. Each thread moves its own data, so
/// the outputs are its inputs, which the GPU must reproduce bit for bit.
pub fn cache_hints() -> Case {
    use CacheEvictionPriority::*;
    use CacheOperator::*;
    const K: usize = 40;
    let kernel = Kernel::builder("cache_hints").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T * K));
        let hs = param_tensor::<F16>("hs", Layout::col_major(T * 16));
        let bs = param_tensor::<u8>("bs", Layout::col_major(T * 32));
        let ws = param_tensor::<u16>("ws", Layout::col_major(T * 2));
        let ds = param_tensor::<f64>("ds", Layout::col_major(T * 4));
        let ls = param_tensor::<i64>("ls", Layout::col_major(T));
        let fs = param_tensor::<f32>("fs", Layout::col_major(T * 4));
        let qs = param_tensor::<BF16>("qs", Layout::col_major(T * 2));
        let is = param_tensor::<i8>("is", Layout::col_major(T * 4));
        let out = param_tensor::<u32>("out", Layout::col_major(T * K));
        let oh = param_tensor::<F16>("oh", Layout::col_major(T * 16));
        let ob = param_tensor::<u8>("ob", Layout::col_major(T * 32));
        let ow = param_tensor::<u16>("ow", Layout::col_major(T * 2));
        let od = param_tensor::<f64>("od", Layout::col_major(T * 4));
        let ol = param_tensor::<i64>("ol", Layout::col_major(T));
        let of = param_tensor::<f32>("of", Layout::col_major(T * 4));
        let oq = param_tensor::<BF16>("oq", Layout::col_major(T * 2));
        let oi = param_tensor::<i8>("oi", Layout::col_major(T * 4));
        let t = thread_idx().x;
        let at = |n: usize, j: usize| IndexExpr::from(t) * n + j;
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| vec![0]);
        let (l, s) = (LoadOptions::default(), StoreOptions::default());
        let (cta, gpu, sys) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu), Some(MemoryScope::Sys));
        for (j, (lo, so)) in hinted_pairs().into_iter().enumerate() {
            let (lp, sp) = hinted_pair_policy(j);
            let lo = LoadOptions { cache_policy: lp.then_some(policy), ..lo };
            let so = StoreOptions { cache_policy: sp.then_some(policy), ..so };
            let v = load(&xs.ptr().offset(at(K, j)), lo);
            store(&out.ptr().offset(at(K, j)), v ^ (j as u32 * 0x0101_0101), so);
        }
        // Vectors of u32.
        let volatile = MemoryOrder::Volatile;
        let v = load_vector::<u32, 4>(
            &xs.ptr().offset(at(K, 24)),
            LoadOptions { sem: volatile, level_prefetch_size: Some(L2PrefetchSize::Size64B), ..l },
        );
        store_vector(&out.ptr().offset(at(K, 24)), v.map(|x| x ^ 0x5a5a_5a5a), StoreOptions { sem: volatile, ..s });
        let v = load_vector::<u32, 2>(
            &xs.ptr().offset(at(K, 28)),
            LoadOptions { sem: MemoryOrder::Relaxed, scope: cta, level1_eviction_priority: Some(NoAllocate), ..l },
        );
        store_vector(&out.ptr().offset(at(K, 28)), v, StoreOptions { cop: Some(Cs), ..s });
        let v = load_vector::<u32, 2>(
            &xs.ptr().offset(at(K, 30)),
            LoadOptions { sem: MemoryOrder::Sc, scope: gpu, cache_policy: Some(policy), ..l },
        );
        store_vector(
            &out.ptr().offset(at(K, 30)),
            v,
            StoreOptions {
                sem: MemoryOrder::Release,
                scope: gpu,
                level1_eviction_priority: Some(EvictLast),
                cache_policy: Some(policy),
                ..s
            },
        );
        let v = load_vector::<u32, 4>(&xs.ptr().offset(at(K, 32)), LoadOptions { cop: Some(Cg), ..l });
        store_vector(&out.ptr().offset(at(K, 32)), v, StoreOptions { sem: MemoryOrder::Sc, scope: sys, ..s });
        // Cache operators and volatile accesses on shared memory.
        let smem = make_smem_tensor::<u32>(Layout::col_major(3 * T));
        store(&smem.ptr().offset(t), load(&xs.ptr().offset(at(K, 36)), l), StoreOptions { cop: Some(Wt), ..s });
        let pair = load_vector::<u32, 2>(&xs.ptr().offset(at(K, 38)), l);
        store_vector(&smem.ptr().offset(IndexExpr::from(t) * 2 + T), pair, StoreOptions { sem: volatile, ..s });
        sync_threads();
        let next = (t + 1) % T as i32;
        let a = load(&smem.ptr().offset(next), LoadOptions { cop: Some(Cs), ..l });
        out.store(at(K, 36), a);
        out.store(at(K, 37), 0u32);
        let p = load_vector::<u32, 2>(
            &smem.ptr().offset(IndexExpr::from(next) * 2 + T),
            LoadOptions { sem: volatile, ..l },
        );
        store_vector(&out.ptr().offset(at(K, 38)), p, s);
        // Other element types, packed into words where they are narrower.
        let h = load_vector::<F16, 8>(
            &hs.ptr().offset(at(16, 0)),
            LoadOptions { level1_eviction_priority: Some(EvictLast), ..l },
        );
        store_vector(&oh.ptr().offset(at(16, 0)), h, StoreOptions { cop: Some(Cs), ..s });
        let h = load(&hs.ptr().offset(at(16, 8)), LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..l });
        store(&oh.ptr().offset(at(16, 8)), h, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let h = load(&hs.ptr().offset(at(16, 9)), LoadOptions { sem: MemoryOrder::Sc, scope: cta, ..l });
        store(&oh.ptr().offset(at(16, 9)), h, StoreOptions { sem: volatile, ..s });
        // A scalar constant (non-coherent) load: the kernel never writes hs.
        let h = load(&hs.ptr().offset(at(16, 10)), LoadOptions { sem: MemoryOrder::Constant, ..l });
        oh.store(at(16, 10), h);
        for k in 11..16 {
            oh.store(at(16, k), F16::from_f32(0.0));
        }
        let b = load_vector::<u8, 16>(&bs.ptr().offset(at(32, 0)), LoadOptions { sem: volatile, ..l });
        store_vector(&ob.ptr().offset(at(32, 0)), b, StoreOptions { cop: Some(Wt), ..s });
        let b = load_vector::<u8, 2>(&bs.ptr().offset(at(32, 16)), LoadOptions { cop: Some(Cg), ..l });
        store_vector(&ob.ptr().offset(at(32, 16)), b, StoreOptions { level1_eviction_priority: Some(EvictFirst), ..s });
        let b = load(&bs.ptr().offset(at(32, 18)), LoadOptions { level1_eviction_priority: Some(EvictLast), ..l });
        store(&ob.ptr().offset(at(32, 18)), b, StoreOptions { cop: Some(Cs), ..s });
        let b = load_vector::<u8, 4>(
            &bs.ptr().offset(at(32, 20)),
            LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l },
        );
        store_vector(&ob.ptr().offset(at(32, 20)), b, StoreOptions { sem: MemoryOrder::Relaxed, scope: cta, ..s });
        let b = load_vector::<u8, 8>(
            &bs.ptr().offset(at(32, 24)),
            LoadOptions { level_prefetch_size: Some(L2PrefetchSize::Size128B), ..l },
        );
        store_vector(&ob.ptr().offset(at(32, 24)), b, StoreOptions { level1_eviction_priority: Some(NoAllocate), ..s });
        let nc = LoadOptions {
            sem: MemoryOrder::Constant,
            cop: Some(Cs),
            level_prefetch_size: Some(L2PrefetchSize::Size64B),
            ..l
        };
        ob.store(at(32, 19), load(&bs.ptr().offset(at(32, 19)), nc));
        let w = load_vector::<u16, 2>(
            &ws.ptr().offset(at(2, 0)),
            LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, level1_eviction_priority: Some(EvictLast), ..l },
        );
        store_vector(&ow.ptr().offset(at(2, 0)), w, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let d = load_vector::<f64, 2>(
            &ds.ptr().offset(at(4, 0)),
            LoadOptions { level1_eviction_priority: Some(NoAllocate), ..l },
        );
        store_vector(&od.ptr().offset(at(4, 0)), d, StoreOptions { cop: Some(Wt), ..s });
        let d = load(&ds.ptr().offset(at(4, 2)), LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l });
        store(&od.ptr().offset(at(4, 2)), d, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let d = load(&ds.ptr().offset(at(4, 3)), LoadOptions { sem: volatile, ..l });
        store(&od.ptr().offset(at(4, 3)), d, StoreOptions { sem: MemoryOrder::Relaxed, scope: sys, ..s });
        let q = load(&ls.ptr().offset(t), LoadOptions { cache_policy: Some(policy), ..l });
        store(&ol.ptr().offset(t), q, StoreOptions { cop: Some(Cg), ..s });
        let f = load_vector::<f32, 4>(
            &fs.ptr().offset(at(4, 0)),
            LoadOptions { level_prefetch_size: Some(L2PrefetchSize::Size256B), ..l },
        );
        store_vector(
            &of.ptr().offset(at(4, 0)),
            f,
            StoreOptions { level1_eviction_priority: Some(EvictUnchanged), ..s },
        );
        let e = load(
            &qs.ptr().offset(at(2, 0)),
            LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, level1_eviction_priority: Some(EvictFirst), ..l },
        );
        store(
            &oq.ptr().offset(at(2, 0)),
            e,
            StoreOptions { sem: MemoryOrder::Relaxed, scope: gpu, level1_eviction_priority: Some(EvictLast), ..s },
        );
        let e = load(&qs.ptr().offset(at(2, 1)), LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..l });
        store(&oq.ptr().offset(at(2, 1)), e, StoreOptions { sem: MemoryOrder::Release, scope: cta, ..s });
        let i = load(&is.ptr().offset(at(4, 0)), LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..l });
        store(&oi.ptr().offset(at(4, 0)), i, StoreOptions { sem: MemoryOrder::Relaxed, scope: cta, ..s });
        let i = load(
            &is.ptr().offset(at(4, 1)),
            LoadOptions { sem: MemoryOrder::Acquire, scope: sys, level1_eviction_priority: Some(EvictFirst), ..l },
        );
        store(
            &oi.ptr().offset(at(4, 1)),
            i,
            StoreOptions { sem: MemoryOrder::Release, scope: gpu, level1_eviction_priority: Some(NoAllocate), ..s },
        );
        let i = load_vector::<i8, 2>(&is.ptr().offset(at(4, 2)), LoadOptions { sem: volatile, ..l });
        store_vector(&oi.ptr().offset(at(4, 2)), i, StoreOptions { sem: MemoryOrder::Sc, scope: gpu, ..s });
    });
    fn check(args: &[KernelArg]) {
        const K: usize = 40;
        let xs = args[0].to_vec::<u32>();
        let out = args[9].to_vec::<u32>();
        for t in 0..T {
            let x = &xs[t * K..t * K + K];
            let o = &out[t * K..t * K + K];
            for j in 0..24 {
                assert_eq!(o[j], x[j] ^ (j as u32 * 0x0101_0101), "thread {t} pair {j}");
            }
            let flip: Vec<u32> = x[24..28].iter().map(|v| v ^ 0x5a5a_5a5a).collect();
            assert_eq!(o[24..28], flip[..], "thread {t}");
            assert_eq!(o[28..36], x[28..36], "thread {t}");
            let n = (t + 1) % T;
            assert_eq!([o[36], o[37]], [xs[n * K + 36], 0], "thread {t}");
            assert_eq!(o[38..40], xs[n * K + 38..n * K + 40], "thread {t}");
        }
        // F16, u8, u16, f64, i64, f32, bf16, i8: each output holds its input
        // where the kernel moved data, and zeros where it filled.
        let h_in = args[1].to_vec::<F16>();
        let h_out = args[10].to_vec::<F16>();
        for t in 0..T {
            assert_eq!(h_out[t * 16..t * 16 + 11], h_in[t * 16..t * 16 + 11], "halves of thread {t}");
            assert!(h_out[t * 16 + 11..t * 16 + 16].iter().all(|h| h.to_f32() == 0.0));
        }
        let (b_in, b_out) = (args[2].to_vec::<u8>(), args[11].to_vec::<u8>());
        for t in 0..T {
            assert_eq!(b_out[t * 32..t * 32 + 32], b_in[t * 32..t * 32 + 32], "bytes of thread {t}");
        }
        assert_eq!(args[12].to_vec::<u16>(), args[3].to_vec::<u16>());
        let (d_in, d_out) = (args[4].to_vec::<f64>(), args[13].to_vec::<f64>());
        assert!(d_in.iter().zip(&d_out).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(args[14].to_vec::<i64>(), args[5].to_vec::<i64>());
        let (f_in, f_out) = (args[6].to_vec::<f32>(), args[15].to_vec::<f32>());
        assert!(f_in.iter().zip(&f_out).all(|(a, b)| a.to_bits() == b.to_bits()));
        let (q_in, q_out) = (args[7].to_vec::<BF16>(), args[16].to_vec::<BF16>());
        assert!(q_in.iter().zip(&q_out).all(|(a, b)| a.0 == b.0));
        assert_eq!(args[17].to_vec::<i8>(), args[8].to_vec::<i8>());
    }
    Case { name: "cache_hints", kernel, cfg: launch(1), small_only: false, check }
}

/// `cute.arch.prefetch` in each form: into L1 and L2, L2 with each eviction
/// priority, predicated, uniform (`prefetchu.L1`), through a pointer of
/// another element type, past the end of the buffer (harmless: nothing is
/// read), and on per-thread memory (generic addressing). The prefetches
/// change nothing, so the result is the plain copy.
pub fn prefetches() -> Case {
    use PrefetchCacheLevel::L2;
    let kernel = Kernel::builder("prefetches").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let out = param_tensor::<f32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let p = xs.ptr().offset(t);
        let l2 = PrefetchOptions { cache_level: Some(L2), ..Default::default() };
        prefetch(&p, PrefetchOptions::default());
        prefetch(&p, l2);
        prefetch(&p, PrefetchOptions { evict_priority: Some(CacheEvictionPriority::EvictLast), ..l2 });
        prefetch(&p, PrefetchOptions { evict_priority: Some(CacheEvictionPriority::EvictNormal), ..l2 });
        prefetch(&p, PrefetchOptions { uniform: true, ..Default::default() });
        prefetch(&xs.ptr().offset(IndexExpr::from(t) + T), PrefetchOptions { predicate: Some(t.lt(32)), ..l2 });
        prefetch(&p.cast::<u8>().offset(2usize), PrefetchOptions::default());
        let stack = make_rmem_tensor::<f32>(Layout::col_major(4));
        stack.store(0, xs.load(t));
        prefetch(&stack.ptr(), PrefetchOptions::default());
        prefetch(&stack.ptr(), PrefetchOptions { predicate: Some(t.ge(16)), ..l2 });
        out.store(t, stack.load(0) * 2.0);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..T {
            assert_eq!(out[t].to_bits(), (xs[t] * 2.0).to_bits(), "thread {t}");
        }
    }
    Case { name: "prefetches", kernel, cfg: launch(1), small_only: false, check }
}

/// Hand-offs at cluster scope (SM90): a release and an acquire on a shared
/// flag, and a vector release and acquire with eviction priorities on a
/// global flag, in each CTA of a cluster of two, which have flags of their
/// own.
pub fn cluster_scope_handoff() -> Case {
    const N: usize = 64;
    let kernel = Kernel::builder("cluster_scope_handoff").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(2 * N));
        let gflags = param_tensor::<u32>("gflags", Layout::col_major(2 * 4));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * N, 2)));
        let t = thread_idx().x;
        let b = block_idx().x;
        let row = b * N as i32 + t;
        let x = xs.load(row);
        let lane = t % 32;
        let data = make_smem_tensor::<u32>(Layout::col_major(2 * 32));
        let sflag = alloc_smem::<u32>(1, 4);
        let gflag = gflags.ptr().offset(IndexExpr::from(b) * 4);
        let cluster = Some(MemoryScope::Cluster);
        if_(t.eq(0), || sflag.store(0u32));
        if_(t.lt(4), || gflags.ptr().offset(IndexExpr::from(b) * 4 + t).store(0u32));
        sync_threads();
        if_(warp_idx().eq(0), || {
            data.store(lane, x + 7u32);
            store(&sflag, 1u32, StoreOptions { sem: MemoryOrder::Release, scope: cluster, ..Default::default() });
        });
        if_(warp_idx().eq(1), || {
            let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: cluster, ..Default::default() };
            while_carry((), |()| load(&sflag, acquire).eq(0u32), |()| ());
            out.store((row, 0), data.load(lane));
            data.store(lane + 32, x * 5u32);
            let one = Uint32::from(1u32);
            let hint = Some(CacheEvictionPriority::NoAllocate);
            let release = StoreOptions {
                sem: MemoryOrder::Release,
                scope: cluster,
                level1_eviction_priority: hint,
                ..Default::default()
            };
            store_vector(&gflag, [one, one], release);
        });
        if_(warp_idx().eq(0), || {
            let hint = Some(CacheEvictionPriority::EvictLast);
            let acquire = LoadOptions {
                sem: MemoryOrder::Acquire,
                scope: cluster,
                level1_eviction_priority: hint,
                ..Default::default()
            };
            let unset = |()| {
                let [p, q] = load_vector::<u32, 2>(&gflag, acquire);
                p.eq(0u32) | q.eq(0u32)
            };
            while_carry((), unset, |()| ());
            out.store((row, 0), data.load(lane + 32));
        });
        out.store(
            (row, 1),
            load(&sflag, LoadOptions { sem: MemoryOrder::Relaxed, scope: cluster, ..Default::default() }),
        );
    });
    fn check(args: &[KernelArg]) {
        const N: usize = 64;
        let (xs, gflags, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for b in 0..2 {
            for t in 0..N {
                let row = b * N + t;
                let from = |warp: usize| xs[b * N + warp * 32 + t % 32];
                let want = if t < 32 { from(1).wrapping_mul(5) } else { from(0).wrapping_add(7) };
                assert_eq!(out[row * 2..row * 2 + 2], [want, 1], "block {b} thread {t}");
            }
            assert_eq!(gflags[b * 4..b * 4 + 4], [1, 1, 0, 0], "block {b}'s flags");
        }
    }
    let cfg = LaunchConfig::new([2, 1, 1], [N as u32, 1, 1]).with_cluster([2, 1, 1]);
    Case { name: "cluster_scope_handoff", kernel, cfg, small_only: false, check }
}

/// Tiled copies of a 32 x 16 tile from global memory to registers and back
/// with the G2R and R2G atoms (CuTe DSL's `CopyG2ROp` and `CopyR2GOp`) and
/// their memory attributes: volatile, acquire, seq_cst and MMIO loads,
/// release, relaxed, seq_cst and MMIO stores, invariant (non-coherent) and
/// constant loads of a buffer the kernel never writes, an L2 prefetch size,
/// eviction priorities, the cache modes, a cache policy given at copy time,
/// `num_bits_per_copy = 0` (vectorised by the copy, or element by element
/// under a predicate and for MMIO), predicated copies, 16-bit elements, and
/// a `make_rmem_tensor` on the register side.
pub fn global_copy_atoms() -> Case {
    use CacheEvictionPriority::*;
    const M: usize = 32;
    const N: usize = 16;
    let kernel = Kernel::builder("global_copy_atoms").block_dim([T as u32, 1, 1]).build(|| {
        let tile = Layout::col_major((M, N));
        let a = param_tensor::<f32>("a", tile.clone());
        let b = param_tensor::<f32>("b", tile.clone());
        let h = param_tensor::<F16>("h", tile.clone());
        let o: Vec<Tensor<f32>> = (0..8).map(|k| param_tensor::<f32>(&format!("o{k}"), tile.clone())).collect();
        let oh = param_tensor::<F16>("oh", tile.clone());
        let t = thread_idx().x;
        let (cta, gpu, sys) = (MemoryScope::Cta, MemoryScope::Gpu, MemoryScope::Sys);
        let g2r = |bits, attrs| make_copy_atom::<f32>(CopyOp::G2R { bits, attrs });
        let r2g = |bits, attrs| make_copy_atom::<f32>(CopyOp::R2G { bits, attrs });
        let load = |memory_order, memory_scope| G2RAttrs { memory_order, memory_scope, ..G2RAttrs::default() };
        let store = |memory_order, memory_scope| R2GAttrs { memory_order, memory_scope, ..R2GAttrs::default() };
        let tiled =
            |atom: &CopyAtom<f32>| make_tiled_copy_tv(atom, &Layout::col_major((8, 8)), &Layout::col_major((4, 2)));
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_first.b64 $0, 0.5;", &[], |_| vec![0]);
        let move_tile =
            |src: &Tensor<f32>, g: CopyAtom<f32>, dst: &Tensor<f32>, r: CopyAtom<f32>, policy: Option<Int64>| {
                let (tg, tr) = (tiled(&g), tiled(&r));
                let s = tg.get_slice(t).partition_s(src);
                let frag = make_fragment_like::<f32, f32>(&s);
                copy_with(&tg, &s, &frag, CopyOptions { cache_policy: policy, ..CopyOptions::default() });
                copy_with(
                    &tr,
                    &frag,
                    &tr.get_slice(t).partition_d(dst),
                    CopyOptions { cache_policy: policy, ..CopyOptions::default() },
                );
            };
        let evict_last = R2GAttrs { l1c_evict_priority: EvictLast, ..store(MemoryOrder::Release, gpu) };
        move_tile(&a, g2r(128, load(MemoryOrder::Volatile, sys)), &o[0], r2g(128, evict_last), None);
        let no_allocate = R2GAttrs { l1c_evict_priority: NoAllocate, ..store(MemoryOrder::Relaxed, cta) };
        move_tile(&a, g2r(128, load(MemoryOrder::Acquire, gpu)), &o[1], r2g(64, no_allocate), None);
        let invariant = G2RAttrs {
            invariant: true,
            l2_prefetch_size: Some(L2PrefetchSize::Size256B),
            load_cache_mode: LoadCacheMode::Streaming,
            ..G2RAttrs::default()
        };
        let streaming = R2GAttrs { store_cache_mode: StoreCacheMode::Streaming, ..R2GAttrs::default() };
        move_tile(&b, g2r(128, invariant), &o[2], r2g(128, streaming), None);
        let through = R2GAttrs { store_cache_mode: StoreCacheMode::WriteThrough, ..R2GAttrs::default() };
        move_tile(&b, g2r(0, load(MemoryOrder::Constant, cta)), &o[3], r2g(0, through), None);
        let last_use = G2RAttrs { load_cache_mode: LoadCacheMode::LastUse, ..G2RAttrs::default() };
        let global = R2GAttrs { store_cache_mode: StoreCacheMode::Global, ..R2GAttrs::default() };
        move_tile(&a, g2r(64, last_use), &o[4], r2g(64, global), Some(policy));
        move_tile(&a, g2r(128, load(MemoryOrder::Sc, gpu)), &o[5], r2g(32, store(MemoryOrder::Sc, cta)), None);
        move_tile(&a, g2r(0, load(MemoryOrder::Mmio, sys)), &o[6], r2g(32, store(MemoryOrder::Mmio, sys)), None);
        // Rows 0 to 19 only, element by element.
        let (tg, tr) =
            (tiled(&g2r(0, load(MemoryOrder::Relaxed, gpu))), tiled(&r2g(0, store(MemoryOrder::Release, gpu))));
        let thr = tg.get_slice(t);
        let s = thr.partition_s(&a);
        let rows = IntTuple::from((20, N));
        let pred = make_predicate(&thr.partition_s(&make_identity_tensor((M, N))), |c| elem_less(c, &rows));
        let frag = make_fragment_like::<f32, f32>(&s);
        frag.fill(0.0);
        copy_with(&tg, &s, &frag, CopyOptions { pred: Some(&pred), ..CopyOptions::default() });
        copy_with(
            &tr,
            &frag,
            &tr.get_slice(t).partition_d(&o[7]),
            CopyOptions { pred: Some(&pred), ..CopyOptions::default() },
        );
        // Eight halves per thread, through the non-coherent cache.
        let hg = make_copy_atom::<F16>(CopyOp::G2R {
            bits: 128,
            attrs: G2RAttrs { invariant: true, l1c_evict_priority: EvictFirst, ..G2RAttrs::default() },
        });
        let hr = make_copy_atom::<F16>(CopyOp::R2G { bits: 128, attrs: store(MemoryOrder::Volatile, sys) });
        let (thr_h, val_h) = (Layout::col_major((4, 16)), Layout::col_major((8, 1)));
        let (tg, tr) = (make_tiled_copy_tv(&hg, &thr_h, &val_h), make_tiled_copy_tv(&hr, &thr_h, &val_h));
        let s = tg.get_slice(t).partition_s(&h);
        let frag = make_fragment_like::<F16, F16>(&s);
        copy(&tg, &s, &frag);
        copy(&tr, &frag, &tr.get_slice(t).partition_d(&oh));
        // The register side as a make_rmem_tensor (CuTe DSL's register
        // tensor, per-thread memory here) rather than a fragment: an acquire
        // load into it and a release store from it (vectorised by the copy).
        let orm = param_tensor::<f32>("orm", tile.clone());
        let (tg, tr) =
            (tiled(&g2r(128, load(MemoryOrder::Acquire, gpu))), tiled(&r2g(0, store(MemoryOrder::Release, cta))));
        let s = tg.get_slice(t).partition_s(&b);
        let stack = make_rmem_tensor::<f32>(Layout::col_major(s.shape().clone()));
        copy(&tg, &s, &stack);
        copy(&tr, &stack, &tr.get_slice(t).partition_d(&orm));
    });
    fn check(args: &[KernelArg]) {
        let (a, b) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (k, want) in [(0, &a), (1, &a), (2, &b), (3, &b), (4, &a), (5, &a), (6, &a)] {
            assert_eq!(bits(&args[3 + k].to_vec::<f32>()), bits(want), "o{k}");
        }
        let o7 = args[10].to_vec::<f32>();
        for n in 0..16 {
            for m in 0..20 {
                assert_eq!(o7[m + 32 * n].to_bits(), a[m + 32 * n].to_bits(), "o7[{m}, {n}]");
            }
        }
        let (h, oh) = (args[2].to_vec::<F16>(), args[11].to_vec::<F16>());
        assert!(h.iter().zip(&oh).all(|(x, y)| x.0 == y.0), "oh");
        assert_eq!(bits(&args[12].to_vec::<f32>()), bits(&b), "orm");
    }
    Case { name: "global_copy_atoms", kernel, cfg: launch(1), small_only: false, check }
}

/// The owner `make_cotiled_copy` gives the element at address `a` of the
/// tiles of [`cotiled_copies`]: thread `(t0, t1)` (index `t0 + 4 t1`) copies
/// addresses `32 t0 + 4 t1 .. + 4`.
pub fn cotiled_owner(a: usize) -> usize {
    a / 32 + 4 * ((a % 32) / 4)
}

/// `make_cotiled_copy` in copies: 32 threads with four values each, spread
/// over a 16 x 8 tile by the atom layout `((4, 8), 4):((32, 4), 1)` (thread
/// `(t0, t1)` takes addresses `32 t0 + 4 t1 .. + 4`), on a row-major tile
/// (whose tiler is the whole tile) and on a column-major one (whose tiler
/// selects rows out of order). Each tile goes from global memory through
/// registers to another tensor in 128-bit G2R (invariant) and R2G copies,
/// and each thread writes its index where it copied, so the outputs show
/// the copy and the thread that `make_cotiled_copy` assigned each element.
pub fn cotiled_copies() -> Case {
    const M: usize = 16;
    const N: usize = 8;
    let kernel = Kernel::builder("cotiled_copies").block_dim([32, 1, 1]).build(|| {
        let t = thread_idx().x;
        let atom_tv = Layout::new(((4, 8), 4), ((32, 4), 1));
        let invariant = G2RAttrs { invariant: true, ..G2RAttrs::default() };
        let g2r = make_copy_atom::<f32>(CopyOp::G2R { bits: 128, attrs: invariant });
        let r2g = make_copy_atom::<f32>(CopyOp::R2G { bits: 128, attrs: R2GAttrs::default() });
        for (k, data) in [Layout::row_major((M, N)), Layout::col_major((M, N))].into_iter().enumerate() {
            let a = param_tensor::<f32>(&format!("a{k}"), data.clone());
            let b = param_tensor::<f32>(&format!("b{k}"), data.clone());
            let owner = param_tensor::<i32>(&format!("owner{k}"), data.clone());
            let (load, store) = (make_cotiled_copy(&g2r, &atom_tv, &data), make_cotiled_copy(&r2g, &atom_tv, &data));
            let src = load.get_slice(t).partition_s(&a);
            let frag = make_fragment_like::<f32, f32>(&src);
            copy(&load, &src, &frag);
            copy(&store, &frag, &store.get_slice(t).partition_d(&b));
            let id = make_copy_atom::<i32>(CopyOp::R2G { bits: 128, attrs: R2GAttrs::default() });
            let mark = make_cotiled_copy(&id, &atom_tv, &data);
            let dst = mark.get_slice(t).partition_d(&owner);
            let mine = make_fragment_like::<i32, i32>(&dst);
            mine.fill(t);
            copy(&mark, &mine, &dst);
        }
    });
    fn check(args: &[KernelArg]) {
        for k in 0..2 {
            let (a, b, owner) =
                (args[3 * k].to_vec::<f32>(), args[3 * k + 1].to_vec::<f32>(), args[3 * k + 2].to_vec::<i32>());
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "tile {k}");
            for (addr, &o) in owner.iter().enumerate() {
                assert_eq!(o as usize, cotiled_owner(addr), "tile {k}, address {addr}");
            }
        }
    }
    Case { name: "cotiled_copies", kernel, cfg: launch_n(1, 32), small_only: false, check }
}

/// Strong (ordered and volatile) vectors of 8- and 16-bit elements, which
/// PTX moves with their element type (`.v4.b8`, `.v2.b16`), since its
/// memory model treats each element of a vector access as one access, and
/// one instruction per four elements beyond four: relaxed, acquire,
/// release, seq_cst and volatile vectors of 2, 4, 8 and 16 elements on
/// global and shared memory, with an eviction priority and a cache policy,
/// and a flag hand-off between warps in which the consumer acquires one byte
/// of the producer's vector release (the same bytes as one of its elements,
/// so the two synchronise). Each thread's outputs are its inputs, or its
/// neighbour's through shared memory, which the GPU must reproduce bit for
/// bit.
pub fn strong_vectors() -> Case {
    use CacheEvictionPriority::*;
    let kernel = Kernel::builder("strong_vectors").block_dim([T as u32, 1, 1]).build(|| {
        let hs = param_tensor::<F16>("hs", Layout::col_major(T * 8));
        let bs = param_tensor::<u8>("bs", Layout::col_major(T * 16));
        let is = param_tensor::<i8>("is", Layout::col_major(T * 8));
        let qs = param_tensor::<BF16>("qs", Layout::col_major(T * 4));
        let ws = param_tensor::<u16>("ws", Layout::col_major(T * 2));
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let oh = param_tensor::<F16>("oh", Layout::col_major(T * 8));
        let ob = param_tensor::<u8>("ob", Layout::col_major(T * 16));
        let oi = param_tensor::<i8>("oi", Layout::col_major(T * 8));
        let oq = param_tensor::<BF16>("oq", Layout::col_major(T * 4));
        let ow = param_tensor::<u16>("ow", Layout::col_major(T * 2));
        let os = param_tensor::<u8>("os", Layout::col_major(T * 8));
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let at = |n: usize| IndexExpr::from(t) * n;
        let (cta, gpu, sys) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu), Some(MemoryScope::Sys));
        let lo = |sem, scope| LoadOptions { sem, scope, ..Default::default() };
        let so = |sem, scope| StoreOptions { sem, scope, ..Default::default() };
        // Eight halves: two instructions each way.
        let h = load_vector::<F16, 8>(&hs.ptr().offset(at(8)), lo(MemoryOrder::Relaxed, gpu));
        store_vector(&oh.ptr().offset(at(8)), h, so(MemoryOrder::Release, gpu));
        // Sixteen bytes: four instructions each way, after one fence for the
        // seq_cst store.
        let b = load_vector::<u8, 16>(&bs.ptr().offset(at(16)), lo(MemoryOrder::Acquire, cta));
        store_vector(&ob.ptr().offset(at(16)), b, so(MemoryOrder::Sc, gpu));
        let i = load_vector::<i8, 8>(&is.ptr().offset(at(8)), lo(MemoryOrder::Sc, sys));
        store_vector(&oi.ptr().offset(at(8)), i, so(MemoryOrder::Volatile, None));
        let q = load_vector::<BF16, 4>(&qs.ptr().offset(at(4)), lo(MemoryOrder::Volatile, None));
        store_vector(&oq.ptr().offset(at(4)), q, so(MemoryOrder::Relaxed, cta));
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| vec![0]);
        let w = load_vector::<u16, 2>(
            &ws.ptr().offset(at(2)),
            LoadOptions {
                level1_eviction_priority: Some(EvictFirst),
                cache_policy: Some(policy),
                ..lo(MemoryOrder::Acquire, gpu)
            },
        );
        store_vector(
            &ow.ptr().offset(at(2)),
            w,
            StoreOptions {
                level1_eviction_priority: Some(NoAllocate),
                cache_policy: Some(policy),
                ..so(MemoryOrder::Relaxed, sys)
            },
        );
        // Shared memory: eight bytes per thread with a release, read by the
        // next thread with an acquire after a barrier.
        let smem = make_smem_tensor::<u8>(Layout::col_major(T * 8));
        let mine = load_vector::<u8, 8>(&bs.ptr().offset(at(16)), LoadOptions::default());
        store_vector(&smem.ptr().offset(at(8)), mine, so(MemoryOrder::Release, cta));
        sync_threads();
        let next = (t + 1) % T as i32;
        let theirs = load_vector::<u8, 8>(&smem.ptr().offset(IndexExpr::from(next) * 8), lo(MemoryOrder::Acquire, cta));
        store_vector(&os.ptr().offset(at(8)), theirs, StoreOptions::default());
        // Warp 0 writes data and then four flag bytes with one release; warp
        // 1 spins on byte 2 with an acquire and reads the data.
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        let flags = alloc_smem::<u8>(4, 4);
        if_(t.lt(4), || flags.offset(t).store(0u8));
        sync_threads();
        let lane = t % 32;
        if_(warp_idx().eq(0), || {
            data.store(lane, xs.load(lane) * 3u32 + 1u32);
            let one = Uint8::from(1u8);
            store_vector(&flags, [one, one, one, one], so(MemoryOrder::Release, cta));
            out.store(t, 0u32);
        });
        if_(warp_idx().eq(1), || {
            while_carry((), |()| load(&flags.offset(2usize), lo(MemoryOrder::Acquire, cta)).eq(0u8), |()| ());
            out.store(t, data.load(lane));
        });
    });
    fn check(args: &[KernelArg]) {
        let half_bits = |a: &KernelArg| a.to_vec::<F16>().iter().map(|h| h.0).collect::<Vec<_>>();
        let bf16_bits = |a: &KernelArg| a.to_vec::<BF16>().iter().map(|h| h.0).collect::<Vec<_>>();
        assert_eq!(half_bits(&args[6]), half_bits(&args[0]), "oh");
        let (bs, ob) = (args[1].to_vec::<u8>(), args[7].to_vec::<u8>());
        assert_eq!(ob, bs, "ob");
        assert_eq!(args[8].to_vec::<i8>(), args[2].to_vec::<i8>(), "oi");
        assert_eq!(bf16_bits(&args[9]), bf16_bits(&args[3]), "oq");
        assert_eq!(args[10].to_vec::<u16>(), args[4].to_vec::<u16>(), "ow");
        let os = args[11].to_vec::<u8>();
        for t in 0..T {
            let n = (t + 1) % T;
            assert_eq!(os[t * 8..t * 8 + 8], bs[n * 16..n * 16 + 8], "os of thread {t}");
        }
        let (xs, out) = (args[5].to_vec::<u32>(), args[12].to_vec::<u32>());
        for t in 0..T {
            let want = if t < 32 { 0 } else { xs[t - 32].wrapping_mul(3).wrapping_add(1) };
            assert_eq!(out[t], want, "out[{t}]");
        }
    }
    Case { name: "strong_vectors", kernel, cfg: launch(1), small_only: false, check }
}

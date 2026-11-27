//! Atomics with memory semantics and scope, the float and packed-half adds,
//! `atomic_fmax` and `atomic_fmin`, reductions (`red`), and a spin lock
//! built from acquire and release atomics. Every result is independent of
//! the order the threads' atomics happen in, so the GPU must match the
//! simulator bit for bit.

use cute::prelude::*;

use super::{Case, SPIN_LIMIT, launch_n};

/// Threads in the single block of most of these kernels: four warps.
const N: usize = 128;

/// The `sem=` and `scope=` combinations the integer kernel runs: the
/// default, then each semantics at each scope but cluster (sm_90).
pub fn order_combos() -> Vec<AtomicOptions> {
    use MemoryOrder::*;
    let mut v = vec![AtomicOptions::default()];
    for sem in [Relaxed, Acquire, Release, AcqRel] {
        for scope in [MemoryScope::Cta, MemoryScope::Gpu, MemoryScope::Sys] {
            v.push(AtomicOptions::new(sem, scope));
        }
    }
    v
}

/// The operations of [`atomic_orders`] per combination and width: add,
/// AND, OR, XOR, unsigned min and max, signed min and max, exchange,
/// compare-and-swap, and a count of the swaps that succeeded.
const KINDS: usize = 11;

/// What each cell of [`atomic_orders`] starts as, by kind.
pub fn order_init(kind: usize, wide: bool) -> u64 {
    let ones = if wide { u64::MAX } else { u32::MAX as u64 };
    let smax = ones >> 1;
    match kind {
        1 | 4 => ones,
        6 => smax,
        7 => smax + 1,
        9 => 3,
        _ => 0,
    }
}

/// [`atomic_orders`]'s cells: every combination's kinds.
pub fn order_cells() -> usize {
    order_combos().len() * KINDS
}

/// The integer atomics, 32- and 64-bit, with every `sem=` and `scope=`
/// (relaxed, acquire, release and acq_rel at CTA, GPU and system scope, and
/// the default), on global and shared memory, every thread of a block on
/// each cell.
pub fn atomic_orders() -> Case {
    let cells = order_cells();
    let kernel = Kernel::builder("atomic_orders").block_dim([N as u32, 1, 1]).build(move || {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let ys = param_tensor::<u64>("ys", Layout::col_major(N));
        let g32 = param_tensor::<u32>("g32", Layout::col_major(cells));
        let g64 = param_tensor::<u64>("g64", Layout::col_major(cells));
        let out32 = param_tensor::<u32>("out32", Layout::col_major(cells));
        let out64 = param_tensor::<u64>("out64", Layout::col_major(cells));
        let s32 = make_smem_tensor::<u32>(Layout::col_major(cells));
        let s64 = make_smem_tensor::<u64>(Layout::col_major(cells));
        let t = thread_idx().x;
        let init32 = make_rmem_tensor::<u32>(Layout::col_major(KINDS));
        let init64 = make_rmem_tensor::<u64>(Layout::col_major(KINDS));
        for k in 0..KINDS {
            init32.store(k, order_init(k, false) as u32);
            init64.store(k, order_init(k, true));
        }
        for base in (0..cells).step_by(N) {
            let c = t + base as i32;
            if_(c.lt(cells as i32), || {
                let kind = c % KINDS as i32;
                g32.store(c, init32.load(kind));
                s32.store(c, init32.load(kind));
                g64.store(c, init64.load(kind));
                s64.store(c, init64.load(kind));
            });
        }
        sync_threads();
        let (x, y) = (xs.load(t), ys.load(t));
        for (k, opts) in order_combos().into_iter().enumerate() {
            for (p32, p64) in [(g32.ptr(), g64.ptr()), (s32.ptr(), s64.ptr())] {
                let at32 = |i: usize| p32.offset(k * KINDS + i);
                let at64 = |i: usize| p64.offset(k * KINDS + i);
                let _ = atomic_add_with(&at32(0), x, opts);
                let _ = atomic_and_with(&at32(1), x, opts);
                let _ = atomic_or_with(&at32(2), x, opts);
                let _ = atomic_xor_with(&at32(3), x, opts);
                let _ = atomic_min_with(&at32(4), x, opts);
                let _ = atomic_max_with(&at32(5), x, opts);
                let _ = atomic_min_with(&at32(6).cast::<i32>(), x.bitcast::<i32>(), opts);
                let _ = atomic_max_with(&at32(7).cast::<i32>(), x.bitcast::<i32>(), opts);
                let _ = atomic_exch_with(&at32(8), 5u32, opts);
                let old = atomic_cas_with(&at32(9), 3u32, 77u32, opts);
                if_(old.eq(3u32), || {
                    let _ = atomic_add_with(&at32(10), 1u32, opts);
                });
                let _ = atomic_add_with(&at64(0), y, opts);
                let _ = atomic_and_with(&at64(1), y, opts);
                let _ = atomic_or_with(&at64(2), y, opts);
                let _ = atomic_xor_with(&at64(3), y, opts);
                let _ = atomic_min_with(&at64(4), y, opts);
                let _ = atomic_max_with(&at64(5), y, opts);
                let _ = atomic_min_with(&at64(6).cast::<i64>(), y.bitcast::<i64>(), opts);
                let _ = atomic_max_with(&at64(7).cast::<i64>(), y.bitcast::<i64>(), opts);
                let _ = atomic_exch_with(&at64(8), 5u64, opts);
                let old = atomic_cas_with(&at64(9), 3u64, 77u64, opts);
                if_(old.eq(3u64), || {
                    let _ = atomic_add_with(&at64(10), 1u64, opts);
                });
            }
        }
        sync_threads();
        for base in (0..cells).step_by(N) {
            let c = t + base as i32;
            if_(c.lt(cells as i32), || {
                out32.store(c, s32.load(c));
                out64.store(c, s64.load(c));
            });
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>());
        let want = |kind: usize, wide: bool| -> u64 {
            let vals: Vec<u64> = if wide { ys.clone() } else { xs.iter().map(|&x| x as u64).collect() };
            let mask = if wide { u64::MAX } else { u32::MAX as u64 };
            let signed = |v: u64| if wide { v as i64 } else { v as u32 as i32 as i64 };
            let init = order_init(kind, wide);
            let r = match kind {
                0 => vals.iter().fold(init, |a, &v| a.wrapping_add(v)),
                1 => vals.iter().fold(init, |a, &v| a & v),
                2 => vals.iter().fold(init, |a, &v| a | v),
                3 => vals.iter().fold(init, |a, &v| a ^ v),
                4 => vals.iter().fold(init, |a, &v| a.min(v)),
                5 => vals.iter().fold(init, |a, &v| a.max(v)),
                6 => vals.iter().fold(init, |a, &v| if signed(v) < signed(a) { v } else { a }),
                7 => vals.iter().fold(init, |a, &v| if signed(v) > signed(a) { v } else { a }),
                8 => 5,
                9 => 77,
                _ => 1,
            };
            r & mask
        };
        let (g32, g64) = (args[2].to_vec::<u32>(), args[3].to_vec::<u64>());
        let (s32, s64) = (args[4].to_vec::<u32>(), args[5].to_vec::<u64>());
        for c in 0..order_cells() {
            let (combo, kind) = (c / KINDS, c % KINDS);
            let what = format!("combination {combo} ({:?}), kind {kind}", order_combos()[combo]);
            assert_eq!(g32[c] as u64, want(kind, false), "global 32-bit, {what}");
            assert_eq!(s32[c] as u64, want(kind, false), "shared 32-bit, {what}");
            assert_eq!(g64[c], want(kind, true), "global 64-bit, {what}");
            assert_eq!(s64[c], want(kind, true), "shared 64-bit, {what}");
        }
    }
    Case { name: "atomic_orders", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// A spin lock in shared memory and one in global memory, each taken with
/// an acquire `atomic_cas` and given back with a release `atomic_exch`,
/// around plain reads and writes of a counter, by lane 0 of each warp. The
/// critical section is inside the loop that tries for the lock, so a warp
/// whose lanes run together also gets through it. Each warp counts whether
/// it got in (bounded by [`SPIN_LIMIT`] tries).
pub fn atomic_lock() -> Case {
    let kernel = Kernel::builder("atomic_lock").block_dim([N as u32, 1, 1]).build(|| {
        let g = param_tensor::<u32>("g", Layout::col_major(2));
        let out = param_tensor::<u32>("out", Layout::col_major(2 + 2 * N / 32));
        let lock = alloc_smem::<u32>(1, 4);
        let count = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        let warp = warp_idx();
        if_(t.eq(0), || {
            lock.store(0u32);
            count.store(0u32);
            g.store(0, 0u32);
            g.store(1, 0u32);
        });
        sync_threads();
        let locks =
            [(lock.clone(), count.clone(), MemoryScope::Cta), (g.ptr(), g.ptr().offset(1usize), MemoryScope::Gpu)];
        for (k, (lock, count, scope)) in locks.into_iter().enumerate() {
            let acquire = AtomicOptions::new(MemoryOrder::Acquire, scope);
            let release = AtomicOptions::new(MemoryOrder::Release, scope);
            if_((t % 32).eq(0), || {
                let (_, done) = while_carry(
                    (Int32::from(0), Boolean::from(false)),
                    |(tries, done)| !done & tries.lt(SPIN_LIMIT),
                    |(tries, done)| {
                        let got = atomic_cas_with(&lock, 0u32, 1u32, acquire).eq(0u32);
                        if_(got, || {
                            count.store(count.load() + 1u32);
                            let _ = atomic_exch_with(&lock, 0u32, release);
                        });
                        (tries + 1, done | got)
                    },
                );
                out.store(warp + (2 + k * N / 32) as i32, select(done, 1u32, 0u32));
            });
        }
        sync_threads();
        if_(t.eq(0), || {
            out.store(0, count.load());
            out.store(1, g.load(1));
        });
    });
    fn check(args: &[KernelArg]) {
        let out = args[1].to_vec::<u32>();
        let warps = (N / 32) as u32;
        assert_eq!(out[..2], [warps, warps], "the counters");
        assert!(out[2..].iter().all(|&d| d == 1), "every warp got the locks: {out:?}");
        assert_eq!(args[0].to_vec::<u32>(), [0, warps], "the global lock and counter");
    }
    Case { name: "atomic_lock", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// Float32 bit patterns for the atomic adds: zeros, subnormals (the
/// smallest, the largest, one in between), the smallest normals, ones,
/// infinities, the canonical NaN and a quiet one, the largest finite value
/// and `2^-24`.
pub const ADD_F32_SPECIALS: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fff_ffff,
    0x7f7f_ffff,
    0x3380_0000,
    0x0040_0000,
    0x8040_0000,
];

/// Float64 bit patterns for the atomic adds: zeros, the smallest and the
/// largest subnormal, ones, infinities, `2^-53`, `1 + 2^-52`, the default
/// NaN.
pub const ADD_F64_SPECIALS: [u64; 11] = [
    0,
    0x8000_0000_0000_0000,
    1,
    0x000f_ffff_ffff_ffff,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x3ca0_0000_0000_0000,
    0x3ff0_0000_0000_0001,
    0x7ff8_0000_0000_0000,
];

/// Pairs of table entries per thread in [`atomic_float_adds`].
const ADD_ROUNDS: usize = 4;

/// Columns of [`atomic_float_adds`]'s 32-bit output: per round, the old and
/// the new Float32 on global and on shared memory, the old and new Float16
/// on each, and the old and new packed pair on each; then the same on the
/// input data (Float32 and Float16).
const ADD_COLS: usize = (ADD_ROUNDS + 1) * 12;

/// Entry `i` and `j` of an `n`-entry table for round `r` of thread `t`.
pub fn add_pair(n: usize, t: usize, r: usize) -> (usize, usize) {
    let idx = (t + N * r) % (n * n);
    (idx / n, idx % n)
}

/// Float32 `a + b` as `atom.add.f32` computes it: rounded to nearest even,
/// subnormal inputs and results flushed to zeros of their signs on global
/// memory (`ftz`), a NaN result the canonical NaN.
pub fn atomic_add_f32_ref(a: u32, b: u32, ftz: bool) -> u32 {
    let flush = |x: u32| if ftz && x & 0x7f80_0000 == 0 { x & 0x8000_0000 } else { x };
    let r = flush((f32::from_bits(flush(a)) + f32::from_bits(flush(b))).to_bits());
    if f32::from_bits(r).is_nan() { 0x7fff_ffff } else { r }
}

/// Float64 `a + b` as `atom.add.f64` computes it: IEEE, a NaN operand
/// quieted (the first, memory's, if both are), an invalid sum the default
/// NaN `0xfff8000000000000`.
pub fn atomic_add_f64_ref(a: u64, b: u64) -> u64 {
    let (x, y) = (f64::from_bits(a), f64::from_bits(b));
    if x.is_nan() {
        return a | 1 << 51;
    }
    if y.is_nan() {
        return b | 1 << 51;
    }
    let r = x + y;
    if r.is_nan() { 0xfff8_0000_0000_0000 } else { r.to_bits() }
}

/// Float16 `a + b` as `atom.add.noftz.f16` computes it: rounded once to
/// nearest even, subnormals kept, a NaN result the canonical NaN. (The sum
/// is rounded to f32 first, which cannot move the rounding to f16: it is
/// exact unless the operands are so far apart that the smaller is far below
/// half a Float16 ulp of the larger.)
pub fn atomic_add_f16_ref(a: u16, b: u16) -> u16 {
    let r = F16::from_f32(F16(a).to_f32() + F16(b).to_f32());
    if r.to_f32().is_nan() { 0x7fff } else { r.0 }
}

/// BFloat16 `a + b` as `atom.add.noftz.bf16` computes it, as
/// [`atomic_add_f16_ref`] does Float16.
pub fn atomic_add_bf16_ref(a: u16, b: u16) -> u16 {
    let r = BF16::from_f32(BF16(a).to_f32() + BF16(b).to_f32());
    if r.to_f32().is_nan() { 0x7fff } else { r.0 }
}

/// The two halves of a packed add, each on its own.
pub fn packed_add_ref(a: u32, b: u32, half: fn(u16, u16) -> u16) -> u32 {
    half(a as u16, b as u16) as u32 | (half((a >> 16) as u16, (b >> 16) as u16) as u32) << 16
}

/// A table of 32-bit patterns in local memory.
fn table32(values: &[u32]) -> Tensor<u32> {
    let table = make_rmem_tensor::<u32>(Layout::col_major(values.len()));
    for (i, &v) in values.iter().enumerate() {
        table.store(i, v);
    }
    table
}

/// The packed pair of round `r` of thread `t` from [`super::F16_SPECIALS`]
/// (or the BFloat16 table): `(cell, operand)`.
pub fn packed_pair(table: &[u16], t: usize, r: usize) -> (u32, u32) {
    let n = table.len();
    let (i, j) = add_pair(n, t, r);
    let w = |lo: usize, hi: usize| table[lo % n] as u32 | (table[hi % n] as u32) << 16;
    (w(i, j), w(j, i + 3))
}

/// The atomic float adds, one thread per address so the results are exact:
/// Float32 on global memory (which flushes subnormals) and on shared memory
/// (which keeps them), Float16 and packed Float16 pairs on both, each on
/// pairs of table entries and on the input data, and Float64 on both; then
/// every thread adding small integers (exact sums) into one Float32,
/// Float64, Float16 and packed cell of each memory.
pub fn atomic_float_adds() -> Case {
    use super::F16_SPECIALS;
    let kernel = Kernel::builder("atomic_float_adds").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(N));
        let ys = param_tensor::<f32>("ys", Layout::col_major(N));
        let hs = param_tensor::<F16>("hs", Layout::col_major(2 * N));
        let gf = param_tensor::<f32>("gf", Layout::col_major(N + 1));
        let gh = param_tensor::<F16>("gh", Layout::col_major(2 * N + 2));
        let gd = param_tensor::<f64>("gd", Layout::col_major(N + 1));
        let out = param_tensor::<u32>("out", Layout::row_major((N, ADD_COLS)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((N, 2 * ADD_ROUNDS * 2)));
        let sums = param_tensor::<u32>("sums", Layout::col_major(8));
        let sf = make_smem_tensor::<f32>(Layout::col_major(N + 1));
        let sh = make_smem_tensor::<F16>(Layout::col_major(2 * N + 2));
        let sd = make_smem_tensor::<f64>(Layout::col_major(N + 1));
        let t = thread_idx().x;
        let f32s = table32(&ADD_F32_SPECIALS);
        let halves: Vec<u32> = F16_SPECIALS.iter().map(|&h| h as u32).collect();
        let f16s = table32(&halves);
        let f64s = make_rmem_tensor::<u64>(Layout::col_major(ADD_F64_SPECIALS.len()));
        for (i, &v) in ADD_F64_SPECIALS.iter().enumerate() {
            f64s.store(i, v);
        }
        if_(t.eq(0), || {
            for (f, h, d) in [(&gf, &gh, &gd), (&sf, &sh, &sd)] {
                f.store(N, 0.0f32);
                h.store(2 * N, F16::ZERO);
                h.store(2 * N + 1, F16::ZERO);
                d.store(N, 0.0f64);
            }
        });
        let col = |c: usize| (t, c);
        let pair = |n: usize, r: usize| {
            let idx = (t + (N * r) as i32) % (n * n) as i32;
            (idx / n as i32, idx % n as i32)
        };
        let word = |table: &Tensor<u32>, n: usize, lo: Int32, hi: Int32| {
            (table.load(lo % n as i32) | (table.load(hi % n as i32) << 16u32)).bitcast::<i32>()
        };
        let half = |table: &Tensor<u32>, i: Int32| table.load(i).to::<u16>().bitcast::<F16>();
        let bits16 = |v: Float16| v.bitcast::<u16>().to::<u32>();
        // One round: Float32 `a + b`, Float16 `ha + hb` and the packed `pa + pb`
        // on each memory, old and new values into columns `c..c + 12`.
        let round = |c: usize, a: Float32, b: Float32, ha: Float16, hb: Float16, pa: Int32, pb: Int32| {
            for (m, (f, h)) in [(&gf, &gh), (&sf, &sh)].into_iter().enumerate() {
                let cell = f.ptr().offset(t);
                cell.store(a);
                let old = atomic_add(&cell, b);
                out.store(col(c + 2 * m), old.bitcast::<u32>());
                out.store(col(c + 2 * m + 1), cell.load().bitcast::<u32>());
                // Float16 cells 2t (scalar) and the pair at 2t, 2t + 1 after it.
                let hcell = h.ptr().offset(IndexExpr::from(t) * 2);
                hcell.store(ha);
                let old = atomic_add(&hcell, hb);
                out.store(col(c + 4 + 2 * m), bits16(old));
                out.store(col(c + 5 + 2 * m), bits16(hcell.load()));
                hcell.cast::<i32>().store(pa);
                let old = atomic_add_packed::<F16, Int32>(&hcell, pb);
                out.store(col(c + 8 + 2 * m), old.bitcast::<u32>());
                out.store(col(c + 9 + 2 * m), hcell.cast::<u32>().load());
            }
        };
        for r in 0..ADD_ROUNDS {
            let (i, j) = pair(ADD_F32_SPECIALS.len(), r);
            let (a, b) = (f32s.load(i).bitcast::<f32>(), f32s.load(j).bitcast::<f32>());
            let (hi, hj) = pair(F16_SPECIALS.len(), r);
            let n = F16_SPECIALS.len();
            let (pa, pb) = (word(&f16s, n, hi, hj), word(&f16s, n, hj, hi + 3));
            round(12 * r, a, b, half(&f16s, hi), half(&f16s, hj), pa, pb);
            // Float64.
            let (i, j) = pair(ADD_F64_SPECIALS.len(), r);
            for (m, d) in [&gd, &sd].into_iter().enumerate() {
                let cell = d.ptr().offset(t);
                cell.store(f64s.load(i).bitcast::<f64>());
                let old = atomic_add(&cell, f64s.load(j).bitcast::<f64>());
                out64.store((t, 4 * r + 2 * m), old.bitcast::<u64>());
                out64.store((t, 4 * r + 2 * m + 1), cell.load().bitcast::<u64>());
            }
        }
        // The input data, as values.
        let (ha, hb) = (hs.load(t * 2), hs.load(t * 2 + 1));
        let pa = hs.ptr().offset(IndexExpr::from(t) * 2).cast::<i32>().load();
        let pb = pa ^ 0x8000_8000u32 as i32;
        round(12 * ADD_ROUNDS, xs.load(t), ys.load(t), ha, hb, pa, pb);
        // Every thread on one cell of each kind.
        sync_threads();
        let small = (t % 8).to::<f32>();
        for (f, h, d) in [(&gf, &gh, &gd), (&sf, &sh, &sd)] {
            let _ = atomic_add(&f.ptr().offset(N), small);
            let _ = atomic_add(&d.ptr().offset(N), small.to::<f64>());
            let _ = atomic_add(&h.ptr().offset(2 * N), small.to::<F16>());
            let pair = (Float16::from(F16::ONE), small.to::<F16>());
            let _ = atomic_add_packed(&h.ptr().offset(2 * N), pair);
            let _ = atomic_add(&h.ptr().offset(2 * N + 1), small.to::<F16>());
        }
        sync_threads();
        if_(t.eq(0), || {
            for (m, (f, h, d)) in [(&gf, &gh, &gd), (&sf, &sh, &sd)].into_iter().enumerate() {
                sums.store(4 * m, f.load(N).bitcast::<u32>());
                sums.store(4 * m + 1, d.load(N).to::<f32>().bitcast::<u32>());
                sums.store(4 * m + 2, bits16(h.load(2 * N)));
                sums.store(4 * m + 3, bits16(h.load(2 * N + 1)));
            }
        });
    });
    fn check(args: &[KernelArg]) {
        use super::F16_SPECIALS;
        let (xs, ys, hs) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<F16>());
        let out = args[6].to_vec::<u32>();
        let out64 = args[7].to_vec::<u64>();
        let sums = args[8].to_vec::<u32>();
        let f16 = atomic_add_f16_ref as fn(u16, u16) -> u16;
        for t in 0..N {
            let row = &out[t * ADD_COLS..(t + 1) * ADD_COLS];
            let mut cases: Vec<(u32, u32, u16, u16, u32, u32)> = (0..ADD_ROUNDS)
                .map(|r| {
                    let (i, j) = add_pair(ADD_F32_SPECIALS.len(), t, r);
                    let (hi, hj) = add_pair(F16_SPECIALS.len(), t, r);
                    let (pa, pb) = packed_pair(&F16_SPECIALS, t, r);
                    (ADD_F32_SPECIALS[i], ADD_F32_SPECIALS[j], F16_SPECIALS[hi], F16_SPECIALS[hj], pa, pb)
                })
                .collect();
            let (ha, hb) = (hs[2 * t].0, hs[2 * t + 1].0);
            let pa = ha as u32 | (hb as u32) << 16;
            cases.push((xs[t].to_bits(), ys[t].to_bits(), ha, hb, pa, pa ^ 0x8000_8000));
            for (r, &(a, b, ha, hb, pa, pb)) in cases.iter().enumerate() {
                let c = 12 * r;
                for (m, ftz) in [(0, true), (1, false)] {
                    let what = format!("thread {t}, round {r}, {}", ["global", "shared"][m]);
                    assert_eq!(
                        row[c + 2 * m..c + 2 * m + 2],
                        [a, atomic_add_f32_ref(a, b, ftz)],
                        "{a:#x} + {b:#x}, {what}"
                    );
                    assert_eq!(
                        row[c + 4 + 2 * m..c + 6 + 2 * m],
                        [ha as u32, f16(ha, hb) as u32],
                        "f16 {ha:#x} + {hb:#x}, {what}"
                    );
                    assert_eq!(
                        row[c + 8 + 2 * m..c + 10 + 2 * m],
                        [pa, packed_add_ref(pa, pb, f16)],
                        "f16x2 {pa:#x} + {pb:#x}, {what}"
                    );
                }
            }
            for r in 0..ADD_ROUNDS {
                let (i, j) = add_pair(ADD_F64_SPECIALS.len(), t, r);
                let (a, b) = (ADD_F64_SPECIALS[i], ADD_F64_SPECIALS[j]);
                for m in 0..2 {
                    let k = t * 4 * ADD_ROUNDS + 4 * r + 2 * m;
                    assert_eq!(out64[k..k + 2], [a, atomic_add_f64_ref(a, b)], "f64 {a:#x} + {b:#x}, thread {t}");
                }
            }
        }
        let f = (0..N as u32).map(|t| t % 8).sum::<u32>() as f32;
        let h = |x: f32| F16::from_f32(x).0 as u32;
        // Cell 2N gets the scalar adds and the packed pairs' 1s, cell 2N + 1
        // the scalar adds and the pairs' high halves.
        let want = [f.to_bits(), f.to_bits(), h(f + N as f32), h(2.0 * f)];
        assert_eq!(sums[..4], want, "global sums");
        assert_eq!(sums[4..], want, "shared sums");
    }
    Case { name: "atomic_float_adds", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// Float32 patterns for `atomic_fmax` and `atomic_fmin`: zeros of both
/// signs, subnormals, ones, the largest finite values, infinities, the
/// canonical NaN and another NaN with the sign bit clear.
pub const EXTREMA_SPECIALS: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x3f80_0000,
    0xbf80_0000,
    0x4020_0000,
    0xc020_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fff_ffff,
    0x7fc0_0000,
    0x0040_0000,
    0x8040_0000,
];

/// The order `atomic_fmax` and `atomic_fmin` follow: IEEE order on numbers
/// (`-0 < +0`), with NaNs whose sign bit is clear above `+inf`, by payload.
pub fn extremum_key(bits: u32) -> i64 {
    if bits >> 31 == 0 { bits as i64 } else { -((bits & 0x7fff_ffff) as i64) - 1 }
}

/// The new contents after `atomic_fmax` (`max`) or `atomic_fmin` of
/// `operand` on `old`.
pub fn extremum_ref(max: bool, old: u32, operand: u32) -> u32 {
    let operand_wins =
        if max { extremum_key(operand) > extremum_key(old) } else { extremum_key(operand) < extremum_key(old) };
    if operand_wins { operand } else { old }
}

/// Columns of [`atomic_float_extrema`]: per round, the old value and the new
/// one after `atomic_fmax` and after `atomic_fmin` on each memory; then the
/// forms with the sign known at build time.
const EXTREMA_COLS: usize = ADD_ROUNDS * 8 + 8;

/// `atomic_fmax` and `atomic_fmin` on pairs of [`EXTREMA_SPECIALS`], one
/// thread per address, on global and shared memory; `atomic_fmax_with` and
/// `atomic_fmin_with` with the sign bit given (with `sem=` and `scope=`),
/// and the deprecated `atomic_max_float32`; and every thread's input value
/// into one cell of each memory.
// Tests the deprecated names, which still have to work.
#[allow(deprecated)]
pub fn atomic_float_extrema() -> Case {
    let kernel = Kernel::builder("atomic_float_extrema").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(N));
        let g = param_tensor::<f32>("g", Layout::col_major(N + 2));
        let out = param_tensor::<u32>("out", Layout::row_major((N, EXTREMA_COLS)));
        let ends = param_tensor::<u32>("ends", Layout::col_major(4));
        let s = make_smem_tensor::<f32>(Layout::col_major(N + 2));
        let t = thread_idx().x;
        let table = table32(&EXTREMA_SPECIALS);
        let n = EXTREMA_SPECIALS.len() as i32;
        if_(t.eq(0), || {
            for m in [&g, &s] {
                m.store(N, f32::NEG_INFINITY);
                m.store(N + 1, f32::INFINITY);
            }
        });
        let col = |c: usize| (t, c);
        for r in 0..ADD_ROUNDS {
            let idx = (t + (N * r) as i32) % (n * n);
            let (a, b) = (table.load(idx / n).bitcast::<f32>(), table.load(idx % n).bitcast::<f32>());
            for (m, mem) in [&g, &s].into_iter().enumerate() {
                let cell = mem.ptr().offset(t);
                let c = 8 * r + 4 * m;
                cell.store(a);
                out.store(col(c), atomic_fmax(&cell, b).bitcast::<u32>());
                out.store(col(c + 1), cell.load().bitcast::<u32>());
                cell.store(a);
                out.store(col(c + 2), atomic_fmin(&cell, b).bitcast::<u32>());
                out.store(col(c + 3), cell.load().bitcast::<u32>());
            }
        }
        let a = table.load(t % n).bitcast::<f32>();
        let opts = AtomicOptions::new(MemoryOrder::AcqRel, MemoryScope::Cta);
        for (m, mem) in [&g, &s].into_iter().enumerate() {
            let cell = mem.ptr().offset(t);
            let c = 8 * ADD_ROUNDS + 4 * m;
            cell.store(a);
            let _ = atomic_fmax_with(&cell, 3.0f32, Some(false), opts);
            let _ = atomic_fmin_with(&cell, -3.0f32, Some(true), AtomicOptions::default());
            out.store(col(c), cell.load().bitcast::<u32>());
            cell.store(a);
            let _ = atomic_max_float32(&cell, 2.0f32, true);
            out.store(col(c + 1), cell.load().bitcast::<u32>());
            cell.store(a);
            let _ = atomic_max_float32(&cell, -2.0f32, false);
            out.store(col(c + 2), cell.load().bitcast::<u32>());
            cell.store(a);
            let _ = atomic_fmin_with(&cell, 0.5f32, Some(false), opts);
            out.store(col(c + 3), cell.load().bitcast::<u32>());
        }
        sync_threads();
        let x = xs.load(t);
        for mem in [&g, &s] {
            let _ = atomic_fmax(&mem.ptr().offset(N), x);
            let _ = atomic_fmin(&mem.ptr().offset(N + 1), x);
        }
        sync_threads();
        if_(t.eq(0), || {
            for (m, mem) in [&g, &s].into_iter().enumerate() {
                ends.store(2 * m, mem.load(N).bitcast::<u32>());
                ends.store(2 * m + 1, mem.load(N + 1).bitcast::<u32>());
            }
        });
    });
    fn check(args: &[KernelArg]) {
        let xs = args[0].to_vec::<f32>();
        let out = args[2].to_vec::<u32>();
        let n = EXTREMA_SPECIALS.len();
        for t in 0..N {
            let row = &out[t * EXTREMA_COLS..(t + 1) * EXTREMA_COLS];
            for r in 0..ADD_ROUNDS {
                let (i, j) = add_pair(n, t, r);
                let (a, b) = (EXTREMA_SPECIALS[i], EXTREMA_SPECIALS[j]);
                for m in 0..2 {
                    let c = 8 * r + 4 * m;
                    let want = [a, extremum_ref(true, a, b), a, extremum_ref(false, a, b)];
                    assert_eq!(row[c..c + 4], want, "fmax/fmin of {a:#x} and {b:#x}, thread {t}, memory {m}");
                }
            }
            let a = EXTREMA_SPECIALS[t % n];
            let three = 3.0f32.to_bits();
            let want = [
                extremum_ref(false, extremum_ref(true, a, three), (-3.0f32).to_bits()),
                extremum_ref(true, a, 2.0f32.to_bits()),
                extremum_ref(true, a, (-2.0f32).to_bits()),
                extremum_ref(false, a, 0.5f32.to_bits()),
            ];
            for m in 0..2 {
                let c = 8 * ADD_ROUNDS + 4 * m;
                assert_eq!(row[c..c + 4], want, "sign_bit forms on {a:#x}, thread {t}, memory {m}");
            }
        }
        let bits: Vec<u32> = xs.iter().map(|x| x.to_bits()).collect();
        let max = bits.iter().fold(f32::NEG_INFINITY.to_bits(), |a, &b| extremum_ref(true, a, b));
        let min = bits.iter().fold(f32::INFINITY.to_bits(), |a, &b| extremum_ref(false, a, b));
        assert_eq!(args[3].to_vec::<u32>(), [max, min, max, min], "every thread's value");
    }
    Case { name: "atomic_float_extrema", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// The reductions of [`reductions`]: each operation and dtype `red` has on
/// sm_89, with its `sem=` and `scope=`, as `(name, dtype, op)`.
pub fn reduction_list() -> Vec<(&'static str, AtomicType, RedOp)> {
    use AtomicType::*;
    vec![
        ("add.u32", U32, RedOp::Add),
        ("add.s32", S32, RedOp::Add),
        ("min.u32", U32, RedOp::Min),
        ("max.u32", U32, RedOp::Max),
        ("min.s32", S32, RedOp::Min),
        ("max.s32", S32, RedOp::Max),
        ("umin on s32", S32, RedOp::Umin),
        ("umax on s32", S32, RedOp::Umax),
        ("and.b32", B32, RedOp::And),
        ("or.b32", B32, RedOp::Or),
        ("xor.b32", B32, RedOp::Xor),
        ("add.u64", U64, RedOp::Add),
        ("add.s64", S64, RedOp::Add),
        ("min.u64", U64, RedOp::Min),
        ("max.s64", S64, RedOp::Max),
        ("umin on s64", S64, RedOp::Umin),
        ("and.b64", B64, RedOp::And),
        ("or.b64", B64, RedOp::Or),
        ("xor.b64", B64, RedOp::Xor),
    ]
}

/// Every integer reduction of [`reduction_list`], 32- and 64-bit, on global
/// and shared memory, relaxed and release at each scope in turn, from every
/// thread of a block; and the float adds (Float32, Float64, Float16 and
/// packed Float16) on small integers, whose sums are exact.
pub fn reductions() -> Case {
    let list = reduction_list();
    let cells = list.len();
    let kernel = Kernel::builder("reductions").block_dim([N as u32, 1, 1]).build(move || {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let ys = param_tensor::<u64>("ys", Layout::col_major(N));
        let g = param_tensor::<u64>("g", Layout::col_major(cells));
        let gfloat = param_tensor::<u32>("gfloat", Layout::col_major(6));
        let out = param_tensor::<u64>("out", Layout::col_major(cells));
        let outf = param_tensor::<u32>("outf", Layout::col_major(6));
        let s = make_smem_tensor::<u64>(Layout::col_major(cells));
        let sfloat = make_smem_tensor::<u32>(Layout::col_major(6));
        let t = thread_idx().x;
        if_(t.lt(cells as i32), || {
            let init = make_rmem_tensor::<u64>(Layout::col_major(cells));
            for (c, &(_, ty, op)) in list.iter().enumerate() {
                init.store(c, reduction_init(ty, op));
            }
            g.store(t, init.load(t));
            s.store(t, init.load(t));
        });
        if_(t.lt(6), || {
            gfloat.store(t, 0u32);
            sfloat.store(t, 0u32);
        });
        sync_threads();
        let (x, y) = (xs.load(t), ys.load(t));
        let scopes = [MemoryScope::Cta, MemoryScope::Gpu, MemoryScope::Sys];
        for (c, &(_, ty, op)) in list.iter().enumerate() {
            let sem = if c % 2 == 0 { MemoryOrder::Relaxed } else { MemoryOrder::Release };
            let opts = if c == 0 { AtomicOptions::default() } else { AtomicOptions::new(sem, scopes[c % 3]) };
            for cells in [g.ptr(), s.ptr()] {
                let cell = cells.offset(c);
                match (ty.bits(), ty.is_signed()) {
                    (32, false) => red(&cell.cast::<u32>(), x, op, ty, opts),
                    (32, true) => red(&cell.cast::<i32>(), x.bitcast::<i32>(), op, ty, opts),
                    (_, false) => red(&cell, y, op, ty, opts),
                    (_, true) => red(&cell.cast::<i64>(), y.bitcast::<i64>(), op, ty, opts),
                }
            }
        }
        let small = (t % 8).to::<f32>();
        let release = AtomicOptions::new(MemoryOrder::Release, MemoryScope::Gpu);
        for cells in [gfloat.ptr(), sfloat.ptr()] {
            red(&cells.cast::<f32>(), small, RedOp::Add, AtomicType::F32, release);
            red(
                &cells.offset(2usize).cast::<f64>(),
                small.to::<f64>(),
                RedOp::Add,
                DType::F64,
                AtomicOptions::default(),
            );
            let halves = cells.offset(4usize).cast::<F16>();
            red(&halves, small.to::<F16>(), RedOp::Add, AtomicType::F16, AtomicOptions::default());
            red_packed(&halves.offset(2usize), (small.to::<F16>(), Float16::from(F16::ONE)), RedOp::Add, release);
        }
        sync_threads();
        if_(t.lt(cells as i32), || out.store(t, s.load(t)));
        if_(t.lt(6), || outf.store(t, sfloat.load(t)));
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>());
        for (c, &(name, ty, op)) in reduction_list().iter().enumerate() {
            let vals: Vec<u64> = if ty.bits() == 32 { xs.iter().map(|&x| x as u64).collect() } else { ys.clone() };
            let want = vals.iter().fold(reduction_init(ty, op), |a, &v| reduction_ref(ty, op, a, v));
            assert_eq!(args[2].to_vec::<u64>()[c], want, "global {name}");
            assert_eq!(args[4].to_vec::<u64>()[c], want, "shared {name}");
        }
        let total = (0..N as u32).map(|t| t % 8).sum::<u32>() as f32;
        let f64_bits = (total as f64).to_bits();
        let halves = F16::from_f32(total).0 as u32 | (F16::from_f32(N as f32).0 as u32) << 16;
        let want =
            [total.to_bits(), 0, f64_bits as u32, (f64_bits >> 32) as u32, F16::from_f32(total).0 as u32, halves];
        assert_eq!(args[3].to_vec::<u32>(), want, "global float sums");
        assert_eq!(args[5].to_vec::<u32>(), want, "shared float sums");
    }
    Case { name: "reductions", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// Where a cell of [`reductions`] starts: the identity of the operation (as
/// the dtype's width and signedness see it; `Umin` and `Umax` compare
/// unsigned).
pub fn reduction_init(ty: AtomicType, op: RedOp) -> u64 {
    let ones = if ty.bits() == 32 { u32::MAX as u64 } else { u64::MAX };
    match op {
        RedOp::And | RedOp::Umin => ones,
        RedOp::Min if ty.is_signed() => ones >> 1,
        RedOp::Min => ones,
        RedOp::Max if ty.is_signed() => (ones >> 1) + 1,
        _ => 0,
    }
}

/// `red` with `op` on dtype `ty` of `v` into `a` (both as unsigned bits of
/// the dtype's width).
pub fn reduction_ref(ty: AtomicType, op: RedOp, a: u64, v: u64) -> u64 {
    let wide = ty.bits() == 64;
    let mask = if wide { u64::MAX } else { u32::MAX as u64 };
    let signed = |x: u64| if wide { x as i64 } else { x as u32 as i32 as i64 };
    let r = match op {
        RedOp::Add => a.wrapping_add(v),
        RedOp::And => a & v,
        RedOp::Or => a | v,
        RedOp::Xor => a ^ v,
        RedOp::Min if ty.is_signed() => {
            if signed(v) < signed(a) {
                v
            } else {
                a
            }
        }
        RedOp::Max if ty.is_signed() => {
            if signed(v) > signed(a) {
                v
            } else {
                a
            }
        }
        RedOp::Min | RedOp::Umin => a.min(v),
        RedOp::Max | RedOp::Umax => a.max(v),
    };
    r & mask
}

/// What needs sm_90: the BFloat16 atomic adds (scalar and packed, one
/// thread per address on pairs of [`super::BF16_SPECIALS`], and every
/// thread on one cell), the BFloat16 reductions, and atomics and reductions
/// with cluster scope, on global and shared memory.
pub fn atomics_sm90() -> Case {
    use super::BF16_SPECIALS;
    let kernel = Kernel::builder("atomics_sm90").block_dim([N as u32, 1, 1]).build(|| {
        let gh = param_tensor::<BF16>("gh", Layout::col_major(2 * N + 4));
        let gc = param_tensor::<u32>("gc", Layout::col_major(2));
        let out = param_tensor::<u32>("out", Layout::row_major((N, ADD_ROUNDS * 8)));
        let sums = param_tensor::<u32>("sums", Layout::col_major(12));
        let sh = make_smem_tensor::<BF16>(Layout::col_major(2 * N + 4));
        let sc = make_smem_tensor::<u32>(Layout::col_major(2));
        let t = thread_idx().x;
        let halves: Vec<u32> = BF16_SPECIALS.iter().map(|&h| h as u32).collect();
        let table = table32(&halves);
        let n = BF16_SPECIALS.len() as i32;
        let half = |i: Int32| table.load(i % n).to::<u16>().bitcast::<BF16>();
        let bits16 = |v: BFloat16| v.bitcast::<u16>().to::<u32>();
        if_(t.lt(4), || {
            gh.store(2 * N as i32 + t, BF16::default());
            sh.store(2 * N as i32 + t, BF16::default());
        });
        if_(t.lt(2), || {
            gc.store(t, 0u32);
            sc.store(t, 0u32);
        });
        sync_threads();
        for r in 0..ADD_ROUNDS {
            let idx = (t + (N * r) as i32) % (n * n);
            let (i, j) = (idx / n, idx % n);
            let pa = (table.load(i % n) | (table.load(j % n) << 16u32)).bitcast::<i32>();
            let pb = (table.load(j % n) | (table.load((i + 3) % n) << 16u32)).bitcast::<i32>();
            for (m, mem) in [&gh, &sh].into_iter().enumerate() {
                let cell = mem.ptr().offset(IndexExpr::from(t) * 2);
                let c = 8 * r + 4 * m;
                cell.store(half(i));
                let old = atomic_add(&cell, half(j));
                out.store((t, c), bits16(old));
                out.store((t, c + 1), bits16(cell.load()));
                cell.cast::<i32>().store(pa);
                let old = atomic_add_packed::<BF16, Int32>(&cell, pb);
                out.store((t, c + 2), old.bitcast::<u32>());
                out.store((t, c + 3), cell.cast::<u32>().load());
            }
        }
        sync_threads();
        // Sums stay at most 256, which BFloat16 holds exactly in any order.
        let small = (t % 2).to::<f32>().to::<BF16>();
        let one = BFloat16::from(BF16::from_f32(1.0));
        let cluster = AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Cluster);
        let release = AtomicOptions::new(MemoryOrder::Release, MemoryScope::Cluster);
        let acq_rel = AtomicOptions::new(MemoryOrder::AcqRel, MemoryScope::Cluster);
        for (mem, counts) in [(&gh, &gc), (&sh, &sc)] {
            let tail = mem.ptr().offset(2 * N);
            let _ = atomic_add_with(&tail, small, cluster);
            red(&tail.offset(1usize), small, RedOp::Add, AtomicType::Bf16, release);
            let _ = atomic_add_packed_with(&tail.offset(2usize), (small, one), cluster);
            red_packed(&tail.offset(2usize), (one, small), RedOp::Add, release);
            let _ = atomic_add_with(&counts.ptr(), 1u32, acq_rel);
            red(&counts.ptr().offset(1usize), 2u32, RedOp::Add, AtomicType::U32, release);
        }
        sync_threads();
        if_(t.eq(0), || {
            for (m, (mem, counts)) in [(&gh, &gc), (&sh, &sc)].into_iter().enumerate() {
                for k in 0..4 {
                    sums.store(6 * m + k, bits16(mem.load(2 * N + k)));
                }
                sums.store(6 * m + 4, counts.load(0));
                sums.store(6 * m + 5, counts.load(1));
            }
        });
    });
    fn check(args: &[KernelArg]) {
        let out = args[2].to_vec::<u32>();
        let bf16 = atomic_add_bf16_ref as fn(u16, u16) -> u16;
        let cols = ADD_ROUNDS * 8;
        let n = BF16_SPECIALS.len();
        for t in 0..N {
            let row = &out[t * cols..(t + 1) * cols];
            for r in 0..ADD_ROUNDS {
                let (i, j) = add_pair(n, t, r);
                let (a, b) = (BF16_SPECIALS[i], BF16_SPECIALS[j]);
                let (pa, pb) = packed_pair(&BF16_SPECIALS, t, r);
                for m in 0..2 {
                    let c = 8 * r + 4 * m;
                    let want = [a as u32, bf16(a, b) as u32, pa, packed_add_ref(pa, pb, bf16)];
                    assert_eq!(row[c..c + 4], want, "bf16 {a:#x} + {b:#x}, {pa:#x} + {pb:#x}, thread {t}, memory {m}");
                }
            }
        }
        let total = (0..N as u32).map(|t| t % 2).sum::<u32>() as f32;
        let bits = |x: f32| BF16::from_f32(x).0 as u32;
        // The packed cells get (small, 1) from the atomic and (1, small) from the red.
        let both = bits(total + N as f32);
        let want = [bits(total), bits(total), both, both, N as u32, 2 * N as u32];
        assert_eq!(args[3].to_vec::<u32>(), [want, want].concat(), "sums");
    }
    Case { name: "atomics_sm90", kernel, cfg: launch_n(1, N), small_only: false, check }
}

//! The pipeline primitives (`cute::pipeline`: states, mbarrier arrays,
//! init sync) and shared-memory partitions (`cute::memory`), held to plain
//! Rust. The kernels that wait on mbarriers are built for sm_89, so that
//! their waits spin on `mbarrier.test_wait` and the SM89 GPU runs them; the
//! cluster and TMA ones are sm_90.

use cute::memory::{SmemAllocator, SmemPartition, calculate_partition_size, calculate_total_usage};
use cute::pipeline::{
    Agent, CooperativeGroup, MbarrierArray, PerStagePipelineState, PipelineOp, PipelineState, PipelineUserType,
    agent_sync, alloc_reserved_mbarrier, compute_k_tile, make_pipeline_state, pipeline_init_arrive, pipeline_init_wait,
};
use cute::prelude::*;

use super::Case;

/// Stages of [`pipeline_state_walk`]'s ring.
const PS_STAGES: usize = 5;
/// Steps of each thread of [`pipeline_state_walk`].
const PS_STEPS: usize = 24;
/// Threads of the walks.
const WALK_THREADS: usize = 32;

/// A `PipelineState` of [`PS_STAGES`] stages per thread, from
/// `make_pipeline_state(Producer)`, through a runtime loop of steps that
/// the thread's operands pick: `advance`, `advance_by(3)`, `advance_by` a
/// runtime count of 1 to 9, `reverse` and `reset_count`. Every step writes
/// the index, count and phase.
pub fn pipeline_state_walk() -> Case {
    let kernel = Kernel::builder("pipeline_state_walk").block_dim([WALK_THREADS as u32, 1, 1]).build(|| {
        let ops = param_tensor::<u32>("ops", Layout::col_major(WALK_THREADS * PS_STEPS));
        let out = param_tensor::<i32>("out", Layout::row_major((WALK_THREADS * PS_STEPS, 3)));
        let t = thread_idx().x;
        let start = make_pipeline_state(PipelineUserType::Producer, PS_STAGES);
        for_range_carry(0, PS_STEPS as i32, 1, start, |k, s| {
            let row = t * PS_STEPS as i32 + k;
            let op = ops.load(row);
            let code = (op % 5u32).to::<i32>();
            let amount = ((op / 5u32) % 9u32).to::<i32>() + 1;
            let mut moved = [s; 5];
            moved[0].advance();
            moved[1].advance_by(3);
            moved[2].advance_by(amount);
            moved[3].reverse();
            moved[4].reset_count();
            let pick = |field: fn(&PipelineState) -> Int32| {
                (0..4).rev().fold(field(&moved[4]), |acc, c| select(code.eq(c), field(&moved[c as usize]), acc))
            };
            let next = PipelineState::new(PS_STAGES, pick(|s| s.count()), pick(|s| s.index()), pick(|s| s.phase()));
            out.store((row, 0), next.index());
            out.store((row, 1), next.count());
            out.store((row, 2), next.phase());
            next
        });
    });
    fn check(args: &[KernelArg]) {
        let (ops, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        let n = PS_STAGES as i32;
        for t in 0..WALK_THREADS {
            let (mut count, mut index, mut phase) = (0i32, 0i32, 1i32);
            for k in 0..PS_STEPS {
                let row = t * PS_STEPS + k;
                let op = ops[row];
                let by = |count: &mut i32, index: &mut i32, phase: &mut i32, m: i32| {
                    *count += m;
                    *index += m;
                    *phase ^= (*index / n) % 2;
                    *index %= n;
                };
                match op % 5 {
                    0 => {
                        count += 1;
                        index += 1;
                        if index == n {
                            index = 0;
                            phase ^= 1;
                        }
                    }
                    1 => by(&mut count, &mut index, &mut phase, 3),
                    2 => by(&mut count, &mut index, &mut phase, ((op / 5) % 9) as i32 + 1),
                    3 => {
                        count -= 1;
                        index -= 1;
                        if index == -1 {
                            index = n - 1;
                            phase ^= 1;
                        }
                    }
                    _ => count = 0,
                }
                assert_eq!(out[row * 3..row * 3 + 3], [index, count, phase], "thread {t}, step {k} (op {op})");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [WALK_THREADS as u32, 1, 1]);
    Case { name: "pipeline_state_walk", kernel, cfg, small_only: false, check }
}

/// Stages of [`per_stage_state_walk`].
const PSS_STAGES: usize = 3;
/// Steps of each thread of [`per_stage_state_walk`].
const PSS_STEPS: usize = 30;

/// A `PerStagePipelineState` of [`PSS_STAGES`] stages per thread through a
/// runtime loop of `next`, `mark_valid`, `mark_invalid` and `reset_count`,
/// picked by the thread's operands, over several rounds of the ring. Every
/// step writes the index, phase and count of the current stage, and its k
/// tile (`compute_k_tile` with k tile 0 in stage `t % 3`).
pub fn per_stage_state_walk() -> Case {
    let kernel = Kernel::builder("per_stage_state_walk").block_dim([WALK_THREADS as u32, 1, 1]).build(|| {
        let ops = param_tensor::<u32>("ops", Layout::col_major(WALK_THREADS * PSS_STEPS));
        let out = param_tensor::<i32>("out", Layout::row_major((WALK_THREADS * PSS_STEPS, 4)));
        let t = thread_idx().x;
        let offset = t % PSS_STAGES as i32;
        let start = PerStagePipelineState::new(PSS_STAGES);
        for_range_carry(0, PSS_STEPS as i32, 1, start, |k, s| {
            let row = t * PSS_STEPS as i32 + k;
            let code = (ops.load(row) % 4u32).to::<i32>();
            if_(code.eq(1), || s.mark_valid());
            if_(code.eq(2), || s.mark_invalid());
            if_(code.eq(3), || s.reset_count());
            let s = if_else_yield(
                code.eq(0),
                || {
                    let mut moved = s.clone();
                    moved.next();
                    moved
                },
                || s.clone(),
            );
            out.store((row, 0), s.index());
            out.store((row, 1), s.phase());
            out.store((row, 2), s.count());
            out.store((row, 3), compute_k_tile(s.index(), s.count(), offset, PSS_STAGES));
            s
        });
    });
    fn check(args: &[KernelArg]) {
        let (ops, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        let n = PSS_STAGES as i32;
        for t in 0..WALK_THREADS {
            let (mut index, mut phases, mut counts) = (0usize, [0i32; PSS_STAGES], [0i32; PSS_STAGES]);
            let offset = (t % PSS_STAGES) as i32;
            for k in 0..PSS_STEPS {
                let row = t * PSS_STEPS + k;
                match ops[row] % 4 {
                    0 => index = (index + 1) % PSS_STAGES,
                    1 => {
                        phases[index] ^= 1;
                        counts[index] += 1;
                    }
                    2 => phases[index] ^= 1,
                    _ => counts = [0; PSS_STAGES],
                }
                let k_tile = (index as i32 - offset).rem_euclid(n) + counts[index] * n;
                let want = [index as i32, phases[index], counts[index], k_tile];
                assert_eq!(out[row * 4..row * 4 + 4], want, "thread {t}, step {k}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [WALK_THREADS as u32, 1, 1]);
    Case { name: "per_stage_state_walk", kernel, cfg, small_only: false, check }
}

/// Stages of [`mbarrier_array_handoff`].
const HO_STAGES: usize = 3;
/// Iterations of [`mbarrier_array_handoff`]: the ring wraps six times.
const HO_ITERS: usize = 20;

/// A 3-stage ring between two warps through a full and an empty
/// `MbarrierArray` (`AsyncThread`, a warp of arrivals each) in reserved
/// storage, indexed by a producer and a consumer `PipelineState`: warp 1
/// waits for an empty stage, fills it and arrives on full; warp 0 waits on
/// full, reads a rotated word and arrives on empty. Built for sm_89, so the
/// waits spin on `mbarrier.test_wait` and the init sync is a CTA barrier.
/// Each warp's lane 0 records the index and phase of every iteration.
pub fn mbarrier_array_handoff() -> Case {
    let kernel = Kernel::builder("mbarrier_array_handoff").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(HO_ITERS * 32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let trace = param_tensor::<i32>("trace", Layout::row_major((2 * HO_ITERS, 2)));
        let lane = lane_idx();
        let storage = alloc_reserved_mbarrier(HO_STAGES, 2);
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let full = MbarrierArray::new(&storage, HO_STAGES, (PipelineOp::AsyncThread, warp), 0);
        let empty = MbarrierArray::new(&storage.offset(HO_STAGES), HO_STAGES, (PipelineOp::AsyncThread, warp), 0);
        let data = make_smem_tensor::<u32>(Layout::col_major(HO_STAGES * 32));
        pipeline_init_arrive(None, false);
        pipeline_init_wait(None);
        let record = |row: Int32, s: &PipelineState| {
            if_(lane.eq(0), || {
                trace.store((row, 0), s.index());
                trace.store((row, 1), s.phase());
            });
        };
        if_else(
            warp_idx().eq(1),
            || {
                let start = make_pipeline_state(PipelineUserType::Producer, HO_STAGES);
                for_range_carry(0, HO_ITERS as i32, 1, start, |k, mut s| {
                    empty.wait(s.index(), s.phase());
                    data.store(s.index() * 32 + lane, src.load(k * 32 + lane) + k.to::<u32>());
                    full.arrive(s.index(), None, None);
                    record(k, &s);
                    s.advance();
                    s
                });
            },
            || {
                let start = (Uint32::from(0u32), make_pipeline_state(PipelineUserType::Consumer, HO_STAGES));
                let (sum, _) = for_range_carry(0, HO_ITERS as i32, 1, start, |k, (sum, mut s)| {
                    full.wait(s.index(), s.phase());
                    let word = data.load(s.index() * 32 + (lane + k) % 32);
                    empty.arrive(s.index(), None, None);
                    record(k + HO_ITERS as i32, &s);
                    s.advance();
                    (sum + word, s)
                });
                out.store(lane, sum);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (src, out, trace) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<i32>());
        for (l, &got) in out.iter().enumerate() {
            let want =
                (0..HO_ITERS).fold(0u32, |acc, k| acc.wrapping_add(src[k * 32 + (l + k) % 32].wrapping_add(k as u32)));
            assert_eq!(got, want, "lane {l}");
        }
        for k in 0..HO_ITERS {
            let (index, wraps) = ((k % HO_STAGES) as i32, (k / HO_STAGES) as i32);
            assert_eq!(trace[k * 2..k * 2 + 2], [index, 1 ^ (wraps & 1)], "producer iteration {k}");
            let c = (HO_ITERS + k) * 2;
            assert_eq!(trace[c..c + 2], [index, wraps & 1], "consumer iteration {k}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    Case { name: "mbarrier_array_handoff", kernel, cfg, small_only: false, check }
}

/// Stages of [`async_load_handoff`].
const AL_STAGES: usize = 2;
/// Tiles of [`async_load_handoff`].
const AL_ITERS: usize = 6;
/// Words per tile of [`async_load_handoff`].
const AL_TILE: usize = 128;

/// A 2-stage ring filled by `cp.async`: warp 0 copies each tile into a
/// stage (16 bytes a thread) and arrives on the full `MbarrierArray` with
/// `PipelineOp::AsyncLoad` (`cp.async.mbarrier.arrive.noinc`, an arrival
/// when its copies land); warp 1 waits on full, writes the stage back to
/// global memory rotated, and arrives on empty. The user buffers come first
/// (`pad_bytes` of them, then the stages), and the mbarriers last, from
/// `alloc_reserved_mbarrier`: the reserved partition still puts them at
/// offset 0. Above 48 KiB of padding, the shared memory lives in the
/// dynamic window. Built for sm_89.
pub fn async_load_handoff(pad_bytes: usize) -> Case {
    let name = if pad_bytes > 48 * 1024 { "async_load_handoff_dynamic" } else { "async_load_handoff" };
    let kernel = Kernel::builder(name).block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(AL_ITERS * AL_TILE));
        let dst = param_tensor::<u32>("dst", Layout::col_major(AL_ITERS * AL_TILE));
        let lane = lane_idx();
        let pad = alloc_smem::<u8>(pad_bytes, 16);
        let data = make_smem_tensor::<u32>(Layout::col_major(AL_STAGES * AL_TILE));
        let storage = alloc_reserved_mbarrier(AL_STAGES, 2);
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let full = MbarrierArray::new(&storage, AL_STAGES, (PipelineOp::AsyncLoad, warp), 0);
        let empty = MbarrierArray::new(&storage.offset(AL_STAGES), AL_STAGES, (PipelineOp::AsyncThread, warp), 0);
        // Touch the padding's last byte, so that it is not an unused buffer.
        if_(thread_idx().x.eq(0), || pad.offset(pad_bytes - 1).store(0u8));
        pipeline_init_arrive(None, false);
        pipeline_init_wait(None);
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let tile = AL_TILE as i32;
        if_else(
            warp_idx().eq(0),
            || {
                let thr = tiled.get_slice(lane);
                let start = make_pipeline_state(PipelineUserType::Producer, AL_STAGES);
                for_range_carry(0, AL_ITERS as i32, 1, start, |k, mut s| {
                    empty.wait(s.index(), s.phase());
                    let from = make_tensor(&src.ptr().offset(k * tile), Layout::col_major(AL_TILE));
                    let to = make_tensor(&data.ptr().offset(s.index() * tile), Layout::col_major(AL_TILE));
                    copy(&tiled, &thr.partition_s(&from), &thr.partition_d(&to));
                    full.arrive(s.index(), None, None);
                    s.advance();
                    s
                });
            },
            || {
                let start = make_pipeline_state(PipelineUserType::Consumer, AL_STAGES);
                for_range_carry(0, AL_ITERS as i32, 1, start, |k, mut s| {
                    full.wait(s.index(), s.phase());
                    for j in 0..4 {
                        let word = data.load(s.index() * tile + (lane * 4 + j + 1) % tile);
                        dst.store(k * tile + lane * 4 + j, word);
                    }
                    empty.arrive(s.index(), None, None);
                    s.advance();
                    s
                });
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (src, dst) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for k in 0..AL_ITERS {
            for i in 0..AL_TILE {
                assert_eq!(dst[k * AL_TILE + i], src[k * AL_TILE + (i + 1) % AL_TILE], "tile {k}, word {i}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check }
}

/// The partition sizes [`partition_sizes`] reports: reserved, cumulative
/// reserved, user, cumulative user and the total usage. The reserved array
/// takes bytes 0 to 24 and the reserved bytes 24 to 29; the user array
/// starts at 32 (aligned to 4) and ends at 72; the user tensor takes 128 to
/// 328 (aligned to 128).
pub const PARTITION_SIZES: [i32; 5] = [29, 29, 299, 328, 328];

/// `calculate_partition_size` and `calculate_total_usage` asked for
/// between allocations in both partitions, written to global memory: the
/// values count every allocation, those after the queries included.
pub fn partition_sizes() -> Case {
    let kernel = Kernel::builder("partition_sizes").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(5));
        let smem = SmemAllocator::new();
        let user = smem.allocate_array::<f32>(10, 4, SmemPartition::User);
        let sizes = [
            calculate_partition_size(SmemPartition::Reserved, false),
            calculate_partition_size(SmemPartition::Reserved, true),
            calculate_partition_size(SmemPartition::User, false),
            calculate_partition_size(SmemPartition::User, true),
            calculate_total_usage(),
        ];
        let reserved = smem.allocate_array::<u64>(3, 8, SmemPartition::Reserved);
        let tensor = smem.allocate_tensor::<u16>(Layout::col_major(100), 128, None, SmemPartition::User);
        let bytes = smem.allocate_bytes(5, 1, SmemPartition::Reserved);
        let t = thread_idx().x;
        if_(t.eq(0), || {
            user.store(1.0f32);
            reserved.store(2u64);
            tensor.store(0, 3u16);
            bytes.store(4u8);
            for (i, v) in sizes.iter().enumerate() {
                out.store(i, *v);
            }
        });
    });
    fn check(args: &[KernelArg]) {
        assert_eq!(args[0].to_vec::<i32>(), PARTITION_SIZES);
    }
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    Case { name: "partition_sizes", kernel, cfg, small_only: false, check }
}

/// The pipeline kernels the SM89 GPU runs that take no target: the state
/// walks and the partition sizes (compiled for sm_80 like the rest of
/// `feature_kernels::all`).
pub fn pipeline_cases() -> Vec<Case> {
    vec![pipeline_state_walk(), per_stage_state_walk(), partition_sizes()]
}

/// The pipeline kernels built for sm_89: the mbarrier hand-offs, under and
/// above 48 KiB of shared memory.
pub fn pipeline_cases_sm89() -> Vec<Case> {
    vec![mbarrier_array_handoff(), async_load_handoff(1024), async_load_handoff(56 * 1024)]
}

/// Words per stage of [`tma_load_pipeline`].
const TMA_WORDS: usize = 64;
/// Tiles of [`tma_load_pipeline`].
const TMA_ROUNDS: usize = 6;

/// The `TmaLoad` arm (sm_90): the full `MbarrierArray` counts one thread and
/// [`TMA_WORDS`] * 4 transaction bytes per phase. Warp 0 waits for an empty
/// stage, arrives on full (its elected lane, with the bytes to expect) and
/// loads the tile with `cp.async.bulk` completing on the stage's barrier;
/// warp 1 waits on full (`mbarrier.try_wait`), sums its word and arrives on
/// empty. No target: what CuTe DSL emits.
pub fn tma_load_pipeline() -> Case {
    let kernel = Kernel::builder("tma_load_pipeline").block_dim([128, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(TMA_ROUNDS * TMA_WORDS));
        let out = param_tensor::<u32>("out", Layout::col_major(TMA_WORDS));
        let t = thread_idx().x;
        let storage = alloc_reserved_mbarrier(2, 2);
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let consumers = CooperativeGroup::new(Agent::Warp, 2);
        let full = MbarrierArray::new(&storage, 2, (PipelineOp::TmaLoad, one), (TMA_WORDS * 4) as i32);
        let empty = MbarrierArray::new(&storage.offset(2), 2, (PipelineOp::AsyncThread, consumers), 0);
        let stages = alloc_smem::<u32>(2 * TMA_WORDS, 128);
        pipeline_init_arrive(None, false);
        pipeline_init_wait(None);
        if_else(
            warp_idx().eq(0),
            || {
                let start = make_pipeline_state(PipelineUserType::Producer, 2);
                for_range_carry(0, TMA_ROUNDS as i32, 1, start, |r, mut s| {
                    empty.wait(s.index(), s.phase());
                    full.arrive(s.index(), None, None);
                    if_(elect_one(), || {
                        cp_async_bulk(
                            &src.ptr().offset(r * TMA_WORDS as i32),
                            &stages.offset(s.index() * TMA_WORDS as i32),
                            (TMA_WORDS * 4) as i32,
                            &full.get_barrier(s.index()),
                        );
                    });
                    s.advance();
                    s
                });
            },
            || {
                if_(warp_idx().lt(3), || {
                    let me = t - 32;
                    let start = (Uint32::from(0u32), make_pipeline_state(PipelineUserType::Consumer, 2));
                    let (sum, _) = for_range_carry(0, TMA_ROUNDS as i32, 1, start, |_, (sum, mut s)| {
                        full.wait(s.index(), s.phase());
                        let word = stages.offset(s.index() * TMA_WORDS as i32 + me).load();
                        empty.arrive(s.index(), None, None);
                        s.advance();
                        (sum + word, s)
                    });
                    out.store(me, sum);
                });
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for (i, &got) in out.iter().enumerate() {
            let want = (0..TMA_ROUNDS).fold(0u32, |acc, r| acc.wrapping_add(src[r * TMA_WORDS + i]));
            assert_eq!(got, want, "word {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    Case { name: "tma_load_pipeline", kernel, cfg, small_only: false, check }
}

/// Two clusters of two CTAs: each CTA makes a full and an empty
/// one-stage `MbarrierArray` (`AsyncThread`, a warp of arrivals), and
/// `pipeline_init_arrive` (relaxed or not) and `pipeline_init_wait` with a
/// 2 x 1 cluster shape publish them to the cluster. Only then does CTA 1's
/// warp 0 arrive on CTA 0's full barrier (`dst` 0, through `mapa`); CTA 0's
/// warp 0 waits on it and arrives on CTA 1's empty barrier (`dst` 1), which
/// CTA 1 waits on. After a cluster `agent_sync`, every thread writes what
/// it saw and whether its CTA's full and empty phases 0 completed.
pub fn cluster_pipeline_init(relaxed: bool) -> Case {
    let name = if relaxed { "cluster_pipeline_init_relaxed" } else { "cluster_pipeline_init" };
    let kernel = Kernel::builder(name).block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::row_major((4 * 64, 3)));
        let t = thread_idx().x;
        let row = block_idx().x * 64 + t;
        let rank = block_idx_in_cluster();
        let storage = alloc_reserved_mbarrier(1, 2);
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let full = MbarrierArray::new(&storage, 1, (PipelineOp::AsyncThread, warp), 0);
        let empty = MbarrierArray::new(&storage.offset(1), 1, (PipelineOp::AsyncThread, warp), 0);
        let cluster = || Some(Shape::from((2, 1)));
        pipeline_init_arrive(cluster(), relaxed);
        pipeline_init_wait(cluster());
        let seen = if_else_yield(
            warp_idx().eq(0),
            || {
                if_else_yield(
                    rank.eq(1),
                    || {
                        full.arrive(0, Some(Int32::from(0)), None);
                        empty.wait(0, 0);
                        Int32::from(11)
                    },
                    || {
                        full.wait(0, 0);
                        empty.arrive(0, Some(Int32::from(1)), None);
                        Int32::from(22)
                    },
                )
            },
            || Int32::from(0),
        );
        agent_sync(Agent::ThreadBlockCluster, false);
        out.store((row, 0), seen);
        out.store((row, 1), full.test_wait(0, 0).to::<i32>());
        out.store((row, 2), empty.test_wait(0, 0).to::<i32>());
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<i32>();
        for b in 0..4 {
            for t in 0..64 {
                let rank = b % 2;
                let seen = if t >= 32 {
                    0
                } else if rank == 1 {
                    11
                } else {
                    22
                };
                let want = [seen, (rank == 0) as i32, (rank == 1) as i32];
                let row = b * 64 + t;
                assert_eq!(out[row * 3..row * 3 + 3], want, "block {b}, thread {t}");
            }
        }
    }
    let cfg = LaunchConfig::new([4, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check }
}

/// The `ClcLoad` arm in a cluster of two CTAs: each CTA's one-stage array
/// counts one thread (with no transaction bytes, there being no cluster
/// launch control response to wait for), and threads 0 and 1 of CTA 0 each
/// arrive with the bytes to expect on the barrier of CTA `t` (its own, and
/// its peer's). Each CTA's warp waits on its own barrier.
pub fn clc_load_arrivals() -> Case {
    let kernel = Kernel::builder("clc_load_arrivals").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(2 * 32));
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        let storage = alloc_reserved_mbarrier(1, 1);
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let bar = MbarrierArray::new(&storage, 1, (PipelineOp::ClcLoad, one), 0);
        let cluster = || Some(Shape::from((2, 1)));
        pipeline_init_arrive(cluster(), false);
        pipeline_init_wait(cluster());
        if_(rank.eq(0) & t.lt(2), || bar.arrive(0, Some(t), None));
        bar.wait(0, 0);
        out.store(block_idx().x * 32 + t, rank + 1);
        agent_sync(Agent::ThreadBlockCluster, true);
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<i32>();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i / 32) as i32 + 1, "thread {i}");
        }
    }
    let cfg = LaunchConfig::new([2, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    Case { name: "clc_load_arrivals", kernel, cfg, small_only: false, check }
}

/// The pipeline kernels that need sm_90.
pub fn pipeline_cases_sm90() -> Vec<Case> {
    vec![tma_load_pipeline(), cluster_pipeline_init(false), cluster_pipeline_init(true), clc_load_arrivals()]
}

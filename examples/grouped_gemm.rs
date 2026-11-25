// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grouped GEMMs on the grouped tile schedulers.
//!
//! [`build`]: a persistent grouped GEMM with SM80 `mma.sync`. The groups'
//! A (`M x K`), B (`N x K`) and C (`M x N`), all row-major, lie one after
//! another in three concatenated buffers, at the element offsets of a
//! `(group_count, 3)` offset table; their shapes are the rows of a
//! `(group_count, 4)` `(M, N, K, L)` tensor. Each CTA walks the linear
//! cluster tiles a [`StaticPersistentGroupTileScheduler`] hands it, which
//! finds each tile's group with a warp-wide search, and computes the tile
//! with four warps of `mma.sync` (`m16n8k16`, F16 in, F32 out), loading its
//! operands straight from global memory.
//!
//! [`moe`]: an MoE-style layer over one concatenated tensor of tokens,
//! `Y[t] = W[g] X[t]` for the tokens `t` of expert `g`, the experts' tokens
//! lying one after another as a `cumsum` of `group_count + 1` boundaries
//! gives them. Warp 0 of each CTA schedules: it runs
//! [`contiguous_group_search`] for each of its tiles of 16 tokens and hands
//! the tile's record (start, group, tokens to the group's end) to warp 1
//! through a two-stage `PipelineAsync`; warp 1 reads it with
//! [`make_contiguous_group_work_tile_info`] and computes the tile, stopping
//! at the record past the last group.
//!
//! Run with `cargo run --release --example grouped_gemm`.

use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions};
use cute::prelude::*;
use cute::utils::{
    PersistentTileSchedulerParams, StaticPersistentGroupTileScheduler, contiguous_group_search,
    create_initial_contiguous_group_search_state, create_initial_search_state, make_contiguous_group_work_tile_info,
};

/// The CTA tile along M, N and K.
pub const BM: usize = 32;
pub const BN: usize = 32;
pub const BK: usize = 16;
/// Threads per CTA of the GEMM: four warps.
pub const THREADS: u32 = 128;

/// A grouped problem: each group's `(M, N, K)`, multiples of the tile
/// (0 allowed for M and N).
#[derive(Clone, Debug)]
pub struct Groups {
    pub shapes: Vec<[usize; 3]>,
}

impl Groups {
    pub fn group_count(&self) -> usize {
        self.shapes.len()
    }

    /// The groups' cluster tiles (one CTA per cluster).
    pub fn total_tiles(&self) -> usize {
        self.shapes.iter().map(|s| (s[0] / BM) * (s[1] / BN)).sum()
    }

    /// The element offsets of each group's A, B and C in the concatenated
    /// buffers, and the buffers' lengths.
    pub fn offsets(&self) -> (Vec<[usize; 3]>, [usize; 3]) {
        let mut at = [0usize; 3];
        let mut out = Vec::new();
        for &[m, n, k] in &self.shapes {
            out.push(at);
            at = [at[0] + m * k, at[1] + n * k, at[2] + m * n];
        }
        (out, at)
    }

    /// The schedule: `(1, 1, total tiles)`, so that the work index is the
    /// linear cluster tile; the tile count is a kernel parameter.
    fn params(total: impl Into<IndexExpr>) -> PersistentTileSchedulerParams {
        PersistentTileSchedulerParams::new(Shape::from((1usize, 1usize, total.into())), [1, 1, 1], Default::default())
    }
}

/// The persistent grouped GEMM for `group_count` groups, the problem
/// shapes cached in registers (`cached`) or loaded for each search.
pub fn build(group_count: usize, cached: bool) -> Kernel {
    let name = if cached { "grouped_gemm_cached" } else { "grouped_gemm" };
    Kernel::builder(name).block_dim([THREADS, 1, 1]).target(Arch::Sm80).build(move || {
        let a = param_ptr::<F16>("a");
        let b = param_ptr::<F16>("b");
        let c = param_ptr::<f32>("c");
        let shapes = param_tensor::<i32>("problem_shape_mnkl", Layout::row_major((group_count, 4)));
        let offsets = param_tensor::<i32>("offsets", Layout::row_major((group_count, 3)));
        let total = param_scalar::<i32>("total_tiles");
        let params = Groups::params(total);
        let mut sched = StaticPersistentGroupTileScheduler::create(
            &params,
            block_idx(),
            grid_dim(),
            [BM, BN, BK],
            create_initial_search_state(),
            group_count,
            &shapes,
            cached,
        );
        sched.prefetch_problem_shapes();
        let work = sched.initial_work_tile_info();
        while_carry(
            (work, sched),
            |(work, _)| work.is_valid_tile(),
            |(work, mut sched)| {
                let r = work.group_search_result();
                let g = r.group_idx;
                let (m, n, k) = (r.problem_shape_m, r.problem_shape_n, r.problem_shape_k);
                let at = |i: usize| offsets.load((g, i));
                let ga = make_tensor(&a.offset(at(0)), Layout::row_major((IndexExpr::from(m), IndexExpr::from(k))));
                let gb = make_tensor(&b.offset(at(1)), Layout::row_major((IndexExpr::from(n), IndexExpr::from(k))));
                let gc = make_tensor(&c.offset(at(2)), Layout::row_major((IndexExpr::from(m), IndexExpr::from(n))));
                gemm_tile(&ga, &gb, &gc, r.cta_tile_idx_m, r.cta_tile_idx_n, r.cta_tile_count_k);
                sched.advance_to_next_work();
                let work = sched.get_current_work();
                (work, sched)
            },
        );
    })
}

/// One `BM x BN` tile of C over `k_tiles` tiles of K: the A and B
/// fragments of each `mma.sync` come straight from global memory.
fn gemm_tile(a: &Tensor<F16>, b: &Tensor<F16>, c: &Tensor<f32>, tile_m: Int32, tile_n: Int32, k_tiles: Int32) {
    let tid = thread_idx().x;
    let cta = (tile_m, tile_n, ..);
    let g_a = local_tile_proj(a, (BM, BN, BK), cta, &[true, false, true]);
    let g_b = local_tile_proj(b, (BM, BN, BK), cta, &[false, true, true]);
    let g_c = local_tile_proj(c, (BM, BN, BK), cta, &[true, true, false]);
    let mma = make_tiled_mma(make_mma_atom(MmaOp::Sm80(MmaKind::M16N8K16F32F16)), Layout::col_major((2, 2, 1)));
    let thr = mma.get_slice(tid);
    let t_a = thr.partition_a(&g_a);
    let t_b = thr.partition_b(&g_b);
    let r_a = thr.partition_fragment_a::<F16, _>(&g_a.slice((.., .., 0)));
    let r_b = thr.partition_fragment_b::<F16, _>(&g_b.slice((.., .., 0)));
    let t_c = thr.partition_c(&g_c);
    let acc = thr.partition_fragment_c::<f32, _>(&g_c);
    acc.fill(0.0);
    let acc = for_range_carry(0, k_tiles, 1, acc, |kt, acc| {
        autovec_copy(&t_a.slice((.., .., .., kt)), &r_a);
        autovec_copy(&t_b.slice((.., .., .., kt)), &r_b);
        gemm(&mma, &acc, &r_a, &r_b, &acc);
        acc
    });
    autovec_copy(&acc, &t_c);
}

/// The launch and arguments of [`build`] for `groups` with A and B from
/// `small_integers`, at most `max_ctas` persistent CTAs. Returns the
/// launch, the arguments, and A and B as `f32`.
pub fn problem(groups: &Groups, max_ctas: u32) -> (LaunchConfig, Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    for &[m, n, k] in &groups.shapes {
        assert!(m % BM == 0 && n % BN == 0 && k % BK == 0 && k > 0, "{m}x{n}x{k} is not a whole number of tiles");
    }
    let (offsets, [la, lb, lc]) = groups.offsets();
    let (a, b) = (small_integers(la.max(1), 1), small_integers(lb.max(1), 2));
    let total = groups.total_tiles();
    let grid = Groups::params(total.max(1)).get_grid_shape(max_ctas);
    let shapes: Vec<i32> = groups.shapes.iter().flat_map(|&[m, n, k]| [m as i32, n as i32, k as i32, 1]).collect();
    let offsets: Vec<i32> = offsets.iter().flat_map(|o| o.map(|x| x as i32)).collect();
    let args = vec![
        KernelArg::f16_from_f32(&a),
        KernelArg::f16_from_f32(&b),
        KernelArg::f32(vec![-1.0; lc.max(1)]),
        KernelArg::buffer(&shapes),
        KernelArg::buffer(&offsets),
        KernelArg::ScalarI32(total as i32),
    ];
    (LaunchConfig::new(grid, [THREADS, 1, 1]), args, a, b)
}

/// `len` integers from -8 to 8, exact in f16, whose sums of products are
/// exact in f32.
pub fn small_integers(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 17) as f32 - 8.0
        })
        .collect()
}

/// Check every group's C against a CPU GEMM.
pub fn check(groups: &Groups, a: &[f32], b: &[f32], c: &[f32]) {
    let (offsets, _) = groups.offsets();
    for (g, (&[m, n, k], o)) in groups.shapes.iter().zip(&offsets).enumerate() {
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|l| a[o[0] + i * k + l] * b[o[1] + j * k + l]).sum();
                assert_eq!(c[o[2] + i * n + j], want, "group {g}: C[{i},{j}] of {m}x{n}x{k}");
            }
        }
    }
}

/// The grouped problems [`main`] and the tests run: a single group, and
/// groups of several shapes with an empty one among them (more than 32,
/// so that the search slides its window).
pub fn problems() -> Vec<Groups> {
    let mut many = Vec::new();
    for g in 0..37 {
        let m = BM * (g % 3);
        let n = BN * (1 + g % 2);
        many.push([m, n, BK * (1 + g % 4)]);
    }
    vec![
        Groups { shapes: vec![[64, 96, 48]] },
        Groups { shapes: vec![[32, 64, 32], [0, 32, 16], [96, 32, 64], [64, 64, 16], [32, 0, 16], [32, 32, 80]] },
        Groups { shapes: many },
    ]
}

/// Tokens per MoE tile.
pub const MOE_TILE: usize = 16;
/// Features in and out of the MoE layer.
pub const MOE_IN: usize = 8;
pub const MOE_OUT: usize = 2;
/// Stages of the tile-record pipeline.
const MOE_STAGES: usize = 2;

/// The MoE layer for `group_count` experts: `x` is `tokens x MOE_IN`, `w`
/// is `(group_count, MOE_OUT, MOE_IN)` and `y` is `tokens x MOE_OUT`, all
/// row-major; `cumsum` has `group_count + 1` entries. Each CTA handles the
/// tiles `block_idx.x + i * grid_dim.x`.
pub fn moe(group_count: usize) -> Kernel {
    Kernel::builder("moe_contiguous").block_dim([64, 1, 1]).target(Arch::Sm80).build(move || {
        let x = param_ptr::<f32>("x");
        let w = param_tensor::<f32>("w", Layout::row_major((group_count, MOE_OUT, MOE_IN)));
        let y = param_ptr::<f32>("y");
        let cumsum = param_tensor::<i32>("cumsum", Layout::col_major(group_count + 1));
        // One record of 4 words per stage.
        let s_info = make_smem_tensor::<i32>(Layout::col_major((4, MOE_STAGES)));
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let pipeline = PipelineAsync::create(MOE_STAGES, warp, warp, PipelineAsyncOptions::default());
        let (producer, consumer) = pipeline.make_participants();
        let lane = lane_idx();
        let gc = group_count as i32;
        if_else(
            make_warp_uniform(warp_idx()).eq(0),
            || {
                // The scheduling warp: search, record, hand over, until the
                // record past the last group.
                let start =
                    (block_idx().x, create_initial_contiguous_group_search_state(), producer, Boolean::from(true));
                let (_, _, mut producer, _) = while_carry(
                    start,
                    |(_, _, _, more)| more,
                    |(linear, state, mut producer, _)| {
                        let handle = producer.acquire_and_advance(None);
                        let s = contiguous_group_search([MOE_TILE, 1, 1], gc, linear, state, &cumsum, 0);
                        let record = s_info.slice((.., handle.index()));
                        if_(lane.eq(0), || {
                            record.store(0, s.cur_start);
                            record.store(1, 0);
                            record.store(2, s.cur_group_idx - 1);
                            record.store(3, s.cur_boundary - s.cur_start);
                        });
                        handle.commit();
                        let more = s.cur_group_idx.le(gc);
                        (linear + grid_dim().x, s, producer, more)
                    },
                );
                producer.tail();
            },
            || {
                // The compute warp: lane `r + 16 e` computes output `e` of
                // token `r` of the tile.
                let (row, out) = (lane % MOE_TILE as i32, lane / MOE_TILE as i32);
                let start = (consumer, Boolean::from(true));
                while_carry(
                    start,
                    |(_, more)| more,
                    |(mut consumer, _)| {
                        let handle = consumer.wait_and_advance(None);
                        let info =
                            make_contiguous_group_work_tile_info(group_count, &s_info.slice((.., handle.index())));
                        handle.release();
                        let valid = info.is_valid_tile();
                        if_(valid & row.lt(info.distance_to_boundary), || {
                            let token = info.cta_coord_m + row;
                            let mut sum = Float32::from(0.0);
                            for d in 0..MOE_IN {
                                let xv = x.offset(token * MOE_IN as i32 + d as i32).load();
                                sum = sum + xv * w.load((info.group_idx, out, d));
                            }
                            y.offset(token * MOE_OUT as i32 + out).store(sum);
                        });
                        (consumer, valid)
                    },
                );
            },
        );
    })
}

/// An MoE problem: tokens per expert.
#[derive(Clone, Debug)]
pub struct Moe {
    pub tokens: Vec<usize>,
}

impl Moe {
    pub fn cumsum(&self) -> Vec<i32> {
        let mut out = vec![0];
        for t in &self.tokens {
            out.push(out.last().unwrap() + *t as i32);
        }
        out
    }

    pub fn total(&self) -> usize {
        self.tokens.iter().sum()
    }
}

/// The launch and arguments of [`moe`] on `ctas` CTAs, and X and W.
pub fn moe_problem(p: &Moe, ctas: u32) -> (LaunchConfig, Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    let t = p.total();
    let x = small_integers((t * MOE_IN).max(1), 3);
    let w = small_integers(p.tokens.len() * MOE_OUT * MOE_IN, 4);
    let args = vec![
        KernelArg::f32(x.clone()),
        KernelArg::f32(w.clone()),
        KernelArg::f32(vec![-1.0; (t * MOE_OUT).max(1)]),
        KernelArg::buffer(&p.cumsum()),
    ];
    (LaunchConfig::new([ctas, 1, 1], [64, 1, 1]), args, x, w)
}

/// Check Y against the CPU: each token through its expert's weights.
pub fn moe_check(p: &Moe, x: &[f32], w: &[f32], y: &[f32]) {
    let cumsum = p.cumsum();
    for g in 0..p.tokens.len() {
        for t in cumsum[g] as usize..cumsum[g + 1] as usize {
            for e in 0..MOE_OUT {
                let want: f32 = (0..MOE_IN).map(|d| x[t * MOE_IN + d] * w[(g * MOE_OUT + e) * MOE_IN + d]).sum();
                assert_eq!(y[t * MOE_OUT + e], want, "token {t} of expert {g}, output {e}");
            }
        }
    }
}

/// The MoE problems: experts with 0 to 40 tokens.
pub fn moe_problems() -> Vec<Moe> {
    vec![
        Moe { tokens: vec![5] },
        Moe { tokens: vec![17, 0, 33, 16, 1, 0, 40, 9] },
        Moe { tokens: (0..23).map(|g| (g * 7) % 29).collect() },
    ]
}

/// Simulate both kernels on every problem and check them.
pub fn run() {
    for groups in problems() {
        for cached in [false, true] {
            let kernel = build(groups.group_count(), cached);
            let (launch, mut args, a, b) = problem(&groups, 3);
            kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
            check(&groups, &a, &b, args[2].as_f32());
        }
    }
    for p in moe_problems() {
        for ctas in [1, 3] {
            let (launch, mut args, x, w) = moe_problem(&p, ctas);
            moe(p.tokens.len())
                .simulate_schedules(launch, &mut args, &SimOptions::default(), &[1, 2])
                .unwrap_or_else(|e| panic!("{e}"));
            moe_check(&p, &x, &w, args[2].as_f32());
        }
    }
}

#[allow(dead_code)]
fn main() {
    run();
    println!("grouped_gemm and moe_contiguous: PASS");
}

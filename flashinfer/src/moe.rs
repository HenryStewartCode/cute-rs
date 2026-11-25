// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's fused Mixture-of-Experts forward, `cutlass_fused_moe`, for Float16 and
//! BFloat16 activations and weights on SM80/SM89. This is cute-rs's own code written after
//! the published design of FlashInfer's CUTLASS backend (`csrc/fused_moe/cutlass_backend/
//! cutlass_fused_moe_kernels.cuh`, the SM80 fused gated-activation grouped GEMM of
//! `cutlass_extensions/gemm/kernel/fused_moe_kernel*.cuh` and the dispatch in
//! `moe_gemm_template_dispatch.h`, all Apache-2.0); no source was translated.
//!
//! **Interface** (FlashInfer's): `input` `[num_tokens, hidden]`; `token_selected_experts`
//! `[num_tokens, top_k]` (i32, each in `0..num_experts`, distinct per token) and
//! `token_final_scales` `[num_tokens, top_k]` (f32), the routing already computed;
//! `fc1_expert_weights` `[num_experts, 2 * inter, hidden]` whose first `inter` rows are the
//! up projection (`w3`) and last `inter` rows the gate (`w1`); `fc2_expert_weights`
//! `[num_experts, hidden, inter]`. The result is
//! `out[t] = sum_i scale[t, i] * W2[e_i] (silu(W1g[e_i] x_t) * (W1u[e_i] x_t))`, the
//! SwiGLU output and each expert's FC2 output rounded to the activation type as FlashInfer's
//! are, accumulated in f32.
//!
//! **Pipeline**, five launches, as FlashInfer's on SM89 (where `isFusedGatedActivation` holds
//! for 16-bit types, so FC1 and SwiGLU are one kernel and there is no finalize fusion):
//!
//! 1. [`routing_kernel`]: one CTA sorts the `num_tokens * top_k` (token, expert) pairs by
//!    expert: warp-aggregated shared-memory atomics (`match.any`) count and rank each pair,
//!    a block scan gives `expert_offsets` (FlashInfer's `expert_first_token_offset`), and each
//!    expert writes its GEMM M tiles into a tile schedule for each GEMM. The order within an
//!    expert is not FlashInfer's stable token order; nothing downstream depends on it.
//! 2. [`expand_kernel`]: copies each token's row to its `top_k` permuted rows (16-byte
//!    vectors), as FlashInfer's `expandInputRowsKernel`.
//! 3. [`gemm1_kernel`]: the grouped FC1 GEMM with SwiGLU in the epilogue. A CTA computes an
//!    up tile and the gate tile of the same columns together, as FlashInfer's
//!    `Fused_Moe_Kernel_sm80` does with two accumulators; here they are one `BN`-wide B tile
//!    whose first half reads up rows and second half gate rows, so each thread holds both
//!    halves of its outputs. `mma.sync.m16n8k16`, `ldmatrix`, a `cp.async` pipeline.
//! 4. [`gemm2_kernel`]: the grouped FC2 GEMM, same mainloop, stored in the activation type.
//! 5. [`finalize_kernel`]: `out[t] = sum_i scale * fc2[permuted row]` in f32
//!    (`finalizeMoeRoutingKernel`).
//!
//! The grouped GEMMs launch one CTA per (M tile, N tile) for an upper bound on the M tiles
//! (`ceil(rows / BM) + min(num_experts, rows)`), M fastest within groups of [`GROUP_M`] tiles
//! for L2 reuse; a CTA reads its expert and rows from the schedule, and those past the last
//! tile exit. (FlashInfer's CUTLASS kernels run persistent CTAs over a problem visitor.) The
//! tile is picked from the average rows per expert ([`pick_tiles`]): FlashInfer's
//! 16-row tile with 1 x 4 warps for decode, larger tiles for prefill.
//!
//! **Routing** ([`route_kernel`], [`Routing`]): router logits (f32, f16 or bf16)
//! `[num_tokens, num_experts]` to `token_selected_experts` and f32 `token_final_scales`, with
//! the semantics of FlashInfer's `RoutingMethodType`s `Default` (softmax, then top-k),
//! `RenormalizeNaive` (softmax, top-k, renormalize), `Renormalize` (top-k, then softmax) and
//! `DeepSeekV3` (`fused_topk_deepseek`: sigmoid plus bias, `n_group` / `topk_group` group
//! selection by top-2 sums, `routed_scaling_factor`), and the trtllm-gen `Llama4` (top-1,
//! sigmoid), `TopK` (raw logits as weights), `Sigmoid`, `SigmoidRenorm`, `MiniMax2` (sigmoid
//! plus bias, no groups, scaled sum-normalize) and `TopKSigmoid` (top-k, then sigmoid). Written after FlashInfer's
//! `csrc/fused_moe/noAuxTcKernels.cu` and its trtllm-gen routing documentation; nothing was
//! translated. One warp per token instead of FlashInfer's CTA of `num_experts` threads. On
//! SM89 FlashInfer runs only the DeepSeek-V3 kernel (trtllm-gen routing is SM100+); the
//! weights here are f32 where `fused_topk_deepseek` writes the logits' type. Experts are
//! returned in descending score order, ties to the lower index.
//!
//! Not yet covered: expert parallelism (`ep_size`, experts outside the node), tensor
//! parallelism, biases, the other activations (GeGLU, ReLU2, SwiGLU with alpha/beta/limit),
//! FP8/FP4/INT quantized weights or activations, the min-latency mode, the other routing
//! methods (Llama4, TopK, Sigmoid, SigmoidRenorm, MiniMax2, TopKSigmoid), fused shared
//! experts, routing over more than 512 experts or top-k above 32, and sizes that are not
//! multiples of the CTA tile. Offsets are 32-bit, so each weight tensor must have fewer than
//! 2^31 elements.
//!
//! `examples/moe.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`, on the
//! GPU against the f64 reference, then benchmarks it on FlashInfer's benchmark shapes.

use cute::prelude::*;

pub use crate::gemm::Tile;
pub use crate::norm::Dtype;

/// Elements per 16-byte vector.
pub const VEC: usize = 8;

/// Threads of the routing CTA (and the largest supported expert count).
pub const ROUTE_THREADS: usize = 512;

/// The M tiles of a group in the GEMMs' tile order.
pub const GROUP_M: usize = 4;

/// The tile for FC1/FC2 GEMMs whose experts average at most 16 rows: 16 x 128 x 64 with
/// 1 x 4 warps, FlashInfer's `CtaShape16x128x64_WarpShape16x32x64`, four stages.
pub const DECODE_TILE: Tile = Tile { bm: 16, bn: 128, bk: 64, stages: 4, wm: 1, wn: 4 };

/// A static upper bound on the rows of a global tensor whose rows hold `cols` elements.
fn row_bound(cols: usize) -> usize {
    ((1usize << 31) - 1) / cols / 128 * 128
}

fn validate(tile: Tile) {
    let vec = VEC;
    let threads = tile.threads();
    let tpr = tile.bk / vec;
    assert!(tile.bk.is_multiple_of(vec) && threads.is_multiple_of(tpr), "bad tile {tile:?}");
    assert!(tile.bm.is_multiple_of(threads / tpr) && tile.bn.is_multiple_of(threads / tpr), "bad tile {tile:?}");
    assert!(tile.bm.is_multiple_of(16 * tile.wm) && tile.bn.is_multiple_of(16 * tile.wn), "bad tile {tile:?}");
    assert!(tile.stages >= 2, "bad tile {tile:?}");
}

/// K-major shared memory for 16-bit elements: atoms of eight rows of `min(bk, 64)`
/// elements, swizzled for conflict-free `ldmatrix`.
fn k_major_smem(rows: usize, bk: usize, stages: usize) -> ComposedLayout {
    let row = bk.min(64);
    let bits = (row / 8).trailing_zeros().min(3);
    let atom = Layout::row_major((8, row));
    make_composed_layout(make_swizzle(bits, 3, 3), 0, tile_to_shape(&atom, (rows, bk, stages)))
}

fn tiled_mma(ab: DType, tile: Tile) -> TiledMma {
    let warps = Layout::col_major((tile.wm, tile.wn, 1));
    let mma = make_tiled_mma(MmaF16BF16Op::new(ab, DType::F32, (16, 8, 16)), warps);
    let perm = |n: usize| Some(Layout::new(n, 1));
    mma.with_permutation_mnk([perm(16 * tile.wm), perm(16 * tile.wn), perm(16)])
}

/// The CTA's mainloop: `g_a` `(BM, BK, k)` (rows at or past `a_rows` zero-filled) times
/// `g_b` `(BN, BK, k)`, both K-major. Returns this thread's MMA slice and f32 accumulators.
fn mainloop<AB: Numeric>(
    tile: Tile,
    k_tiles: usize,
    g_a: &Tensor<AB>,
    a_rows: Int32,
    g_b: &Tensor<AB>,
) -> (ThrMma, Tensor<f32>) {
    let Tile { bm, bn, bk, stages, .. } = tile;
    validate(tile);
    let threads = tile.threads();
    let tid = thread_idx().x;
    let s_a = make_smem_tensor::<AB>(k_major_smem(bm, bk, stages));
    let s_b = make_smem_tensor::<AB>(k_major_smem(bn, bk, stages));
    let cp = make_copy_atom::<AB>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
    let tpr = bk / VEC;
    let g2s = make_tiled_copy_tv(&cp, &Layout::row_major((threads / tpr, tpr)), &Layout::row_major((1, VEC)));
    let thr = g2s.get_slice(tid);
    let t_a_g = thr.partition_s(g_a);
    let t_a_s = thr.partition_d(&s_a);
    let t_b_g = thr.partition_s(g_b);
    let t_b_s = thr.partition_d(&s_b);
    let c_a = thr.partition_s(&make_identity_tensor((bm, bk)));
    let p_a = make_predicate(&c_a, |c| elem_less(c.mode(0).clone(), a_rows));

    let mma = tiled_mma(AB::DTYPE, tile);
    let thr_mma = mma.get_slice(tid);
    let r_a = thr_mma.partition_fragment_a::<AB, _>(&s_a.slice((.., .., 0)));
    let r_b = thr_mma.partition_fragment_b::<AB, _>(&s_b.slice((.., .., 0)));
    let acc = thr_mma.partition_fragment_c::<f32, _>(&make_identity_tensor((bm, bn)));
    acc.fill(0.0);
    let s2r_a = make_tiled_copy_a(&make_copy_atom::<AB>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
    let s2r_b = make_tiled_copy_b(&make_copy_atom::<AB>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
    let thr_s2r_a = s2r_a.get_slice(tid);
    let thr_s2r_b = s2r_b.get_slice(tid);
    let t_a_sv = thr_s2r_a.partition_s(&s_a);
    let t_b_sv = thr_s2r_b.partition_s(&s_b);
    let r_a_v = thr_s2r_a.retile_d(&r_a);
    let r_b_v = thr_s2r_b.retile_d(&r_b);
    let k_blocks = r_a.shape().mode(2).size();

    let issue = |k_tile: Int32, stage: Int32| {
        copy_if(&g2s, &p_a, &t_a_g.slice((.., .., .., k_tile)), &t_a_s.slice((.., .., .., stage)));
        copy(&g2s, &t_b_g.slice((.., .., .., k_tile)), &t_b_s.slice((.., .., .., stage)));
    };
    for s in 0..stages - 1 {
        if s < k_tiles {
            issue(Int32::from(s as i32), Int32::from(s as i32));
        }
        cp_async_commit_group();
    }
    let acc = for_range_carry(0, k_tiles, 1, acc, |k_tile, acc| {
        cp_async_wait_group((stages - 2) as u32);
        sync_threads();
        let next = k_tile + (stages - 1) as i32;
        if_(next.lt(k_tiles), || issue(next, next % stages as i32));
        cp_async_commit_group();
        let stage = k_tile % stages as i32;
        for kb in 0..k_blocks {
            copy(&s2r_a, &t_a_sv.slice((.., .., kb, stage)), &r_a_v.slice((.., .., kb)));
            copy(&s2r_b, &t_b_sv.slice((.., .., kb, stage)), &r_b_v.slice((.., .., kb)));
            gemm(&mma, &acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), &acc);
        }
        acc
    });
    cp_async_wait_group(0);
    (thr_mma, acc)
}

/// `silu(x) = x / (1 + exp(-x))` with `ex2.approx.ftz` and a fast divide.
fn silu(x: Float32) -> Float32 {
    let e = cute::math::exp2(x * Float32::from(-std::f32::consts::LOG2_E), true);
    cute::math::div(x, e + Float32::from(1.0f32), true)
}

/// A MoE layer's sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MoeShape {
    pub hidden: usize,
    pub inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

impl MoeShape {
    pub const fn new(hidden: usize, inter: usize, num_experts: usize, top_k: usize) -> MoeShape {
        MoeShape { hidden, inter, num_experts, top_k }
    }

    /// FLOPs of `num_tokens` tokens (both GEMMs).
    pub fn flops(&self, num_tokens: usize) -> f64 {
        6.0 * (num_tokens * self.top_k * self.hidden * self.inter) as f64
    }
}

/// The upper bound on the M tiles of `rows` permuted rows over `num_experts` experts in
/// `bm`-row tiles, rounded up to a multiple of [`GROUP_M`].
pub fn m_tile_bound(rows: usize, num_experts: usize, bm: usize) -> usize {
    (rows.div_ceil(bm) + num_experts.min(rows)).next_multiple_of(GROUP_M)
}

// ---------------------------------------------------------------------------------------------
// Routing: sort the (token, expert) pairs by expert
// ---------------------------------------------------------------------------------------------

/// The routing (permutation) kernel. Parameters: `token_selected_experts` (i32,
/// `[num_tokens * top_k]`), outputs `permuted_of` (the permuted row of each pair),
/// `expert_offsets` (`num_experts + 1`), `tiles1` (`num_experts + 1`, each expert's first
/// FC1 M tile), `sched1` (the expert of each FC1 M tile), `tiles2`, `sched2` (the same for
/// FC2), and `num_tokens`. One CTA of [`ROUTE_THREADS`] threads.
pub fn routing_kernel(num_experts: usize, top_k: usize, bm1: usize, bm2: usize) -> Kernel {
    assert!((1..=ROUTE_THREADS).contains(&num_experts), "at most {ROUTE_THREADS} experts");
    let name = format!("flashinfer_moe_routing_e{num_experts}_k{top_k}_m{bm1}_{bm2}");
    let warps = ROUTE_THREADS / 32;
    Kernel::builder(&name).block_dim([ROUTE_THREADS as u32, 1, 1]).build(|| {
        let selected = param_ptr::<i32>("token_selected_experts");
        let permuted_of = param_ptr::<i32>("permuted_of");
        let offsets = param_ptr::<i32>("expert_offsets");
        let tiles1 = param_ptr::<i32>("tiles1");
        let sched1 = param_ptr::<i32>("sched1");
        let tiles2 = param_ptr::<i32>("tiles2");
        let sched2 = param_ptr::<i32>("sched2");
        let num_tokens = param_scalar::<i32>("num_tokens");
        let tid = thread_idx().x;
        let lane = lane_idx();
        let warp = warp_idx();
        let e_n = num_experts as i32;
        let nt = ROUTE_THREADS as i32;
        let counts = make_smem_tensor::<i32>(Layout::col_major(num_experts));
        let warp_sums = make_smem_tensor::<i32>(Layout::col_major((warps, 3)));
        if_(tid.lt(e_n), || counts.store(tid, Int32::from(0)));
        sync_threads();

        // Count and rank: the lanes of a warp with the same expert add their number to its
        // counter once, and each takes its rank among them.
        let total = num_tokens * top_k as i32;
        for_range(0, total, nt, |j0| {
            let j = j0 + tid;
            let valid = j.lt(total);
            let e = select(valid, selected.load_at(j.min(total - 1)), Int32::from(-1));
            let peers = match_sync(FULL_MASK, e, MatchKind::Any);
            let below = peers & lanemask_lt();
            let leader = popc(peers ^ (peers - Uint32::from(1u32))) - Uint32::from(1u32);
            let base = Var::new(Int32::from(0));
            if_(valid & below.eq(Uint32::from(0u32)), || {
                base.set(atomic_add(&counts.ptr().offset(e), popc(peers).to::<i32>()));
            });
            let base = shuffle_sync(base.get(), leader.to::<i32>());
            if_(valid, || permuted_of.store_at(j, base + popc(below).to::<i32>()));
        });
        sync_threads();

        // Exclusive scans over the experts of the rows and of each GEMM's M tiles.
        let c = select(tid.lt(e_n), counts.load(tid.min(e_n - 1)), Int32::from(0));
        let t1 = (c + (bm1 as i32 - 1)) / bm1 as i32;
        let t2 = (c + (bm2 as i32 - 1)) / bm2 as i32;
        let scan = |v: Int32| {
            let mut x = v;
            for d in [1, 2, 4, 8, 16] {
                let y = shuffle_sync_up(x, d);
                x = select(lane.ge(d), x + y, x);
            }
            x
        };
        let (ic, i1, i2) = (scan(c), scan(t1), scan(t2));
        if_(lane.eq(31), || {
            warp_sums.store((warp, 0), ic);
            warp_sums.store((warp, 1), i1);
            warp_sums.store((warp, 2), i2);
        });
        sync_threads();
        let (mut pc, mut p1, mut p2) = (ic - c, i1 - t1, i2 - t2);
        for w in 0..warps {
            let before = Int32::from(w as i32).lt(warp);
            pc = pc + select(before, warp_sums.load((w, 0)), Int32::from(0));
            p1 = p1 + select(before, warp_sums.load((w, 1)), Int32::from(0));
            p2 = p2 + select(before, warp_sums.load((w, 2)), Int32::from(0));
        }
        if_(tid.lt(e_n), || {
            offsets.store_at(tid, pc);
            tiles1.store_at(tid, p1);
            tiles2.store_at(tid, p2);
            if_(tid.eq(e_n - 1), || {
                offsets.store_at(e_n, pc + c);
                tiles1.store_at(e_n, p1 + t1);
                tiles2.store_at(e_n, p2 + t2);
            });
            for_range(0, t1, 1, |x| sched1.store_at(p1 + x, tid));
            for_range(0, t2, 1, |x| sched2.store_at(p2 + x, tid));
        });
        sync_threads();
        if_(tid.lt(e_n), || counts.store(tid, pc));
        sync_threads();
        // Each pair's permuted row: its expert's first row plus its rank.
        for_range(tid, total, nt, |j| {
            let e = selected.load_at(j);
            permuted_of.store_at(j, counts.load(e) + permuted_of.load_at(j));
        });
    })
}

// ---------------------------------------------------------------------------------------------
// Routing: router logits to `token_selected_experts` and `token_final_scales`
// ---------------------------------------------------------------------------------------------

/// The element type of the router logits (and of the DeepSeek-V3 bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoreType {
    F32,
    F16,
    BF16,
}

impl ScoreType {
    pub fn name(self) -> &'static str {
        match self {
            ScoreType::F32 => "f32",
            ScoreType::F16 => "f16",
            ScoreType::BF16 => "bf16",
        }
    }

    pub fn round(self, x: f32) -> f32 {
        match self {
            ScoreType::F32 => x,
            ScoreType::F16 => Dtype::F16.round(x),
            ScoreType::BF16 => Dtype::BF16.round(x),
        }
    }

    pub fn buffer(self, v: &[f32]) -> KernelArg {
        match self {
            ScoreType::F32 => KernelArg::f32(v.to_vec()),
            ScoreType::F16 => Dtype::F16.buffer(v),
            ScoreType::BF16 => Dtype::BF16.buffer(v),
        }
    }
}

/// A routing method (FlashInfer's `RoutingMethodType` names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Routing {
    /// `Default` (softmax over all experts, then top-k; the weights are the probabilities),
    /// or with `renormalize` `RenormalizeNaive` (the k probabilities divided by their sum).
    SoftmaxTopK { renormalize: bool },
    /// `Renormalize`: top-k of the logits, then softmax over the k.
    Renormalize,
    /// `DeepSeekV3` (`fused_topk_deepseek`): `s = sigmoid(logit)`, `b = s + bias`; a group's
    /// score is the sum of its two largest `b`; the `topk_group` best of `n_group` groups are
    /// kept, and the top-k `b` among their experts chosen; the weights are
    /// `s / (sum s + 1e-20) * routed_scaling_factor`.
    DeepSeekV3 { n_group: usize, topk_group: usize },
    /// `Llama4`: top-1 of the logits, weight `sigmoid(logit)` (`top_k` must be 1).
    Llama4,
    /// `TopK`: top-k of the logits; the weights are the logits themselves.
    TopK,
    /// `Sigmoid`: top-k of `sigmoid(logit)`; the weights are those scores.
    Sigmoid,
    /// `SigmoidRenorm`: `Sigmoid`, then the weights divided by `sum + 1e-20`.
    SigmoidRenorm,
    /// `MiniMax2`: DeepSeek-V3 without groups: top-k of `sigmoid(logit) + bias` (the bias
    /// only selects), weights `s / (sum s + 1e-20) * routed_scaling_factor`.
    MiniMax2,
    /// `TopKSigmoid`: top-k of the logits, weight `sigmoid(logit)`.
    TopKSigmoid,
}

impl Routing {
    pub fn name(self) -> String {
        match self {
            Routing::SoftmaxTopK { renormalize: false } => "softmax_topk".into(),
            Routing::SoftmaxTopK { renormalize: true } => "softmax_topk_renorm".into(),
            Routing::Renormalize => "renormalize".into(),
            Routing::DeepSeekV3 { n_group, topk_group } => format!("deepseek_v3_g{n_group}_tg{topk_group}"),
            Routing::Llama4 => "llama4".into(),
            Routing::TopK => "topk".into(),
            Routing::Sigmoid => "sigmoid".into(),
            Routing::SigmoidRenorm => "sigmoid_renorm".into(),
            Routing::MiniMax2 => "minimax2".into(),
            Routing::TopKSigmoid => "topk_sigmoid".into(),
        }
    }
}

/// Tokens (warps) per CTA of [`route_kernel`].
pub const ROUTE_WARPS: usize = 4;

/// The launch of [`route_kernel`] for `num_tokens` tokens.
pub fn route_launch(num_tokens: usize) -> LaunchConfig {
    LaunchConfig::new([num_tokens.div_ceil(ROUTE_WARPS).max(1) as u32, 1, 1], [32 * ROUTE_WARPS as u32, 1, 1])
}

/// The routing kernel: router logits `[num_tokens, num_experts]` to `token_selected_experts`
/// (i32) and `token_final_scales` (f32), both `[num_tokens, top_k]`, in descending order of
/// the selection score (ties go to the lower expert index). Parameters: `logits`, `bias`
/// (`[num_experts]`, read by DeepSeek-V3 and MiniMax2 only), `token_selected_experts`,
/// `token_final_scales`, `num_tokens`, `routed_scaling_factor` (DeepSeek-V3 and MiniMax2 only).
///
/// One warp per token (FlashInfer's kernels use a CTA of `num_experts` threads per token):
/// lane `l` holds the `C = ceil(num_experts / 32)` contiguous experts `l * C ..`; each of the
/// `top_k` rounds is a local argmax and a five-step butterfly. For DeepSeek-V3 a group's
/// two largest scores are merged across its `group_size / C` lanes, and the groups are
/// chosen by `topk_group` rounds over one group per lane.
pub fn route_kernel(routing: Routing, score: ScoreType, num_experts: usize, top_k: usize) -> Kernel {
    match score {
        ScoreType::F32 => route_of::<f32>(routing, score, num_experts, top_k),
        ScoreType::F16 => route_of::<F16>(routing, score, num_experts, top_k),
        ScoreType::BF16 => route_of::<BF16>(routing, score, num_experts, top_k),
    }
}

/// The warp-wide argmax of `(vals[j], base + j)` with payload `pay[j]`: ties to the lower index.
fn warp_argmax(vals: &[Float32], pay: &[Float32], base: Int32) -> (Float32, Int32, Float32) {
    let (mut v, mut i, mut p) = (vals[0], base, pay[0]);
    for j in 1..vals.len() {
        let better = vals[j].gt(v);
        v = select(better, vals[j], v);
        i = select(better, base + j as i32, i);
        p = select(better, pay[j], p);
    }
    for off in [16, 8, 4, 2, 1] {
        let (ov, oi, op) = (shuffle_sync_bfly(v, off), shuffle_sync_bfly(i, off), shuffle_sync_bfly(p, off));
        let better = ov.gt(v) | (ov.eq(v) & oi.lt(i));
        v = select(better, ov, v);
        i = select(better, oi, i);
        p = select(better, op, p);
    }
    (v, i, p)
}

fn route_of<T: Numeric>(routing: Routing, score: ScoreType, num_experts: usize, top_k: usize) -> Kernel {
    let e_n = num_experts;
    assert!((1..=32 * 16).contains(&e_n) && (1..=32).contains(&top_k) && top_k <= e_n, "{e_n} experts, top {top_k}");
    let c = e_n.div_ceil(32);
    if let Routing::DeepSeekV3 { n_group, topk_group } = routing {
        let g = e_n / n_group.max(1);
        assert!(
            (1..=32).contains(&n_group) && e_n.is_multiple_of(32) && e_n.is_multiple_of(n_group) && g >= 2,
            "DeepSeek-V3 routing needs num_experts a multiple of 32 and of n_group <= 32"
        );
        assert!(g.is_multiple_of(c) && (g / c).is_power_of_two(), "group size {g} vs {c} experts per lane");
        assert!((1..=n_group).contains(&topk_group) && topk_group * g >= top_k, "topk_group {topk_group}");
    }
    if routing == Routing::Llama4 {
        assert_eq!(top_k, 1, "Llama4 routing selects one expert");
    }
    let name = format!("flashinfer_moe_route_{}_{}_e{e_n}_k{top_k}", routing.name(), score.name());
    Kernel::builder(&name).block_dim([32 * ROUTE_WARPS as u32, 1, 1]).build(|| {
        let logits = param_ptr::<T>("logits");
        let bias = param_ptr::<T>("bias");
        let out_idx = param_ptr::<i32>("token_selected_experts");
        let out_w = param_ptr::<f32>("token_final_scales");
        let num_tokens = param_scalar::<i32>("num_tokens");
        let scale = param_scalar::<f32>("routed_scaling_factor");
        let lane = lane_idx();
        let token = block_idx().x * ROUTE_WARPS as i32 + warp_idx();
        let ninf = Float32::from(f32::NEG_INFINITY);
        let one = Float32::from(1.0f32);
        let exp = |x: Float32| cute::math::exp2(x * Float32::from(std::f32::consts::LOG2_E), true);
        let sig = |v: Float32| cute::math::div(one, one + exp(Float32::from(0.0f32) - v), true);
        if_(token.lt(num_tokens), || {
            let base = lane * c as i32;
            let row = token * e_n as i32;
            let valid: Vec<Boolean> = (0..c).map(|j| (base + j as i32).lt(e_n as i32)).collect();
            let x: Vec<Float32> = (0..c)
                .map(|j| {
                    let e = (base + j as i32).min(e_n as i32 - 1);
                    select(valid[j], logits.load_at(row + e).to::<f32>(), ninf)
                })
                .collect();
            let mut picks: Vec<(Int32, Float32)> = Vec::with_capacity(top_k);
            let mut topk = |mut vals: Vec<Float32>, pay: &[Float32]| {
                for _ in 0..top_k {
                    let (v, i, p) = warp_argmax(&vals, pay, base);
                    picks.push((i, p));
                    let _ = v;
                    for (j, val) in vals.iter_mut().enumerate() {
                        *val = select((base + j as i32).eq(i), ninf, *val);
                    }
                }
            };
            let weights: Vec<Float32> = match routing {
                Routing::SoftmaxTopK { renormalize } => {
                    let max = warp_reduction_max(x.iter().skip(1).fold(x[0], |a, &b| a.max(b)), 32);
                    let ex: Vec<Float32> =
                        (0..c).map(|j| select(valid[j], exp(x[j] - max), Float32::from(0.0f32))).collect();
                    let local = ex.iter().skip(1).fold(ex[0], |a, &b| a + b);
                    topk(x.clone(), &ex);
                    let denom = if renormalize {
                        picks.iter().skip(1).fold(picks[0].1, |a, p| a + p.1)
                    } else {
                        warp_reduction_sum(local, 32)
                    };
                    picks.iter().map(|p| cute::math::div(p.1, denom, true)).collect()
                }
                Routing::Renormalize => {
                    topk(x.clone(), &x);
                    let top = picks[0].1;
                    let ex: Vec<Float32> = picks.iter().map(|p| exp(p.1 - top)).collect();
                    let sum = ex.iter().skip(1).fold(ex[0], |a, &b| a + b);
                    ex.iter().map(|&e| cute::math::div(e, sum, true)).collect()
                }
                Routing::DeepSeekV3 { n_group, topk_group } => {
                    let g = e_n / n_group;
                    let lanes = g / c; // lanes per group
                    let s: Vec<Float32> = x.iter().map(|&v| sig(v)).collect();
                    let b: Vec<Float32> = (0..c)
                        .map(|j| s[j] + bias.load_at((base + j as i32).min(e_n as i32 - 1)).to::<f32>())
                        .collect();
                    // The two largest `b` of this lane, merged over the group's lanes.
                    let (mut t1, mut t2) = (b[0], ninf);
                    for &v in &b[1..] {
                        t2 = t2.max(t1.min(v));
                        t1 = t1.max(v);
                    }
                    let mut off = lanes / 2;
                    while off >= 1 {
                        let (o1, o2) = (shuffle_sync_bfly(t1, off as i32), shuffle_sync_bfly(t2, off as i32));
                        t2 = t1.min(o1).max(t2.max(o2));
                        t1 = t1.max(o1);
                        off /= 2;
                    }
                    // Lane `k < n_group` holds group `k`'s score; choose `topk_group` groups.
                    let gl = lane.min(n_group as i32 - 1);
                    let gs = shuffle_sync(t1 + t2, gl * lanes as i32);
                    let mut gv = select(lane.lt(n_group as i32), gs, ninf);
                    let mut chosen = Int32::from(0);
                    for _ in 0..topk_group {
                        let (_, gi, _) = warp_argmax(&[gv], &[gv], lane);
                        let me = lane.eq(gi);
                        chosen = select(me, Int32::from(1), chosen);
                        gv = select(me, ninf, gv);
                    }
                    let mine = shuffle_sync(chosen, lane / lanes as i32).ne(0);
                    let masked: Vec<Float32> = (0..c).map(|j| select(mine & valid[j], b[j], ninf)).collect();
                    topk(masked, &s);
                    let sum = picks.iter().skip(1).fold(picks[0].1, |a, p| a + p.1);
                    let k = cute::math::div(scale, sum + Float32::from(1e-20f32), true);
                    picks.iter().map(|p| p.1 * k).collect()
                }
                Routing::TopK => {
                    topk(x.clone(), &x);
                    picks.iter().map(|p| p.1).collect()
                }
                Routing::Llama4 | Routing::TopKSigmoid => {
                    topk(x.clone(), &x);
                    picks.iter().map(|p| sig(p.1)).collect()
                }
                Routing::Sigmoid | Routing::SigmoidRenorm => {
                    let s: Vec<Float32> = (0..c).map(|j| select(valid[j], sig(x[j]), ninf)).collect();
                    topk(s.clone(), &s);
                    if routing == Routing::Sigmoid {
                        picks.iter().map(|p| p.1).collect()
                    } else {
                        let sum = picks.iter().skip(1).fold(picks[0].1, |a, p| a + p.1);
                        let k = cute::math::div(one, sum + Float32::from(1e-20f32), true);
                        picks.iter().map(|p| p.1 * k).collect()
                    }
                }
                Routing::MiniMax2 => {
                    let s: Vec<Float32> = x.iter().map(|&v| sig(v)).collect();
                    let b: Vec<Float32> = (0..c)
                        .map(|j| {
                            let e = (base + j as i32).min(e_n as i32 - 1);
                            select(valid[j], s[j] + bias.load_at(e).to::<f32>(), ninf)
                        })
                        .collect();
                    topk(b, &s);
                    let sum = picks.iter().skip(1).fold(picks[0].1, |a, p| a + p.1);
                    let k = cute::math::div(scale, sum + Float32::from(1e-20f32), true);
                    picks.iter().map(|p| p.1 * k).collect()
                }
            };
            let at = token * top_k as i32;
            for (r, (p, w)) in picks.iter().zip(&weights).enumerate() {
                if_(lane.eq(r as i32), || {
                    out_idx.store_at(at + r as i32, p.0);
                    out_w.store_at(at + r as i32, *w);
                });
            }
        });
    })
}

/// A routing problem with its host data.
pub struct RouteProblem {
    pub routing: Routing,
    pub score: ScoreType,
    pub num_tokens: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub routed_scaling_factor: f32,
    /// `[num_tokens, num_experts]`, rounded to `score`.
    pub logits: Vec<f32>,
    /// `[num_experts]`, rounded to `score`.
    pub bias: Vec<f32>,
}

impl RouteProblem {
    /// Normal-ish random logits and (for DeepSeek-V3) a small random bias.
    pub fn new(
        routing: Routing,
        score: ScoreType,
        num_tokens: usize,
        num_experts: usize,
        top_k: usize,
        seed: u64,
    ) -> RouteProblem {
        let u =
            |n: usize, s: u64, scale: f32| uniform(n, s, scale).into_iter().map(|x| score.round(x)).collect::<Vec<_>>();
        let logits: Vec<f32> = {
            let (a, b) =
                (uniform(num_tokens * num_experts, seed, 1.0), uniform(num_tokens * num_experts, seed + 1, 1.0));
            a.iter().zip(&b).map(|(x, y)| score.round(1.5 * (x + y))).collect()
        };
        let bias = u(num_experts, seed + 2, 0.1);
        let routed_scaling_factor =
            if matches!(routing, Routing::DeepSeekV3 { .. } | Routing::MiniMax2) { 2.5 } else { 1.0 };
        RouteProblem { routing, score, num_tokens, num_experts, top_k, routed_scaling_factor, logits, bias }
    }

    pub fn kernel(&self) -> Kernel {
        route_kernel(self.routing, self.score, self.num_experts, self.top_k)
    }

    pub fn launch(&self) -> LaunchConfig {
        route_launch(self.num_tokens)
    }

    /// `logits, bias, token_selected_experts, token_final_scales, num_tokens,
    /// routed_scaling_factor`.
    pub fn args(&self) -> Vec<KernelArg> {
        vec![
            self.score.buffer(&self.logits),
            self.score.buffer(&self.bias),
            KernelArg::buffer(&vec![-1i32; self.num_tokens * self.top_k]),
            KernelArg::f32(vec![0.0; self.num_tokens * self.top_k]),
            KernelArg::ScalarI32(self.num_tokens as i32),
            KernelArg::ScalarF32(self.routed_scaling_factor),
        ]
    }

    /// Token `t`'s selection keys (higher is chosen first; `-inf` not eligible) and weight
    /// numerators in f64, and the weights of a chosen set.
    fn keys(&self, t: usize) -> (Vec<f64>, Vec<f64>) {
        let e_n = self.num_experts;
        let x: Vec<f64> = self.logits[t * e_n..(t + 1) * e_n].iter().map(|&v| v as f64).collect();
        match self.routing {
            Routing::SoftmaxTopK { .. }
            | Routing::Renormalize
            | Routing::TopK
            | Routing::Llama4
            | Routing::TopKSigmoid => (x.clone(), x),
            Routing::Sigmoid | Routing::SigmoidRenorm => {
                let s: Vec<f64> = x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
                (s.clone(), s)
            }
            Routing::MiniMax2 => {
                let s: Vec<f64> = x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
                let b = s.iter().zip(&self.bias).map(|(s, &b)| s + b as f64).collect();
                (b, s)
            }
            Routing::DeepSeekV3 { n_group, topk_group } => {
                let s: Vec<f64> = x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
                let b: Vec<f64> = s.iter().zip(&self.bias).map(|(s, &b)| s + b as f64).collect();
                let g = e_n / n_group;
                let mut gs: Vec<(f64, usize)> = (0..n_group)
                    .map(|k| {
                        let mut v = b[k * g..(k + 1) * g].to_vec();
                        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        (v[0] + v[1], k)
                    })
                    .collect();
                gs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                let keep: Vec<usize> = gs[..topk_group].iter().map(|p| p.1).collect();
                let keys = (0..e_n).map(|e| if keep.contains(&(e / g)) { b[e] } else { f64::NEG_INFINITY }).collect();
                (keys, s)
            }
        }
    }

    fn weights(&self, t: usize, chosen: &[usize], num: &[f64]) -> Vec<f64> {
        match self.routing {
            Routing::SoftmaxTopK { renormalize } => {
                let x = &num;
                let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
                let denom = if renormalize { chosen.iter().map(|&e| p[e]).sum() } else { p.iter().sum::<f64>() };
                chosen.iter().map(|&e| p[e] / denom).collect()
            }
            Routing::Renormalize => {
                let max = chosen.iter().map(|&e| num[e]).fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = chosen.iter().map(|&e| (num[e] - max).exp()).sum();
                chosen.iter().map(|&e| (num[e] - max).exp() / sum).collect()
            }
            Routing::TopK | Routing::Sigmoid => chosen.iter().map(|&e| num[e]).collect(),
            Routing::Llama4 | Routing::TopKSigmoid => chosen.iter().map(|&e| 1.0 / (1.0 + (-num[e]).exp())).collect(),
            Routing::SigmoidRenorm => {
                let sum: f64 = chosen.iter().map(|&e| num[e]).sum();
                chosen.iter().map(|&e| num[e] / (sum + 1e-20)).collect()
            }
            Routing::DeepSeekV3 { .. } | Routing::MiniMax2 => {
                let _ = t;
                let sum: f64 = chosen.iter().map(|&e| num[e]).sum();
                chosen.iter().map(|&e| num[e] / (sum + 1e-20) * self.routed_scaling_factor as f64).collect()
            }
        }
    }

    /// The reference selection of every token (descending key, ties to the lower index).
    pub fn reference(&self) -> (Vec<i32>, Vec<f64>) {
        let (mut idx, mut w) = (Vec::new(), Vec::new());
        for t in 0..self.num_tokens {
            let (keys, num) = self.keys(t);
            let mut order: Vec<usize> = (0..self.num_experts).collect();
            order.sort_by(|&a, &b| keys[b].partial_cmp(&keys[a]).unwrap().then(a.cmp(&b)));
            let chosen = &order[..self.top_k];
            idx.extend(chosen.iter().map(|&e| e as i32));
            w.extend(self.weights(t, chosen, &num));
        }
        (idx, w)
    }

    /// Checks `got_idx` / `got_w` against the reference: each token's experts must be distinct
    /// and in descending key order, with keys equal to the reference's top-k keys within
    /// 1e-5 (so near-ties may resolve either way); the weights, recomputed in f64 for the
    /// chosen experts, must agree within 1e-5 relative. Returns the largest weight error in
    /// units of that tolerance, or infinity if the selection is wrong.
    pub fn check(&self, got_idx: &[i32], got_w: &[f32]) -> f64 {
        let k = self.top_k;
        let mut worst = 0.0f64;
        for t in 0..self.num_tokens {
            let (keys, num) = self.keys(t);
            let mut sorted = keys.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let chosen: Vec<usize> = got_idx[t * k..(t + 1) * k].iter().map(|&e| e as usize).collect();
            for (r, &e) in chosen.iter().enumerate() {
                if e >= self.num_experts || chosen[..r].contains(&e) || (keys[e] - sorted[r]).abs() > 1e-5 {
                    return f64::INFINITY;
                }
            }
            let want = self.weights(t, &chosen, &num);
            for (g, w) in got_w[t * k..(t + 1) * k].iter().zip(&want) {
                worst = worst.max((*g as f64 - w).abs() / (1e-5 * (w.abs() + 1e-3)));
            }
        }
        worst
    }
}

/// Simulate `p` and return [`RouteProblem::check`].
pub fn simulate_route(p: &RouteProblem) -> f64 {
    let k = p.kernel();
    let mut args = p.args();
    k.simulate(p.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(&args[2].to_vec::<i32>(), &args[3].to_vec::<f32>())
}

// ---------------------------------------------------------------------------------------------
// Expand and finalize
// ---------------------------------------------------------------------------------------------

fn row_threads(hidden: usize) -> usize {
    (hidden / VEC).min(256)
}

/// Copies each token's row of `input` to its `top_k` rows of `permuted_input`. Parameters:
/// `input`, `permuted_of`, `permuted_input`. One CTA per token.
pub fn expand_kernel(dtype: Dtype, hidden: usize, top_k: usize) -> Kernel {
    match dtype {
        Dtype::F16 => expand_of::<F16>(dtype, hidden, top_k),
        Dtype::BF16 => expand_of::<BF16>(dtype, hidden, top_k),
    }
}

fn expand_of<T: Numeric>(dtype: Dtype, hidden: usize, top_k: usize) -> Kernel {
    assert!(hidden.is_multiple_of(VEC));
    let threads = row_threads(hidden);
    let name = format!("flashinfer_moe_expand_{}_h{hidden}_k{top_k}", dtype.name());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let input = param_ptr::<T>("input");
        let permuted_of = param_ptr::<i32>("permuted_of");
        let out = param_ptr::<T>("permuted_input");
        let tok = block_idx().x;
        let nv = (hidden / VEC) as i32;
        let v8 = VEC as i32;
        let dst: Vec<Int32> = (0..top_k).map(|i| permuted_of.load_at(tok * top_k as i32 + i as i32)).collect();
        for_range(thread_idx().x, nv, threads as i32, |v| {
            let x = load_vector::<T, VEC>(&input.offset((tok * nv + v) * v8), LoadOptions::default());
            for d in &dst {
                store_vector::<T, VEC>(&out.offset((*d * nv + v) * v8), x, StoreOptions::default());
            }
        });
    })
}

/// `out[t] = sum_i scales[t, i] * fc2_out[permuted_of[t, i]]` in f32. Parameters: `fc2_out`,
/// `permuted_of`, `token_final_scales`, `out`. One CTA per token.
pub fn finalize_kernel(dtype: Dtype, hidden: usize, top_k: usize) -> Kernel {
    match dtype {
        Dtype::F16 => finalize_of::<F16>(dtype, hidden, top_k),
        Dtype::BF16 => finalize_of::<BF16>(dtype, hidden, top_k),
    }
}

fn finalize_of<T: Numeric>(dtype: Dtype, hidden: usize, top_k: usize) -> Kernel {
    assert!(hidden.is_multiple_of(VEC));
    let threads = row_threads(hidden);
    let name = format!("flashinfer_moe_finalize_{}_h{hidden}_k{top_k}", dtype.name());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let y = param_ptr::<T>("fc2_out");
        let permuted_of = param_ptr::<i32>("permuted_of");
        let scales = param_ptr::<f32>("token_final_scales");
        let out = param_ptr::<T>("out");
        let tok = block_idx().x;
        let nv = (hidden / VEC) as i32;
        let v8 = VEC as i32;
        let k = top_k as i32;
        let src: Vec<(Int32, Float32)> =
            (0..top_k).map(|i| (permuted_of.load_at(tok * k + i as i32), scales.load_at(tok * k + i as i32))).collect();
        for_range(thread_idx().x, nv, threads as i32, |v| {
            let rows: Vec<[Value<T>; VEC]> = src
                .iter()
                .map(|(p, _)| load_vector::<T, VEC>(&y.offset((*p * nv + v) * v8), LoadOptions::default()))
                .collect();
            let mut acc = [Float32::from(0.0f32); VEC];
            for ((_, s), r) in src.iter().zip(&rows) {
                for j in 0..VEC {
                    acc[j] = acc[j] + *s * r[j].to::<f32>();
                }
            }
            let o: [Value<T>; VEC] = std::array::from_fn(|j| acc[j].to::<T>());
            store_vector::<T, VEC>(&out.offset((tok * nv + v) * v8), o, StoreOptions::default());
        });
    })
}

// ---------------------------------------------------------------------------------------------
// Grouped GEMMs
// ---------------------------------------------------------------------------------------------

/// This CTA's M tile and N tile for `n_tiles` N tiles: groups of [`GROUP_M`] M tiles walk
/// every N tile, M fastest.
fn cta_tile(n_tiles: usize) -> (Int32, Int32) {
    let id = block_idx().x;
    let per_group = (GROUP_M * n_tiles) as i32;
    let within = id % per_group;
    (id / per_group * GROUP_M as i32 + within % GROUP_M as i32, within / GROUP_M as i32)
}

/// For M tile `m_tile` of the schedule: `body(expert, first row, rows left in the expert)`,
/// if the tile exists.
fn with_tile(
    num_experts: usize,
    bm: usize,
    m_tile: Int32,
    offsets: &Ptr<i32>,
    tiles: &Ptr<i32>,
    sched: &Ptr<i32>,
    body: impl FnOnce(Int32, Int32, Int32),
) {
    let total = tiles.load_at(num_experts as i32);
    if_(m_tile.lt(total), || {
        let e = sched.load_at(m_tile);
        let row0 = offsets.load_at(e) + (m_tile - tiles.load_at(e)) * bm as i32;
        let rows = offsets.load_at(e + 1) - row0;
        body(e, row0, rows);
    });
}

/// The FC1 GEMM with SwiGLU: `h = silu(A W1g^T) * (A W1u^T)` per expert, `A` the permuted
/// input `[rows, hidden]`, `h` `[rows, inter]`. Parameters: `permuted_input`,
/// `fc1_expert_weights`, `h`, `expert_offsets`, `tiles1`, `sched1`. The tile's `bn` covers
/// `bn / 2` outputs (up and gate). Launch with [`gemm1_launch`].
pub fn gemm1_kernel(dtype: Dtype, shape: MoeShape, tile: Tile) -> Kernel {
    match dtype {
        Dtype::F16 => gemm1_of::<F16>(dtype, shape, tile),
        Dtype::BF16 => gemm1_of::<BF16>(dtype, shape, tile),
    }
}

pub fn gemm1_launch(shape: MoeShape, rows: usize, tile: Tile) -> LaunchConfig {
    let m = m_tile_bound(rows, shape.num_experts, tile.bm);
    LaunchConfig::new([(m * shape.inter / (tile.bn / 2)) as u32, 1, 1], [tile.threads() as u32, 1, 1])
}

fn gemm1_of<T: Numeric>(dtype: Dtype, shape: MoeShape, tile: Tile) -> Kernel {
    let MoeShape { hidden: k, inter, num_experts, .. } = shape;
    let Tile { bm, bn, bk, .. } = tile;
    let h = bn / 2;
    assert!(k.is_multiple_of(bk), "hidden {k} must be a multiple of {bk}");
    assert!(h.is_multiple_of(16 * tile.wn) && inter.is_multiple_of(h), "inter {inter} / tile {tile:?}");
    assert!(num_experts * 2 * inter * k < 1 << 31, "offsets are 32-bit");
    let n_tiles = inter / h;
    let name = format!("flashinfer_moe_gemm1_swiglu_{}_k{k}_n{inter}_e{num_experts}_{}", dtype.name(), tile.name());
    Kernel::builder(&name).block_dim([tile.threads() as u32, 1, 1]).build(|| {
        let a = param_ptr::<T>("permuted_input");
        let w = param_ptr::<T>("fc1_expert_weights");
        let out = param_ptr::<T>("h");
        let offsets = param_ptr::<i32>("expert_offsets");
        let tiles = param_ptr::<i32>("tiles1");
        let sched = param_ptr::<i32>("sched1");
        let (m_tile, n_tile) = cta_tile(n_tiles);
        with_tile(num_experts, bm, m_tile, &offsets, &tiles, &sched, |e, row0, rows| {
            let a_all = make_tensor(&a, Layout::row_major((row_bound(k), k)));
            let g_a =
                local_tile_proj(&domain_offset((row0, 0), &a_all), (bm, bn, bk), (0, 0, ..), &[true, false, true]);
            // B `(BN, BK, k)`: columns `0..h` are up rows `n_tile * h ..`, columns `h..2h` the
            // gate rows `inter + n_tile * h ..` of expert `e`.
            let w_e = w.offset(e * (2 * inter * k) as i32 + n_tile * (h * k) as i32);
            let g_b = make_tensor(&w_e, Layout::new(((h, 2), bk, k / bk), ((k, inter * k), 1, bk)));
            let (thr_mma, acc) = mainloop(tile, k / bk, &g_a, rows, &g_b);
            // The accumulators' second half holds the gate values of the first half's
            // columns (`h` is a multiple of the MMA's N tile): store `silu(gate) * up` of the
            // first half; the gate columns alias the up columns and are predicated off.
            let o_t = out.offset(row0 * inter as i32 + n_tile * h as i32);
            let g_c = make_tensor(&o_t, Layout::new((bm, (h, 2)), (inter, (1, 0))));
            let t_c_g = thr_mma.partition_c(&g_c);
            let c_c = thr_mma.partition_c(&make_identity_tensor((bm, bn)));
            let pred = make_predicate(&c_c, |c| {
                elem_less(c.mode(0).clone(), rows) & elem_less(c.mode(1).clone(), Int32::from(h as i32))
            });
            let half = acc.size() / 2;
            let o = make_fragment_like::<T, _>(&acc);
            for i in 0..acc.size() {
                let v = if i < half { silu(acc.load(i + half)) * acc.load(i) } else { Float32::from(0.0f32) };
                o.store(i, v.to::<T>());
            }
            copy_if(&make_copy_atom::<T>(CopyOp::Universal { bits: 32 }), &pred, &o, &t_c_g);
        });
    })
}

/// The FC2 GEMM: `y = h W2^T` per expert, `h` `[rows, inter]`, `y` `[rows, hidden]`.
/// Parameters: `h`, `fc2_expert_weights`, `fc2_out`, `expert_offsets`, `tiles2`, `sched2`.
/// Launch with [`gemm2_launch`].
pub fn gemm2_kernel(dtype: Dtype, shape: MoeShape, tile: Tile) -> Kernel {
    match dtype {
        Dtype::F16 => gemm2_of::<F16>(dtype, shape, tile),
        Dtype::BF16 => gemm2_of::<BF16>(dtype, shape, tile),
    }
}

pub fn gemm2_launch(shape: MoeShape, rows: usize, tile: Tile) -> LaunchConfig {
    let m = m_tile_bound(rows, shape.num_experts, tile.bm);
    LaunchConfig::new([(m * shape.hidden / tile.bn) as u32, 1, 1], [tile.threads() as u32, 1, 1])
}

fn gemm2_of<T: Numeric>(dtype: Dtype, shape: MoeShape, tile: Tile) -> Kernel {
    let MoeShape { hidden: n, inter: k, num_experts, .. } = shape;
    let Tile { bm, bn, bk, .. } = tile;
    assert!(k.is_multiple_of(bk) && n.is_multiple_of(bn), "inter {k} / hidden {n} vs tile {tile:?}");
    assert!(num_experts * n * k < 1 << 31, "offsets are 32-bit");
    let n_tiles = n / bn;
    let name = format!("flashinfer_moe_gemm2_{}_k{k}_n{n}_e{num_experts}_{}", dtype.name(), tile.name());
    Kernel::builder(&name).block_dim([tile.threads() as u32, 1, 1]).build(|| {
        let a = param_ptr::<T>("h");
        let w = param_ptr::<T>("fc2_expert_weights");
        let y = param_ptr::<T>("fc2_out");
        let offsets = param_ptr::<i32>("expert_offsets");
        let tiles = param_ptr::<i32>("tiles2");
        let sched = param_ptr::<i32>("sched2");
        let (m_tile, n_tile) = cta_tile(n_tiles);
        with_tile(num_experts, bm, m_tile, &offsets, &tiles, &sched, |e, row0, rows| {
            let a_all = make_tensor(&a, Layout::row_major((row_bound(k), k)));
            let g_a =
                local_tile_proj(&domain_offset((row0, 0), &a_all), (bm, bn, bk), (0, 0, ..), &[true, false, true]);
            let w_e = w.offset(e * (n * k) as i32 + n_tile * (bn * k) as i32);
            let g_b = make_tensor(&w_e, Layout::new((bn, bk, k / bk), (k, 1, bk)));
            let (thr_mma, acc) = mainloop(tile, k / bk, &g_a, rows, &g_b);
            let y_t = y.offset(row0 * n as i32 + n_tile * bn as i32);
            let g_c = make_tensor(&y_t, Layout::new((bm, bn), (n, 1)));
            let t_c_g = thr_mma.partition_c(&g_c);
            let c_c = thr_mma.partition_c(&make_identity_tensor((bm, bn)));
            let pred = make_predicate(&c_c, |c| elem_less(c.mode(0).clone(), rows));
            let o = make_fragment_like::<T, _>(&acc);
            for i in 0..acc.size() {
                o.store(i, acc.load(i).to::<T>());
            }
            copy_if(&make_copy_atom::<T>(CopyOp::Universal { bits: 32 }), &pred, &o, &t_c_g);
        });
    })
}

// ---------------------------------------------------------------------------------------------
// The problem
// ---------------------------------------------------------------------------------------------

fn uniform(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Random routing: `top_k` distinct experts per token, uniformly, and positive scales that
/// sum to one per token (as softmax-top-k-renormalize routing of random logits gives).
pub fn random_routing(num_tokens: usize, num_experts: usize, top_k: usize, seed: u64) -> (Vec<i32>, Vec<f32>) {
    assert!(top_k <= num_experts);
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut sel = Vec::with_capacity(num_tokens * top_k);
    let mut scales = Vec::with_capacity(num_tokens * top_k);
    for _ in 0..num_tokens {
        let mut chosen: Vec<i32> = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let e = (next() % num_experts as u64) as i32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let w: Vec<f32> = (0..top_k).map(|_| 0.1 + (next() % 1000) as f32 / 1000.0).collect();
        let s: f32 = w.iter().sum();
        sel.extend(chosen);
        scales.extend(w.iter().map(|x| x / s));
    }
    (sel, scales)
}

/// The GEMM tiles for `rows` permuted rows over `num_experts` experts (measured on an RTX
/// 2000 Ada): [`DECODE_TILE`] for both GEMMs while experts average at most 16 rows, then
/// 64-row tiles of 2 x 2 warps, then 128 x 256 x 64 with 2 x 4 warps and two stages.
pub fn pick_tiles(rows: usize, num_experts: usize) -> (Tile, Tile) {
    let avg = rows.div_ceil(num_experts);
    if avg <= 16 {
        (DECODE_TILE, DECODE_TILE)
    } else if avg <= 64 {
        let t = Tile { bm: 64, bn: 128, bk: 64, stages: 3, wm: 2, wn: 2 };
        (t, t)
    } else {
        let t = Tile { bm: 128, bn: 256, bk: 64, stages: 2, wm: 2, wn: 4 };
        (t, t)
    }
}

/// A MoE problem with its host data.
pub struct Problem {
    pub dtype: Dtype,
    pub shape: MoeShape,
    pub num_tokens: usize,
    /// `[num_tokens, hidden]`.
    pub x: Vec<f32>,
    /// `[num_experts, 2 * inter, hidden]` (up rows, then gate rows).
    pub w1: Vec<f32>,
    /// `[num_experts, hidden, inter]`.
    pub w2: Vec<f32>,
    /// `[num_tokens, top_k]`.
    pub selected: Vec<i32>,
    /// `[num_tokens, top_k]`.
    pub scales: Vec<f32>,
    pub tile1: Tile,
    pub tile2: Tile,
}

/// The buffers of the pipeline, in order: the kernels' parameters index into them.
pub mod buf {
    pub const INPUT: usize = 0;
    pub const SELECTED: usize = 1;
    pub const SCALES: usize = 2;
    pub const W1: usize = 3;
    pub const W2: usize = 4;
    pub const PERMUTED_OF: usize = 5;
    pub const OFFSETS: usize = 6;
    pub const TILES1: usize = 7;
    pub const SCHED1: usize = 8;
    pub const TILES2: usize = 9;
    pub const SCHED2: usize = 10;
    pub const PERMUTED_INPUT: usize = 11;
    pub const H: usize = 12;
    pub const FC2_OUT: usize = 13;
    pub const OUT: usize = 14;
    pub const COUNT: usize = 15;
}

/// One launch of the pipeline: its kernel, launch, and the buffers it takes (in parameter
/// order, followed by `scalars`).
pub struct Step {
    pub kernel: Kernel,
    pub launch: LaunchConfig,
    pub buffers: Vec<usize>,
    pub scalars: Vec<KernelArg>,
}

/// The five launches for `num_tokens` tokens of `shape` in `dtype` with GEMM tiles
/// `tile1`, `tile2`.
pub fn pipeline(dtype: Dtype, shape: MoeShape, num_tokens: usize, tile1: Tile, tile2: Tile) -> Vec<Step> {
    use buf::*;
    let MoeShape { hidden, num_experts, top_k, .. } = shape;
    let rows = num_tokens * top_k;
    let rt = row_threads(hidden) as u32;
    vec![
        Step {
            kernel: routing_kernel(num_experts, top_k, tile1.bm, tile2.bm),
            launch: LaunchConfig::new([1, 1, 1], [ROUTE_THREADS as u32, 1, 1]),
            buffers: vec![SELECTED, PERMUTED_OF, OFFSETS, TILES1, SCHED1, TILES2, SCHED2],
            scalars: vec![KernelArg::ScalarI32(num_tokens as i32)],
        },
        Step {
            kernel: expand_kernel(dtype, hidden, top_k),
            launch: LaunchConfig::new([num_tokens as u32, 1, 1], [rt, 1, 1]),
            buffers: vec![INPUT, PERMUTED_OF, PERMUTED_INPUT],
            scalars: vec![],
        },
        Step {
            kernel: gemm1_kernel(dtype, shape, tile1),
            launch: gemm1_launch(shape, rows, tile1),
            buffers: vec![PERMUTED_INPUT, W1, H, OFFSETS, TILES1, SCHED1],
            scalars: vec![],
        },
        Step {
            kernel: gemm2_kernel(dtype, shape, tile2),
            launch: gemm2_launch(shape, rows, tile2),
            buffers: vec![H, W2, FC2_OUT, OFFSETS, TILES2, SCHED2],
            scalars: vec![],
        },
        Step {
            kernel: finalize_kernel(dtype, hidden, top_k),
            launch: LaunchConfig::new([num_tokens as u32, 1, 1], [rt, 1, 1]),
            buffers: vec![FC2_OUT, PERMUTED_OF, SCALES, OUT],
            scalars: vec![],
        },
    ]
}

/// The sizes in elements of the pipeline's buffers ([`buf`] order).
pub fn buffer_lens(shape: MoeShape, num_tokens: usize, tile1: Tile, tile2: Tile) -> [usize; buf::COUNT] {
    let MoeShape { hidden, inter, num_experts, top_k } = shape;
    let rows = num_tokens * top_k;
    [
        num_tokens * hidden,
        rows,
        rows,
        num_experts * 2 * inter * hidden,
        num_experts * hidden * inter,
        rows,
        num_experts + 1,
        num_experts + 1,
        m_tile_bound(rows, num_experts, tile1.bm),
        num_experts + 1,
        m_tile_bound(rows, num_experts, tile2.bm),
        rows * hidden,
        rows * inter,
        rows * hidden,
        num_tokens * hidden,
    ]
}

impl Problem {
    /// Random inputs, weights scaled so that every stage's values are of order one, and
    /// [`random_routing`].
    pub fn new(dtype: Dtype, shape: MoeShape, num_tokens: usize, seed: u64) -> Problem {
        let MoeShape { hidden, inter, num_experts, top_k } = shape;
        let round = |v: Vec<f32>| v.into_iter().map(|x| dtype.round(x)).collect::<Vec<_>>();
        let x = round(uniform(num_tokens * hidden, seed, 1.0));
        let w1 = round(uniform(num_experts * 2 * inter * hidden, seed + 1, 3.0 / (hidden as f32).sqrt()));
        let w2 = round(uniform(num_experts * hidden * inter, seed + 2, 3.0 / (inter as f32).sqrt()));
        let (selected, scales) = random_routing(num_tokens, num_experts, top_k, seed + 3);
        let (tile1, tile2) = pick_tiles(num_tokens * top_k, num_experts);
        Problem { dtype, shape, num_tokens, x, w1, w2, selected, scales, tile1, tile2 }
    }

    pub fn rows(&self) -> usize {
        self.num_tokens * self.shape.top_k
    }

    pub fn pipeline(&self) -> Vec<Step> {
        pipeline(self.dtype, self.shape, self.num_tokens, self.tile1, self.tile2)
    }

    /// The host buffers ([`buf`] order), outputs and scratch zeroed.
    pub fn buffers(&self) -> Vec<KernelArg> {
        let lens = buffer_lens(self.shape, self.num_tokens, self.tile1, self.tile2);
        let zeros = |n: usize| self.dtype.buffer(&vec![0.0; n]);
        let izeros = |n: usize| KernelArg::buffer(&vec![0i32; n]);
        use buf::*;
        (0..COUNT)
            .map(|i| match i {
                INPUT => self.dtype.buffer(&self.x),
                SELECTED => KernelArg::buffer(&self.selected),
                SCALES => KernelArg::f32(self.scales.clone()),
                W1 => self.dtype.buffer(&self.w1),
                W2 => self.dtype.buffer(&self.w2),
                PERMUTED_INPUT | H | FC2_OUT | OUT => zeros(lens[i]),
                _ => izeros(lens[i]),
            })
            .collect()
    }

    pub fn flops(&self) -> f64 {
        self.shape.flops(self.num_tokens)
    }

    /// The f64 reference `[num_tokens, hidden]`.
    pub fn reference(&self) -> Vec<f64> {
        let MoeShape { hidden, inter, top_k, .. } = self.shape;
        let mut out = vec![0.0f64; self.num_tokens * hidden];
        let mut h = vec![0.0f64; inter];
        for t in 0..self.num_tokens {
            let x = &self.x[t * hidden..(t + 1) * hidden];
            for i in 0..top_k {
                let e = self.selected[t * top_k + i] as usize;
                let s = self.scales[t * top_k + i] as f64;
                let w1 = &self.w1[e * 2 * inter * hidden..(e + 1) * 2 * inter * hidden];
                for (j, hj) in h.iter_mut().enumerate() {
                    let dot = |row: usize| -> f64 {
                        w1[row * hidden..(row + 1) * hidden].iter().zip(x).map(|(&w, &v)| w as f64 * v as f64).sum()
                    };
                    let (up, gate) = (dot(j), dot(inter + j));
                    *hj = gate / (1.0 + (-gate).exp()) * up;
                }
                let w2 = &self.w2[e * hidden * inter..(e + 1) * hidden * inter];
                for n in 0..hidden {
                    let y: f64 = w2[n * inter..(n + 1) * inter].iter().zip(&h).map(|(&w, &v)| w as f64 * v).sum();
                    out[t * hidden + n] += s * y;
                }
            }
        }
        out
    }

    /// The largest error of `got` against the reference in units of the tolerance (two
    /// roundings of the activation type relative to the result and to its RMS): at most 1
    /// passes.
    pub fn check(&self, got: &[f32]) -> f64 {
        let want = self.reference();
        assert_eq!(got.len(), want.len());
        let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len().max(1) as f64).sqrt();
        let rel = 2.0 * self.dtype.tolerance();
        got.iter().zip(&want).map(|(&g, &w)| (g as f64 - w).abs() / (rel * (w.abs() + rms))).fold(0.0, f64::max)
    }
}

/// Run `p`'s pipeline on the simulator (with cute-rs's race and alignment checks) and return
/// [`Problem::check`] of the output.
pub fn simulate(p: &Problem) -> f64 {
    let mut bufs = p.buffers();
    for step in p.pipeline() {
        let mut args: Vec<KernelArg> = step.buffers.iter().map(|&b| bufs[b].clone()).collect();
        args.extend(step.scalars.iter().cloned());
        step.kernel.simulate(step.launch, &mut args).unwrap_or_else(|e| panic!("{}: {e}", step.kernel.name()));
        for (a, &b) in args.into_iter().zip(&step.buffers) {
            bufs[b] = a;
        }
    }
    p.check(&p.dtype.read(&bufs[buf::OUT]))
}

#[cfg(feature = "cuda")]
pub mod gpu {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::error::Error;
    use std::rc::Rc;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, DeviceBuffer};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Dtype, MoeShape, Problem, RouteProblem, Tile, buf, buffer_lens, pipeline};

    type Loaded<'c> = Rc<(Kernel, CudaFunction<'c>)>;

    /// The pipeline's kernels, compiled and loaded when first needed.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: Toolchain,
        opts: CompileOptions,
        cache: RefCell<HashMap<String, Loaded<'c>>>,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(&tc).ok_or("no target for this GPU")?;
            Ok(Kernels { ctx, tc, opts: CompileOptions::new().gpu_arch(arch), cache: RefCell::new(HashMap::new()) })
        }

        fn get(&self, k: Kernel) -> Result<Loaded<'c>, Box<dyn Error>> {
            let name = k.name().to_string();
            if let Some(l) = self.cache.borrow().get(&name) {
                return Ok(l.clone());
            }
            let f = self.ctx.load(&k.compile_with(&self.tc, &self.opts)?)?;
            let l = Rc::new((k, f));
            self.cache.borrow_mut().insert(name, l.clone());
            Ok(l)
        }
    }

    /// Fills `n` elements with pseudo-random values in `[-scale, scale]`. Parameters: `out`,
    /// `n`, `seed`, `scale`.
    fn fill_kernel(dtype: Dtype) -> Kernel {
        fn of<T: Numeric>(dtype: Dtype) -> Kernel {
            Kernel::builder(format!("flashinfer_moe_fill_{}", dtype.name())).block_dim([256, 1, 1]).build(|| {
                let out = param_ptr::<T>("out");
                let n = param_scalar::<i32>("n");
                let seed = param_scalar::<i32>("seed");
                let scale = param_scalar::<f32>("scale");
                let stride = grid_dim().x * 256;
                for_range(block_idx().x * 256 + thread_idx().x, n, stride, |i| {
                    let mut v = (i ^ seed).to::<u32>() * Uint32::from(0x9E37_79B1u32);
                    v = v ^ (v >> Uint32::from(15u32));
                    v = v * Uint32::from(0x85EB_CA77u32);
                    v = v ^ (v >> Uint32::from(13u32));
                    let u = (v >> Uint32::from(8u32)).to::<f32>() * Float32::from(1.0f32 / (1u32 << 23) as f32);
                    out.store_at(i, ((u - Float32::from(1.0f32)) * scale).to::<T>());
                });
            })
        }
        match dtype {
            Dtype::F16 => of::<F16>(dtype),
            Dtype::BF16 => of::<BF16>(dtype),
        }
    }

    fn read(b: &DeviceBuffer<'_>, dtype: Dtype) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(match dtype {
            Dtype::F16 => b.to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
            Dtype::BF16 => b.to_vec::<u16>()?.into_iter().map(|x| f32::from_bits((x as u32) << 16)).collect(),
        })
    }

    /// Run `p` on the GPU once and return [`Problem::check`] of the output.
    pub fn check(ks: &Kernels<'_>, p: &Problem) -> Result<f64, Box<dyn Error>> {
        let host = p.buffers();
        let bufs: Vec<DeviceBuffer<'_>> = host.iter().map(|a| ks.ctx.upload(a)).collect::<Result<_, _>>()?;
        let views: Vec<&DeviceBuffer<'_>> = bufs.iter().collect();
        launch_views(ks, p.dtype, p.shape, p.num_tokens, (p.tile1, p.tile2), &views, None)?;
        Ok(p.check(&read(&bufs[buf::OUT], p.dtype)?))
    }

    /// Device-resident weights of a shape, filled with pseudo-random values.
    pub struct Weights<'c> {
        pub w1: DeviceBuffer<'c>,
        pub w2: DeviceBuffer<'c>,
    }

    pub fn random_weights<'c>(ks: &Kernels<'c>, dtype: Dtype, shape: MoeShape) -> Result<Weights<'c>, Box<dyn Error>> {
        let ctx = ks.ctx;
        let fill = ks.get(fill_kernel(dtype))?;
        let MoeShape { hidden, inter, num_experts, .. } = shape;
        let mut out = Vec::new();
        for (n, scale, seed) in [
            (num_experts * 2 * inter * hidden, 1.0 / (hidden as f32).sqrt(), 1),
            (num_experts * hidden * inter, 1.0 / (inter as f32).sqrt(), 2),
        ] {
            let b = ctx.alloc_bytes(n * 2)?;
            let args = [
                DeviceArg::Buffer(&b),
                DeviceArg::Scalar(KernelArg::ScalarI32(n as i32)),
                DeviceArg::Scalar(KernelArg::ScalarI32(seed)),
                DeviceArg::Scalar(KernelArg::ScalarF32(scale)),
            ];
            let stream = ctx.stream()?;
            ctx.launch_builder(&fill.0, &fill.1, LaunchConfig::new([22 * 64, 1, 1], [256, 1, 1]))
                .stream(&stream)
                .args(&args)
                .submit()?;
            stream.synchronize()?;
            out.push(b);
        }
        let w2 = out.pop().unwrap();
        let w1 = out.pop().unwrap();
        Ok(Weights { w1, w2 })
    }

    /// Time `num_tokens` tokens of `shape` on `weights` (random inputs and
    /// [`super::random_routing`]); returns the CUDA-graph GPU time in microseconds.
    pub fn bench(
        ks: &Kernels<'_>,
        dtype: Dtype,
        shape: MoeShape,
        weights: &Weights<'_>,
        num_tokens: usize,
        tiles: (Tile, Tile),
        iterations: usize,
    ) -> Result<f64, Box<dyn Error>> {
        let ctx = ks.ctx;
        let lens = buffer_lens(shape, num_tokens, tiles.0, tiles.1);
        let (sel, scales) = super::random_routing(num_tokens, shape.num_experts, shape.top_k, 7);
        let x: Vec<f32> = (0..lens[buf::INPUT]).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut bufs = Vec::new();
        for (i, &len) in lens.iter().enumerate() {
            let b = match i {
                buf::W1 | buf::W2 => None,
                buf::INPUT => Some(ctx.upload(&dtype.buffer(&x))?),
                buf::SELECTED => Some(ctx.upload_slice(&sel)?),
                buf::SCALES => Some(ctx.upload_slice(&scales)?),
                buf::PERMUTED_INPUT | buf::H | buf::FC2_OUT | buf::OUT => Some(ctx.alloc_bytes(len * 2)?),
                _ => Some(ctx.alloc_bytes(len * 4)?),
            };
            bufs.push(b);
        }
        // The weights are borrowed: launch with views of them in their slots.
        let views: Vec<&DeviceBuffer<'_>> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| match i {
                buf::W1 => &weights.w1,
                buf::W2 => &weights.w2,
                _ => b.as_ref().unwrap(),
            })
            .collect();
        let t = launch_views(ks, dtype, shape, num_tokens, tiles, &views, Some(iterations))?;
        Ok(t.unwrap())
    }

    /// Run the pipeline on `bufs` (device buffers in [`buf`] order) once; if `time`, return
    /// the average GPU time in microseconds of `iterations` CUDA-graph replays. The
    /// environment variable `MOE_STEPS=0,2,..` limits the timed launches to those steps of
    /// [`pipeline`] (for a per-kernel breakdown).
    fn launch_views(
        ks: &Kernels<'_>,
        dtype: Dtype,
        shape: MoeShape,
        num_tokens: usize,
        tiles: (Tile, Tile),
        bufs: &[&DeviceBuffer<'_>],
        time: Option<usize>,
    ) -> Result<Option<f64>, Box<dyn Error>> {
        let steps = pipeline(dtype, shape, num_tokens, tiles.0, tiles.1);
        let mut loaded = Vec::new();
        for s in steps {
            let l = ks.get(s.kernel)?;
            let mut args: Vec<DeviceArg> = s.buffers.iter().map(|&b| DeviceArg::Buffer(bufs[b])).collect();
            args.extend(s.scalars.into_iter().map(DeviceArg::Scalar));
            loaded.push((l, s.launch, args));
        }
        let ctx = ks.ctx;
        let only: Option<Vec<usize>> =
            std::env::var("MOE_STEPS").ok().map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
        let run = |st: &cute::cuda::CudaStream<'_>, all: bool| -> Result<(), Box<dyn Error>> {
            for (i, (l, launch, args)) in loaded.iter().enumerate() {
                if all || only.as_ref().is_none_or(|o| o.contains(&i)) {
                    ctx.launch_builder(&l.0, &l.1, *launch).stream(st).args(args).submit()?;
                }
            }
            Ok(())
        };
        let stream = ctx.stream()?;
        run(&stream, true)?;
        stream.synchronize()?;
        Ok(match time {
            Some(iterations) => {
                let opts = BenchmarkOptions::new().warmup_iterations(3).iterations(iterations).use_cuda_graphs(true);
                Some(benchmark(&stream, &opts, &mut |st: &cute::cuda::CudaStream<'_>, _w: usize| run(st, false))?)
            }
            None => None,
        })
    }

    /// Run routing problem `p` on the GPU once; returns [`RouteProblem::check`] and, if
    /// `time`, the CUDA-graph GPU time in microseconds.
    pub fn route(
        ks: &Kernels<'_>,
        p: &RouteProblem,
        check: bool,
        time: bool,
    ) -> Result<(f64, Option<f64>), Box<dyn Error>> {
        let ctx = ks.ctx;
        let l = ks.get(p.kernel())?;
        let host = p.args();
        let bufs: Vec<DeviceBuffer<'_>> = host[..4].iter().map(|a| ctx.upload(a)).collect::<Result<_, _>>()?;
        let mut args: Vec<DeviceArg> = bufs.iter().map(DeviceArg::Buffer).collect();
        args.extend(host[4..].iter().cloned().map(DeviceArg::Scalar));
        let launch = p.launch();
        let mut go = |st: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&l.0, &l.1, launch).stream(st).args(&args).submit()?;
            Ok(())
        };
        let stream = ctx.stream()?;
        go(&stream, 0)?;
        stream.synchronize()?;
        let err = if check { p.check(&bufs[2].to_vec::<i32>()?, &bufs[3].to_vec::<f32>()?) } else { 0.0 };
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut go)?)
        } else {
            None
        };
        Ok((err, t))
    }
}

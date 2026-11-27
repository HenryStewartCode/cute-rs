// SPDX-License-Identifier: Apache-2.0

//! `merge_state`, `merge_state_in_place` and `merge_states`, after FlashInfer's
//! `MergeStateKernel`, `MergeStateInPlaceKernel`, `MergeStatesKernel` and
//! `MergeStatesLargeNumIndexSetsKernel` (`include/flashinfer/attention/cascade.cuh`) and
//! their launchers. (`decode::merge_kernel` is the variable-length merge
//! of split-KV states with one thread per element; these are FlashInfer's vectorised
//! public merges.)
//!
//! A state is an attention output `v` (`[seq, heads, head_dim]`, Float16 or BFloat16) and
//! its log-sum-exp `s` (`[seq, heads]`, Float32, base 2). Merging states `i` gives `v =
//! sum_i 2^(s_i - max s) v_i / sum_i 2^(s_i - max s)` and `s = max s + log2 sum_i 2^(s_i -
//! max s)`; an empty state (`s = -inf`) weighs nothing, and merging only empty states gives
//! `v = 0, s = -inf`.
//!
//! * **Geometry**: 16-byte vectors (8 elements), `bdx = head_dim / 8` threads along a head
//!   and one `threadIdx.y` row per head (`block = (bdx, heads)`, so `heads * head_dim / 8
//!   <= 1024`, FlashInfer's limit too), one block per sequence position; Float32
//!   arithmetic, `ex2.approx.ftz` and `lg2.approx.ftz`.
//! * **[`merge_state_kernel`]**: `(v_a, s_a, v_b, s_b) -> (v, s)`. **[`merge_state_in_place_kernel`]**:
//!   `(v, s) <- merge((v, s), (v_other, s_other))`, optionally skipping the positions whose
//!   `mask` byte is 0.
//! * **[`merge_states_kernel`]**: `[seq, sets, heads, head_dim]` to `[seq, heads, head_dim]`
//!   when `sets < seq`: each thread folds its `sets` states in order, two per step with the
//!   loads issued together. **[`merge_states_large_kernel`]** when `sets >= seq`: grid
//!   `(seq, heads)`, 128 threads as `(bdx, 128 / bdx)`, each `ty` row takes every
//!   `(128 / bdx)`th state through a four-stage `cp.async` pipeline of shared memory, the
//!   `S` values staged through shared memory, and the rows' partial states are merged at
//!   the end through shared memory (partial outputs held in the input type there, as in
//!   FlashInfer).
//!
//! Not yet covered: head dimensions other than 64, 128 and 256 (FlashInfer widens the
//! vector to `head_dim / 32` beyond 256), the variable-length `VariableLengthMergeStates`
//! (`decode::merge_kernel`'s kind) and `AttentionSum`, `heads` times `head_dim / 8` above
//! 1024 threads, and a null `s_merged`. Merging a single non-empty state is the same as
//! FlashInfer's copy, but a single state with `s = -inf` gives 0, not a copy of `v`.
//!
//! `examples/cascade.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`,
//! on the GPU against a reference, then benchmarks it.

use cute::prelude::*;

pub use crate::norm::Dtype;

/// Elements per thread per vector access (16 bytes of a 16-bit type).
pub const VEC: usize = 8;
/// Threads of the large-`sets` kernel's block.
pub const LARGE_THREADS: usize = 128;
/// `cp.async` stages of the large-`sets` kernel.
pub const STAGES: usize = 4;

/// A thread's running state: `m` (the largest `s` so far), `d` (the sum of
/// `2^(s - m)`, the merged states counting 1 each) and the weighted sum `o`.
struct State {
    m: Var<Float32>,
    d: Var<Float32>,
    o: Vec<Var<Float32>>,
}

impl State {
    fn new() -> State {
        State {
            m: Var::new(Float32::from(f32::NEG_INFINITY)),
            d: Var::new(Float32::from(1.0f32)),
            o: (0..VEC).map(|_| Var::new(Float32::from(0.0f32))).collect(),
        }
    }

    fn reset(&self) {
        self.m.set(Float32::from(f32::NEG_INFINITY));
        self.d.set(Float32::from(1.0f32));
        for o in &self.o {
            o.set(Float32::from(0.0f32));
        }
    }

    /// Merge the state `(v, s)` with weight 1. `fmax` drops the NaN of `-inf - -inf`.
    fn merge(&self, v: &[Float32], s: Float32) {
        let (m_prev, d_prev) = (self.m.get(), self.d.get());
        let m = fmax(m_prev, s);
        let neg_inf = Float32::from(f32::NEG_INFINITY);
        let f1 = cute::math::exp2(fmax(m_prev - m, neg_inf), true);
        let f2 = cute::math::exp2(fmax(s - m, neg_inf), true);
        self.d.set(d_prev * f1 + f2);
        for (o, v) in self.o.iter().zip(v) {
            o.set(o.get() * f1 + *v * f2);
        }
        self.m.set(m);
    }

    /// Divide by `d` (an empty state, `d = 0`, stays zero).
    fn normalize(&self) {
        let d = self.d.get();
        let rcp = select(d.gt(0.0f32), cute::math::rcp(d, true), Float32::from(0.0f32));
        for o in &self.o {
            o.set(o.get() * rcp);
        }
    }

    fn lse(&self) -> Float32 {
        self.m.get() + cute::math::log2(self.d.get(), true)
    }
}

/// The weights `(2^(sa - m) / d, 2^(sb - m) / d)` of two states and `m + log2 d`: an empty
/// pair weighs 0.
fn pair_weights(sa: Float32, sb: Float32) -> (Float32, Float32, Float32) {
    let neg_inf = Float32::from(f32::NEG_INFINITY);
    let m = fmax(sa, sb);
    let wa = cute::math::exp2(fmax(sa - m, neg_inf), true);
    let wb = cute::math::exp2(fmax(sb - m, neg_inf), true);
    let d = wa + wb;
    let zero = Float32::from(0.0f32);
    let ok = d.gt(0.0f32);
    (
        select(ok, cute::math::div(wa, d, true), zero),
        select(ok, cute::math::div(wb, d, true), zero),
        cute::math::log2(d, true) + m,
    )
}

fn convert<T: Numeric>(v: &[Value<T>; VEC]) -> Vec<Float32> {
    v.iter().map(|x| x.to::<f32>()).collect()
}

fn check_head(head_dim: usize, heads: usize) -> usize {
    assert!(matches!(head_dim, 64 | 128 | 256), "head_dim = {head_dim}: only 64, 128 and 256");
    let bdx = head_dim / VEC;
    assert!(heads >= 1 && bdx * heads <= 1024, "heads = {heads} * {bdx} threads exceeds 1024");
    bdx
}

/// `merge_state`: parameters `v_a, s_a, v_b, s_b, v_merged, s_merged`; launch `seq` blocks
/// of `(head_dim / 8, heads)` threads.
pub fn merge_state_kernel(dtype: Dtype, head_dim: usize, heads: usize) -> Kernel {
    match dtype {
        Dtype::F16 => merge_state_of::<F16>(dtype, head_dim, heads),
        Dtype::BF16 => merge_state_of::<BF16>(dtype, head_dim, heads),
    }
}

fn merge_state_of<T: Numeric>(dtype: Dtype, head_dim: usize, heads: usize) -> Kernel {
    let bdx = check_head(head_dim, heads);
    let name = format!("flashinfer_merge_state_{}_d{head_dim}_h{heads}", dtype.name());
    Kernel::builder(&name).block_dim([bdx as u32, heads as u32, 1]).build(|| {
        let v_a = param_ptr::<T>("v_a");
        let s_a = param_ptr::<f32>("s_a");
        let v_b = param_ptr::<T>("v_b");
        let s_b = param_ptr::<f32>("s_b");
        let v_m = param_ptr::<T>("v_merged");
        let s_m = param_ptr::<f32>("s_merged");
        let (tx, head, pos) = (thread_idx().x, thread_idx().y, block_idx().x);
        let at = pos * heads as i32 + head;
        let (wa, wb, lse) = pair_weights(s_a.load_at(at), s_b.load_at(at));
        let off = (at * bdx as i32 + tx) * VEC as i32;
        let a = convert(&load_vector::<T, VEC>(&v_a.offset(off), LoadOptions::default()));
        let b = convert(&load_vector::<T, VEC>(&v_b.offset(off), LoadOptions::default()));
        let out: [Value<T>; VEC] = std::array::from_fn(|i| (wa * a[i] + wb * b[i]).to::<T>());
        store_vector::<T, VEC>(&v_m.offset(off), out, StoreOptions::default());
        if_(tx.eq(0), || s_m.store_at(at, lse));
    })
}

/// `merge_state_in_place`: parameters `v, s, v_other, s_other` and, if `masked`, `mask`
/// (one byte per position; 0 leaves the position unchanged); same launch as
/// [`merge_state_kernel`].
pub fn merge_state_in_place_kernel(dtype: Dtype, head_dim: usize, heads: usize, masked: bool) -> Kernel {
    match dtype {
        Dtype::F16 => in_place_of::<F16>(dtype, head_dim, heads, masked),
        Dtype::BF16 => in_place_of::<BF16>(dtype, head_dim, heads, masked),
    }
}

fn in_place_of<T: Numeric>(dtype: Dtype, head_dim: usize, heads: usize, masked: bool) -> Kernel {
    let bdx = check_head(head_dim, heads);
    let tag = if masked { "_masked" } else { "" };
    let name = format!("flashinfer_merge_state_in_place{tag}_{}_d{head_dim}_h{heads}", dtype.name());
    Kernel::builder(&name).block_dim([bdx as u32, heads as u32, 1]).build(|| {
        let v = param_ptr::<T>("v");
        let s = param_ptr::<f32>("s");
        let v_other = param_ptr::<T>("v_other");
        let s_other = param_ptr::<f32>("s_other");
        let mask = masked.then(|| param_ptr::<u8>("mask"));
        let (tx, head, pos) = (thread_idx().x, thread_idx().y, block_idx().x);
        let body = || {
            let at = pos * heads as i32 + head;
            let (w, w_other, lse) = pair_weights(s.load_at(at), s_other.load_at(at));
            let off = (at * bdx as i32 + tx) * VEC as i32;
            let a = convert(&load_vector::<T, VEC>(&v.offset(off), LoadOptions::default()));
            let b = convert(&load_vector::<T, VEC>(&v_other.offset(off), LoadOptions::default()));
            let out: [Value<T>; VEC] = std::array::from_fn(|i| (w * a[i] + w_other * b[i]).to::<T>());
            store_vector::<T, VEC>(&v.offset(off), out, StoreOptions::default());
            if_(tx.eq(0), || s.store_at(at, lse));
        };
        match &mask {
            Some(mask) => if_(mask.load_at(pos).ne(0u8), body),
            None => body(),
        }
    })
}

/// `merge_states` for `sets < seq`: parameters `v, s, v_merged, s_merged, sets` with `v`
/// `[seq, sets, heads, head_dim]` and `s` `[seq, sets, heads]`; launch as
/// [`merge_state_kernel`].
pub fn merge_states_kernel(dtype: Dtype, head_dim: usize, heads: usize) -> Kernel {
    match dtype {
        Dtype::F16 => merge_states_of::<F16>(dtype, head_dim, heads),
        Dtype::BF16 => merge_states_of::<BF16>(dtype, head_dim, heads),
    }
}

fn merge_states_of<T: Numeric>(dtype: Dtype, head_dim: usize, heads: usize) -> Kernel {
    let bdx = check_head(head_dim, heads);
    let name = format!("flashinfer_merge_states_{}_d{head_dim}_h{heads}", dtype.name());
    Kernel::builder(&name).block_dim([bdx as u32, heads as u32, 1]).build(|| {
        let v = param_ptr::<T>("v");
        let s = param_ptr::<f32>("s");
        let v_m = param_ptr::<T>("v_merged");
        let s_m = param_ptr::<f32>("s_merged");
        let n = param_scalar::<i32>("sets");
        let (tx, head, pos) = (thread_idx().x, thread_idx().y, block_idx().x);
        let (h, v8) = (heads as i32, VEC as i32);
        let st = State::new();
        // Two states per step, their loads issued together; a second state past the end
        // reads the last one and has `s = -inf`, which weighs nothing.
        let steps = (n + 1) / 2;
        for_range(0, steps, 1, |step| {
            let loads: Vec<(Float32, Vec<Float32>)> = (0..2)
                .map(|k| {
                    let i = step * 2 + k;
                    let at = (pos * n + i.min(n - 1)) * h + head;
                    let sv = s.load_at(at);
                    let x =
                        convert(&load_vector::<T, VEC>(&v.offset((at * bdx as i32 + tx) * v8), LoadOptions::default()));
                    (select(i.lt(n), sv, Float32::from(f32::NEG_INFINITY)), x)
                })
                .collect();
            for (sv, x) in &loads {
                st.merge(x, *sv);
            }
        });
        st.normalize();
        let at = pos * h + head;
        let out: [Value<T>; VEC] = std::array::from_fn(|i| st.o[i].get().to::<T>());
        store_vector::<T, VEC>(&v_m.offset((at * bdx as i32 + tx) * v8), out, StoreOptions::default());
        if_(tx.eq(0), || s_m.store_at(at, st.lse()));
    })
}

/// `merge_states` for `sets >= seq`: parameters `v, s, v_merged, s_merged, sets,
/// num_heads`; launch grid `(seq, heads)`, block `(head_dim / 8, 128 / (head_dim / 8))`.
pub fn merge_states_large_kernel(dtype: Dtype, head_dim: usize) -> Kernel {
    match dtype {
        Dtype::F16 => large_of::<F16>(dtype, head_dim),
        Dtype::BF16 => large_of::<BF16>(dtype, head_dim),
    }
}

fn large_of<T: Numeric>(dtype: Dtype, head_dim: usize) -> Kernel {
    let bdx = check_head(head_dim, 1);
    let bdy = LARGE_THREADS / bdx;
    let name = format!("flashinfer_merge_states_large_{}_d{head_dim}", dtype.name());
    Kernel::builder(&name).block_dim([bdx as u32, bdy as u32, 1]).build(|| {
        let v = param_ptr::<T>("v");
        let s = param_ptr::<f32>("s");
        let v_m = param_ptr::<T>("v_merged");
        let s_m = param_ptr::<f32>("s_merged");
        let n = param_scalar::<i32>("sets");
        let h = param_scalar::<i32>("num_heads");
        let (tx, ty) = (thread_idx().x, thread_idx().y);
        let (pos, head) = (block_idx().x, block_idx().y);
        let (bdx_i, bdy_i, v8) = (bdx as i32, bdy as i32, VEC as i32);
        let stage_vecs = bdy * bdx; // 16-byte vectors per stage

        let v_smem = make_smem_tensor_aligned::<T>(Layout::col_major(STAGES * stage_vecs * VEC), 16);
        let s_smem = make_smem_tensor::<f32>(Layout::col_major(LARGE_THREADS));
        let atom = make_copy_atom::<T>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        const BOUND: usize = 1 << 30;
        let g = make_tensor(&v, Layout::col_major((VEC, BOUND / VEC)));
        let sv = make_tensor(&v_smem.ptr(), Layout::col_major((VEC, STAGES * stage_vecs)));
        // Stage the states `it * bdy + ty` into `slot`; one past the last reads the last
        // state (a valid address), never used.
        let load = |it: Int32, slot: Int32| {
            let set = (it * bdy_i + ty).min(n - 1);
            let gvec = ((pos * n + set) * h + head) * bdx_i + tx;
            let svec = (slot * bdy_i + ty) * bdx_i + tx;
            copy(&atom, &g.slice((.., gvec)), &sv.slice((.., svec)));
        };
        for it in 0..STAGES {
            load(Int32::from(it as i32), Int32::from(it as i32));
            cp_async_commit_group();
        }
        let st = State::new();
        let iters = (n + (bdy_i - 1)) / bdy_i;
        for_range(0, iters, 1, |iter| {
            // The log-sum-exps of the next `bdx * bdy` states, staged by all threads.
            if_((iter % bdx_i).eq(0), || {
                let idx = iter * bdy_i + ty * bdx_i + tx;
                let val = if_else_yield(idx.lt(n), || s.load_at((pos * n + idx) * h + head), || Float32::from(0.0f32));
                s_smem.store(ty * bdx_i + tx, val);
            });
            cp_async_wait_group(STAGES as u32 - 1);
            sync_threads();
            let slot = iter % STAGES as i32;
            let x = convert(&load_vector::<T, VEC>(
                &v_smem.ptr().offset(((slot * bdy_i + ty) * bdx_i + tx) * v8),
                LoadOptions::default(),
            ));
            if_((iter * bdy_i + ty).lt(n), || st.merge(&x, s_smem.load((iter % bdx_i) * bdy_i + ty)));
            sync_threads();
            load(iter + STAGES as i32, slot);
            cp_async_commit_group();
        });
        cp_async_wait_group(0);
        sync_threads();

        // Merge the `bdy` rows' states; the partial outputs go through shared memory in
        // the input type.
        st.normalize();
        let mine: [Value<T>; VEC] = std::array::from_fn(|i| st.o[i].get().to::<T>());
        store_vector::<T, VEC>(&v_smem.ptr().offset((ty * bdx_i + tx) * v8), mine, StoreOptions::default());
        if_(tx.eq(0), || s_smem.store(ty, st.lse()));
        st.reset();
        sync_threads();
        for r in 0..bdy {
            let x = convert(&load_vector::<T, VEC>(
                &v_smem.ptr().offset((r as i32 * bdx_i + tx) * v8),
                LoadOptions::default(),
            ));
            st.merge(&x, s_smem.load(r as i32));
        }
        st.normalize();
        let at = pos * h + head;
        let out: [Value<T>; VEC] = std::array::from_fn(|i| st.o[i].get().to::<T>());
        store_vector::<T, VEC>(&v_m.offset((at * bdx_i + tx) * v8), out, StoreOptions::default());
        if_(tx.eq(0) & ty.eq(0), || s_m.store_at(at, st.lse()));
    })
}

/// Which entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// `merge_state` of sets 0 and 1.
    MergeState,
    /// `merge_state_in_place` of set 0 with set 1, optionally masked.
    InPlace { masked: bool },
    /// `merge_states` of all the sets; picks the large-`sets` kernel when `sets >= seq`.
    MergeStates,
}

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::MergeState => "merge_state",
            Op::InPlace { masked: false } => "merge_state_in_place",
            Op::InPlace { masked: true } => "merge_state_in_place (mask)",
            Op::MergeStates => "merge_states",
        }
    }
}

/// A test problem: `sets` states for each of `seq` positions and `heads` heads, `v`
/// `[seq, sets, heads, head_dim]` (every value exactly representable in `dtype`) and `s`
/// `[seq, sets, heads]`, with about one state in 16 empty (`s = -inf`).
pub struct Problem {
    pub dtype: Dtype,
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub sets: usize,
    pub v: Vec<f32>,
    pub s: Vec<f32>,
    /// One byte per position, for the masked in-place merge (about one in 3 is 0).
    pub mask: Vec<u8>,
}

impl Problem {
    pub fn new(dtype: Dtype, seq: usize, heads: usize, head_dim: usize, sets: usize, seed: u64) -> Problem {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f32 / (1u64 << 31) as f32
        };
        let v = (0..seq * sets * heads * head_dim).map(|_| dtype.round((next() * 2.0 - 1.0) * 2.0)).collect();
        let s = (0..seq * sets * heads)
            .map(|_| {
                let u = next();
                if u < 1.0 / 16.0 { f32::NEG_INFINITY } else { (next() * 2.0 - 1.0) * 6.0 }
            })
            .collect();
        let mask = (0..seq).map(|_| (next() >= 1.0 / 3.0) as u8).collect();
        Problem { dtype, seq, heads, head_dim, sets, v, s, mask }
    }

    /// Whether `merge_states` takes the large-`sets` kernel.
    pub fn large(&self) -> bool {
        self.sets >= self.seq
    }

    pub fn kernel(&self, op: Op) -> Kernel {
        let (t, d, h) = (self.dtype, self.head_dim, self.heads);
        match op {
            Op::MergeState => merge_state_kernel(t, d, h),
            Op::InPlace { masked } => merge_state_in_place_kernel(t, d, h, masked),
            Op::MergeStates if self.large() => merge_states_large_kernel(t, d),
            Op::MergeStates => merge_states_kernel(t, d, h),
        }
    }

    pub fn launch(&self, op: Op) -> LaunchConfig {
        let bdx = (self.head_dim / VEC) as u32;
        match op {
            Op::MergeStates if self.large() => {
                LaunchConfig::new([self.seq as u32, self.heads as u32, 1], [bdx, LARGE_THREADS as u32 / bdx, 1])
            }
            _ => LaunchConfig::new([self.seq as u32, 1, 1], [bdx, self.heads as u32, 1]),
        }
    }

    /// Set `k` of every position: `(v, s)` as `[seq, heads, head_dim]` and `[seq, heads]`.
    fn set(&self, k: usize) -> (Vec<f32>, Vec<f32>) {
        let (hd, h, n) = (self.head_dim, self.heads, self.sets);
        let (mut v, mut s) = (Vec::new(), Vec::new());
        for p in 0..self.seq {
            let at = (p * n + k) * h;
            v.extend_from_slice(&self.v[at * hd..(at + h) * hd]);
            s.extend_from_slice(&self.s[at..at + h]);
        }
        (v, s)
    }

    /// The kernel's arguments (see the kernels), and the positions of the output `v` and
    /// `s` among them.
    pub fn args(&self, op: Op) -> (Vec<KernelArg>, usize, usize) {
        let t = self.dtype;
        let out_v = || t.buffer(&vec![0.0; self.seq * self.heads * self.head_dim]);
        let out_s = || KernelArg::buffer(&vec![0.0f32; self.seq * self.heads]);
        match op {
            Op::MergeState => {
                let ((va, sa), (vb, sb)) = (self.set(0), self.set(1));
                let a = vec![
                    t.buffer(&va),
                    KernelArg::buffer(&sa),
                    t.buffer(&vb),
                    KernelArg::buffer(&sb),
                    out_v(),
                    out_s(),
                ];
                (a, 4, 5)
            }
            Op::InPlace { masked } => {
                let ((va, sa), (vb, sb)) = (self.set(0), self.set(1));
                let mut a = vec![t.buffer(&va), KernelArg::buffer(&sa), t.buffer(&vb), KernelArg::buffer(&sb)];
                if masked {
                    a.push(KernelArg::buffer(&self.mask));
                }
                (a, 0, 1)
            }
            Op::MergeStates => {
                let mut a = vec![t.buffer(&self.v), KernelArg::buffer(&self.s), out_v(), out_s()];
                a.push(KernelArg::ScalarI32(self.sets as i32));
                if self.large() {
                    a.push(KernelArg::ScalarI32(self.heads as i32));
                }
                (a, 2, 3)
            }
        }
    }

    /// The f64 reference: merged `v` (`[seq, heads, head_dim]`) and `s` (`[seq, heads]`).
    pub fn reference(&self, op: Op) -> (Vec<f64>, Vec<f64>) {
        let (hd, h, n) = (self.head_dim, self.heads, self.sets);
        let used = if op == Op::MergeStates { n } else { 2 };
        let (mut out_v, mut out_s) = (Vec::new(), Vec::new());
        for p in 0..self.seq {
            for head in 0..h {
                let at = |k: usize| (p * n + k) * h + head;
                if let Op::InPlace { masked: true } = op
                    && self.mask[p] == 0
                {
                    out_v.extend(self.v[at(0) * hd..(at(0) + 1) * hd].iter().map(|&x| x as f64));
                    out_s.push(self.s[at(0)] as f64);
                    continue;
                }

                let m = (0..used).map(|k| self.s[at(k)] as f64).fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = (0..used).map(|k| (self.s[at(k)] as f64 - m).exp2()).collect();
                let d: f64 = if m == f64::NEG_INFINITY { 0.0 } else { w.iter().sum() };
                for e in 0..hd {
                    let x: f64 = (0..used).map(|k| w[k] * self.v[at(k) * hd + e] as f64).sum();
                    out_v.push(if d > 0.0 { x / d } else { 0.0 });
                }
                out_s.push(if d > 0.0 { m + d.log2() } else { f64::NEG_INFINITY });
            }
        }
        (out_v, out_s)
    }

    /// Check the outputs against the reference (the scaled error of `v` against
    /// [`Dtype::tolerance`], the log-sum-exp to `2e-5` of `1 + |s|`; `-inf` exactly);
    /// returns the largest scaled errors `(v, s)`. The large-`sets` kernel also rounds its
    /// partial outputs to the type, so it gets twice the tolerance.
    pub fn check(&self, op: Op, got_v: &[f32], got_s: &[f32]) -> (f64, f64) {
        let (want_v, want_s) = self.reference(op);
        let tol = self.dtype.tolerance() * if op == Op::MergeStates && self.large() { 2.0 } else { 1.0 };
        let mut worst = (0.0f64, 0.0f64);
        for (i, (&g, &r)) in got_v.iter().zip(&want_v).enumerate() {
            let e = (g as f64 - r).abs() / (1.0 + r.abs());
            assert!(e <= tol, "{}: v element {i}: {g} against {r} ({:?})", op.name(), self.shape());
            worst.0 = worst.0.max(e);
        }
        for (i, (&g, &r)) in got_s.iter().zip(&want_s).enumerate() {
            if r == f64::NEG_INFINITY {
                assert!(g == f32::NEG_INFINITY, "{}: lse {i}: {g} against -inf ({:?})", op.name(), self.shape());
                continue;
            }
            let e = (g as f64 - r).abs() / (1.0 + r.abs());
            assert!(e <= 2e-5, "{}: lse {i}: {g} against {r} ({:?})", op.name(), self.shape());
            worst.1 = worst.1.max(e);
        }
        worst
    }

    fn shape(&self) -> (usize, usize, usize, usize) {
        (self.seq, self.heads, self.head_dim, self.sets)
    }

    /// Bytes read and written by `op`.
    pub fn bytes(&self, op: Op) -> f64 {
        let (v, s) = ((self.seq * self.heads * self.head_dim * 2) as f64, (self.seq * self.heads * 4) as f64);
        match op {
            Op::MergeState => 3.0 * v + 3.0 * s,
            Op::InPlace { .. } => 3.0 * v + 3.0 * s,
            Op::MergeStates => (self.sets as f64 + 1.0) * (v + s),
        }
    }
}

/// Simulate `op` on `p` and check it against the f64 reference; returns the largest scaled
/// errors `(v, s)`.
pub fn simulate(p: &Problem, op: Op) -> (f64, f64) {
    let k = p.kernel(op);
    let (mut args, vi, si) = p.args(op);
    k.simulate(p.launch(op), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    p.check(op, &p.dtype.read(&args[vi]), &args[si].to_vec::<f32>())
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::collections::HashMap;
    use std::error::Error;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Dtype, Op, Problem};

    /// Kernels compiled and loaded on first use, per kernel name.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: &'c Toolchain,
        loaded: HashMap<String, (Kernel, CudaFunction<'c>)>,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: &'c Toolchain) -> Kernels<'c> {
            Kernels { ctx, tc, loaded: HashMap::new() }
        }

        pub fn get(&mut self, p: &Problem, op: Op) -> Result<&(Kernel, CudaFunction<'c>), Box<dyn Error>> {
            let k = p.kernel(op);
            let key = k.name().to_string();
            if !self.loaded.contains_key(&key) {
                let arch = self.ctx.best_target(self.tc).ok_or("no target for this GPU")?;
                let f = self.ctx.load(&k.compile_with(self.tc, &CompileOptions::new().gpu_arch(arch))?)?;
                self.loaded.insert(key.clone(), (k, f));
            }
            Ok(&self.loaded[&key])
        }
    }

    /// Run `op` on `p` once; if `check`, compare with the f64 reference and return the
    /// largest scaled errors; if `time`, also return the average GPU time in microseconds of
    /// launches replayed from a CUDA graph (the in-place merge, which changes its inputs,
    /// is timed as it is: each replay merges again).
    #[allow(clippy::type_complexity)]
    pub fn run(
        ctx: &CudaContext,
        ks: &mut Kernels<'_>,
        p: &Problem,
        op: Op,
        check: bool,
        time: bool,
    ) -> Result<((f64, f64), Option<f64>), Box<dyn Error>> {
        let (k, f) = ks.get(p, op)?;
        let (host, vi, si) = p.args(op);
        let mut bufs = Vec::new();
        for a in &host {
            if !matches!(a, KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_)) {
                bufs.push(ctx.upload(a)?);
            }
        }
        let mut bi = bufs.iter();
        let args: Vec<DeviceArg> = host
            .iter()
            .map(|a| match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => DeviceArg::Scalar(a.clone()),
                _ => DeviceArg::Buffer(bi.next().unwrap()),
            })
            .collect();
        let stream = ctx.stream()?;
        let mut launch = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(k, f, p.launch(op)).stream(s).args(&args).submit()?;
            Ok(())
        };
        launch(&stream, 0)?;
        stream.synchronize()?;
        let mut err = (0.0, 0.0);
        if check {
            let v: Vec<f32> = match p.dtype {
                Dtype::F16 => bufs[vi].to_vec::<F16>()?.into_iter().map(|x| x.to_f32()).collect(),
                Dtype::BF16 => {
                    bufs[vi].to_vec::<u16>()?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
                }
            };
            err = p.check(op, &v, &bufs[si].to_vec::<f32>()?);
        }
        let t = if time {
            let opts = BenchmarkOptions::new().warmup_iterations(10).iterations(100).use_cuda_graphs(true);
            Some(benchmark(&stream, &opts, &mut launch)?)
        } else {
            None
        };
        Ok((err, t))
    }
}

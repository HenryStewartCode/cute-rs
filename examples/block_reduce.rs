// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/reduction/block_smem_reduce.py
//   (the kernel `_kernel`, its helpers and the host data of `run`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A thread block reducing a tile through shared memory, ported from CuTe
//! DSL's `experimental/primitives/reduction/block_smem_reduce.py`.
//!
//! Each block owns `ROWS_PER_CTA` rows of `THREADS * ITEMS_PER_LANE`
//! elements and reduces each row (`REDUCE_OP`: add, max or min) in `DTYPE`
//! (Int32, Float32, Float16 or BFloat16):
//!
//! * each thread loads its `ITEMS_PER_LANE` contiguous elements of the row
//!   into a register fragment and folds them (the example's `Vector.reduce`);
//! * each warp reduces its lanes' values: `warp_redux_sync` for Int32
//!   (`redux.sync`; the example's `prims.redux_sync`), a butterfly of
//!   `shuffle_sync_bfly` otherwise (`redux.sync.f32` needs sm_100a);
//! * lane 0 of each warp combines the warp's value into the row's
//!   shared-memory cell: `red` (`red.relaxed.cta.shared`, the example's
//!   `prims.red`) for Int32 and for the float sums, `atomic_fmax` or
//!   `atomic_fmin` for Float32 max and min; PTX has no scalar
//!   `red.max.f16`, so Float16 and BFloat16 max and min store each warp's
//!   value and combine them after the barrier instead;
//! * after a barrier (`sync_threads`, the example's
//!   `prims.barrier_cta_sync(0)`) one thread per row stores the result, in
//!   `DTYPE`.
//!
//! The order in which the warps' `red`s land varies, so a sum is one of the
//! orders' results; the simulator and the GPU each give one of them. The
//! BFloat16 sum needs `red.add.noftz.bf16` (sm_90).
//!
//! Run with `cargo run --release --example block_reduce`.

use cute::prelude::*;

/// Rows per block (`_ROWS_PER_CTA`).
pub const ROWS_PER_CTA: usize = 4;
/// Warps per block (`_NUM_WARPS`).
pub const NUM_WARPS: usize = 4;
/// Threads per block (`_THREADS`).
pub const THREADS: usize = WARP_SIZE * NUM_WARPS;
/// The example's default row tiles (`_DEFAULT_ROW_TILES`) and elements per
/// lane (`_DEFAULT_ITEMS_PER_LANE`).
pub const DEFAULT_ROW_TILES: usize = 4;
pub const DEFAULT_ITEMS_PER_LANE: usize = 4;

/// `REDUCE_OP`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReduceOp {
    Add,
    Max,
    Min,
}

impl ReduceOp {
    pub const ALL: [ReduceOp; 3] = [ReduceOp::Add, ReduceOp::Max, ReduceOp::Min];

    fn name(self) -> &'static str {
        match self {
            ReduceOp::Add => "add",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
        }
    }
}

/// A `DTYPE` of the example, with what the kernel does for it and the same
/// arithmetic on the host.
pub trait ReduceDtype: Arith + PartialEq {
    const NAME: &'static str;
    /// Whether max and min combine warp results in shared memory (Float16
    /// and BFloat16, whose scalar `red` has only add).
    const SMEM_MINMAX: bool;

    /// The identity of `op` (`_identity_value`).
    fn identity(op: ReduceOp) -> Self;

    /// Element `i` of the example's input (`run`): `(i % 37) - 18`, halved
    /// for the floats.
    fn input(i: usize) -> Self;

    /// One warp's reduction of its lanes' values.
    fn warp_reduce(value: Value<Self>, op: ReduceOp) -> Value<Self> {
        warp_reduce_tree(value, op)
    }

    /// Lane 0's contribution to a row's shared-memory cell (`_smem_red`).
    fn smem_red(ptr: &Ptr<Self>, value: Value<Self>, op: ReduceOp);

    /// `op` on the host, with the device's rounding.
    fn host_combine(a: Self, b: Self, op: ReduceOp) -> Self;
}

/// `red` with `op` on a shared-memory cell, relaxed at CTA scope.
fn red_op(op: ReduceOp) -> RedOp {
    match op {
        ReduceOp::Add => RedOp::Add,
        ReduceOp::Max => RedOp::Max,
        ReduceOp::Min => RedOp::Min,
    }
}

fn relaxed_cta() -> AtomicOptions {
    AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Cta)
}

impl ReduceDtype for i32 {
    const NAME: &'static str = "i32";
    const SMEM_MINMAX: bool = false;

    fn identity(op: ReduceOp) -> i32 {
        match op {
            ReduceOp::Add => 0,
            ReduceOp::Max => i32::MIN,
            ReduceOp::Min => i32::MAX,
        }
    }

    fn input(i: usize) -> i32 {
        (i % 37) as i32 - 18
    }

    fn warp_reduce(value: Int32, op: ReduceOp) -> Int32 {
        let kind = match op {
            ReduceOp::Add => ReduxKind::Add,
            ReduceOp::Max => ReduxKind::Max,
            ReduceOp::Min => ReduxKind::Min,
        };
        warp_redux_sync(value, kind)
    }

    fn smem_red(ptr: &Ptr<i32>, value: Int32, op: ReduceOp) {
        red(ptr, value, red_op(op), AtomicType::S32, relaxed_cta());
    }

    fn host_combine(a: i32, b: i32, op: ReduceOp) -> i32 {
        match op {
            ReduceOp::Add => a.wrapping_add(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        }
    }
}

impl ReduceDtype for f32 {
    const NAME: &'static str = "f32";
    const SMEM_MINMAX: bool = false;

    fn identity(op: ReduceOp) -> f32 {
        match op {
            ReduceOp::Add => 0.0,
            ReduceOp::Max => -f32::MAX,
            ReduceOp::Min => f32::MAX,
        }
    }

    fn input(i: usize) -> f32 {
        ((i % 37) as f32 - 18.0) * 0.5
    }

    fn smem_red(ptr: &Ptr<f32>, value: Float32, op: ReduceOp) {
        match op {
            ReduceOp::Add => red(ptr, value, RedOp::Add, AtomicType::F32, relaxed_cta()),
            ReduceOp::Max => {
                let _ = atomic_fmax_with(ptr, value, None, relaxed_cta());
            }
            ReduceOp::Min => {
                let _ = atomic_fmin_with(ptr, value, None, relaxed_cta());
            }
        }
    }

    fn host_combine(a: f32, b: f32, op: ReduceOp) -> f32 {
        match op {
            ReduceOp::Add => a + b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        }
    }
}

impl ReduceDtype for F16 {
    const NAME: &'static str = "f16";
    const SMEM_MINMAX: bool = true;

    fn identity(op: ReduceOp) -> F16 {
        F16::from_f32(match op {
            ReduceOp::Add => 0.0,
            ReduceOp::Max => -65504.0,
            ReduceOp::Min => 65504.0,
        })
    }

    fn input(i: usize) -> F16 {
        F16::from_f32(f32::input(i))
    }

    fn smem_red(ptr: &Ptr<F16>, value: Float16, op: ReduceOp) {
        assert_eq!(op, ReduceOp::Add, "Float16 max and min combine in shared memory");
        red(ptr, value, RedOp::Add, AtomicType::F16, relaxed_cta());
    }

    fn host_combine(a: F16, b: F16, op: ReduceOp) -> F16 {
        let (x, y) = (a.to_f32(), b.to_f32());
        match op {
            // The f32 sum of two Float16 values rounds to the same Float16 as
            // the exact sum.
            ReduceOp::Add => F16::from_f32(x + y),
            ReduceOp::Max => F16::from_f32(x.max(y)),
            ReduceOp::Min => F16::from_f32(x.min(y)),
        }
    }
}

impl ReduceDtype for BF16 {
    const NAME: &'static str = "bf16";
    const SMEM_MINMAX: bool = true;

    fn identity(op: ReduceOp) -> BF16 {
        BF16::from_f32(match op {
            ReduceOp::Add => 0.0,
            ReduceOp::Max => -3.0e38,
            ReduceOp::Min => 3.0e38,
        })
    }

    fn input(i: usize) -> BF16 {
        BF16::from_f32(f32::input(i))
    }

    fn smem_red(ptr: &Ptr<BF16>, value: BFloat16, op: ReduceOp) {
        assert_eq!(op, ReduceOp::Add, "BFloat16 max and min combine in shared memory");
        red(ptr, value, RedOp::Add, AtomicType::Bf16, relaxed_cta());
    }

    fn host_combine(a: BF16, b: BF16, op: ReduceOp) -> BF16 {
        let (x, y) = (a.to_f32(), b.to_f32());
        match op {
            ReduceOp::Add => BF16::from_f32(x + y),
            ReduceOp::Max => BF16::from_f32(x.max(y)),
            ReduceOp::Min => BF16::from_f32(x.min(y)),
        }
    }
}

/// Two values combined with `op` (`_combine_value`).
fn combine<T: ReduceDtype>(lhs: Value<T>, rhs: Value<T>, op: ReduceOp) -> Value<T> {
    match op {
        ReduceOp::Add => lhs + rhs,
        ReduceOp::Max => lhs.max(rhs),
        ReduceOp::Min => lhs.min(rhs),
    }
}

/// A warp's reduction by a butterfly of shuffles (`_warp_reduce_tree`).
fn warp_reduce_tree<T: ReduceDtype>(value: Value<T>, op: ReduceOp) -> Value<T> {
    let mut value = value;
    for offset in [16, 8, 4, 2, 1] {
        let other = shuffle_sync_bfly(value, offset);
        value = combine(value, other, op);
    }
    value
}

/// The kernel (`_kernel`) for `row_tiles` tiles of `ROWS_PER_CTA` rows,
/// `items_per_lane` elements per lane and row, and `op`; launch it with
/// [`launch`].
pub fn build<T: ReduceDtype>(row_tiles: usize, items_per_lane: usize, op: ReduceOp) -> Kernel {
    let rows = row_tiles * ROWS_PER_CTA;
    let row_width = THREADS * items_per_lane;
    // Float16 and BFloat16 max and min keep every warp's result.
    let need_smem_reduction = T::SMEM_MINMAX && op != ReduceOp::Add;
    let name = format!("block_smem_reduce_{}_{}", T::NAME, op.name());
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let src = param_tensor::<T>("src", Layout::row_major((rows, row_width)));
        let out = param_tensor::<T>("out", Layout::col_major(rows));
        let row_tile = block_idx().x;
        let tx = thread_idx().x;
        let warp_idx = tx / WARP_SIZE as i32;
        let lane_idx = tx % WARP_SIZE as i32;
        let smem_buf_elems = if need_smem_reduction { ROWS_PER_CTA * NUM_WARPS } else { ROWS_PER_CTA };
        let smem_buf = make_smem_tensor::<T>(Layout::col_major(smem_buf_elems));
        let row_base = row_tile * ROWS_PER_CTA as i32;
        if !need_smem_reduction {
            if_(tx.lt(ROWS_PER_CTA as i32), || smem_buf.store(tx, T::identity(op)));
            sync_threads();
        }
        for tile_row in 0..ROWS_PER_CTA {
            let row = row_base + tile_row as i32;
            // This thread's contiguous elements of the row, folded locally.
            let values = make_rmem_tensor::<T>(Layout::col_major(items_per_lane));
            for i in 0..items_per_lane {
                values.store(i, src.load((row, tx * items_per_lane as i32 + i as i32)));
            }
            let mut lane_out = values.load(0);
            for i in 1..items_per_lane {
                lane_out = combine(lane_out, values.load(i), op);
            }
            let warp_out = T::warp_reduce(lane_out, op);
            if need_smem_reduction {
                if_(lane_idx.eq(0), || smem_buf.store(warp_idx + (tile_row * NUM_WARPS) as i32, warp_out));
            } else {
                if_(lane_idx.eq(0), || T::smem_red(&smem_buf.ptr().offset(tile_row), warp_out, op));
            }
        }
        // Every warp's contribution lands before a row's result is read.
        sync_threads();
        if_(tx.lt(ROWS_PER_CTA as i32), || {
            let row = row_base + tx;
            if need_smem_reduction {
                let row_smem_base = tx * NUM_WARPS as i32;
                let mut result = smem_buf.load(row_smem_base);
                for warp in 1..NUM_WARPS {
                    result = combine(result, smem_buf.load(row_smem_base + warp as i32), op);
                }
                out.store(row, result);
            } else {
                out.store(row, smem_buf.load(tx));
            }
        });
    })
}

/// One block per row tile (`_host`).
pub fn launch(row_tiles: usize) -> LaunchConfig {
    LaunchConfig::new([row_tiles as u32, 1, 1], [THREADS as u32, 1, 1])
}

/// The example's input for `row_tiles` and `items_per_lane`, and an output
/// buffer.
pub fn example_args<T: ReduceDtype>(row_tiles: usize, items_per_lane: usize) -> Vec<KernelArg> {
    let rows = row_tiles * ROWS_PER_CTA;
    let src: Vec<T> = (0..rows * THREADS * items_per_lane).map(T::input).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![T::identity(ReduceOp::Add); rows])]
}

/// Every result a row can have: the kernel's lane folds and warp trees in
/// order, then the warps' contributions in each order they can land in (one
/// order for max and min, and for the shared-memory combine).
pub fn row_results<T: ReduceDtype>(row: &[T], op: ReduceOp) -> Vec<T> {
    let items = row.len() / THREADS;
    let lanes: Vec<T> = (0..THREADS)
        .map(|t| {
            let v = &row[t * items..(t + 1) * items];
            v[1..].iter().fold(v[0], |a, &b| T::host_combine(a, b, op))
        })
        .collect();
    let warps: Vec<T> = lanes
        .chunks(WARP_SIZE)
        .map(|w| {
            let mut v = w.to_vec();
            for offset in [16, 8, 4, 2, 1] {
                v = (0..WARP_SIZE).map(|l| T::host_combine(v[l], v[l ^ offset], op)).collect();
            }
            v[0]
        })
        .collect();
    if T::SMEM_MINMAX && op != ReduceOp::Add {
        return vec![warps[1..].iter().fold(warps[0], |a, &b| T::host_combine(a, b, op))];
    }
    let mut results = Vec::new();
    for perm in permutations(NUM_WARPS) {
        let r = perm.iter().fold(T::identity(op), |a, &w| T::host_combine(a, warps[w], op));
        if !results.contains(&r) {
            results.push(r);
        }
    }
    results
}

/// Every order of `0..n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut all = Vec::new();
    for p in permutations(n - 1) {
        for k in 0..=p.len() {
            let mut q = p.clone();
            q.insert(k, n - 1);
            all.push(q);
        }
    }
    all
}

/// Check `args` after a run: each row's result is one the kernel can give.
pub fn check<T: ReduceDtype + std::fmt::Debug>(args: &[KernelArg], op: ReduceOp) {
    let (src, out) = (args[0].to_vec::<T>(), args[1].to_vec::<T>());
    let row_width = src.len() / out.len();
    for (r, &got) in out.iter().enumerate() {
        let want = row_results(&src[r * row_width..(r + 1) * row_width], op);
        assert!(want.contains(&got), "{} {op:?}, row {r}: {got:?}, expected one of {want:?}", T::NAME);
    }
}

/// Build, simulate and check one configuration on the example's data.
pub fn run<T: ReduceDtype + std::fmt::Debug>(row_tiles: usize, items_per_lane: usize, op: ReduceOp) -> SimReport {
    let kernel = build::<T>(row_tiles, items_per_lane, op);
    let mut args = example_args::<T>(row_tiles, items_per_lane);
    let report = kernel
        .simulate_all_orders(launch(row_tiles), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    check::<T>(&args, op);
    report
}

#[allow(dead_code)]
fn main() {
    for op in ReduceOp::ALL {
        run::<i32>(DEFAULT_ROW_TILES, DEFAULT_ITEMS_PER_LANE, op);
        run::<f32>(DEFAULT_ROW_TILES, DEFAULT_ITEMS_PER_LANE, op);
        run::<F16>(DEFAULT_ROW_TILES, DEFAULT_ITEMS_PER_LANE, op);
        run::<BF16>(DEFAULT_ROW_TILES, DEFAULT_ITEMS_PER_LANE, op);
        println!("block_smem_reduce {}: correct for i32, f32, f16 and bf16", op.name());
    }
    let kernel = build::<i32>(DEFAULT_ROW_TILES, DEFAULT_ITEMS_PER_LANE, ReduceOp::Add);
    match Toolchain::discover() {
        Some(tc) => match kernel.compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let count = |s: &str| ptx.lines().filter(|l| l.contains(s)).count();
                println!(
                    "PTX for {}: {} redux.sync.add.s32, {} red.relaxed.cta.shared.add.s32",
                    kernel.name(),
                    count("redux.sync.add.s32"),
                    count("red.relaxed.cta.shared.add.s32")
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_apply.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A generic element-wise kernel, ported from the CuTe DSL example
//! `elementwise_apply.py`: `C = op(inputs...)` on `M x N` row-major
//! matrices, with `op` a Rust closure over the inputs' `TensorSsa` values
//! (CuTe DSL passes a `Constexpr` callable over `thrInput.load()`).
//!
//! The tiling is CuTe DSL's: 4 x 64 threads (row-major), each owning 16
//! rows of 16 bytes (`recast_layout` to the element type: 16 x 4 Float32
//! or 16 x 8 Float16), combined with `make_layout_tv`. Every input and the
//! output are `zipped_divide`d by the tile and composed with a block remap,
//! the block's tile is composed with the thread-value layout, and each
//! thread loads its values whole (`load_ssa`: 128-bit loads, since
//! `autovec_copy` finds the 16-byte rows), applies `op` and stores the
//! result whole. As in CuTe DSL, an identity-tensor predicate is computed
//! and not used: the shapes here divide the tile.
//!
//! The operations of the example's test: `add`, `sub`, `mul`, `truediv`,
//! `floordiv` and `leaky_relu` on Float32, `add` and `mul` on Float16 and
//! BFloat16. `M` and `N` are static.
//!
//! Run with `cargo run --example elementwise_apply`.

use cute::prelude::*;

pub const M: usize = 256;
pub const N: usize = 512;
/// Threads per block: the 4 x 64 thread layout.
pub const THREADS: u32 = 256;

/// The operations of the example's test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyOp {
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    LeakyRelu,
}

impl ApplyOp {
    pub const F32: [ApplyOp; 6] =
        [ApplyOp::Add, ApplyOp::Sub, ApplyOp::Mul, ApplyOp::TrueDiv, ApplyOp::FloorDiv, ApplyOp::LeakyRelu];
    pub const HALF: [ApplyOp; 2] = [ApplyOp::Add, ApplyOp::Mul];

    pub fn inputs(self) -> usize {
        if self == ApplyOp::LeakyRelu { 1 } else { 2 }
    }

    fn name(self) -> &'static str {
        match self {
            ApplyOp::Add => "add",
            ApplyOp::Sub => "sub",
            ApplyOp::Mul => "mul",
            ApplyOp::TrueDiv => "truediv",
            ApplyOp::FloorDiv => "floordiv",
            ApplyOp::LeakyRelu => "leaky_relu",
        }
    }
}

/// The slope of `leaky_relu`.
pub const ALPHA: f32 = 0.01;

/// A kernel and its launch.
pub struct Apply {
    pub kernel: Kernel,
    pub cfg: LaunchConfig,
}

/// `t`'s values for thread `tidx` of block `(bx, by)`: the tile of the
/// remapped `zipped_divide`, composed with the thread-value layout.
fn thread_view<X>(t: &Tensor<X>, tiler: &IntTuple, remap: &Layout, tv: &Layout, b: Dim3, tidx: Int32) -> Tensor<X> {
    zipped_divide(t, tiler).compose((.., remap.clone())).slice(((.., ..), (b.x, b.y))).compose(tv).slice((tidx, ..))
}

/// `C = op(inputs...)` on `T` elements: `inputs` row-major `M x N`
/// parameters `in0`, `in1`, ... and the output `c`.
pub fn elementwise_apply<T: Numeric>(name: &str, inputs: usize, op: impl Fn(&[TensorSsa<T>]) -> TensorSsa<T>) -> Apply {
    const COALESCED_LDST_BYTES: usize = 16;
    let thr_layout = make_ordered_layout((4, 64), &[1, 0]);
    let val_layout = make_ordered_layout((16, COALESCED_LDST_BYTES), &[1, 0]);
    let val_layout = recast_layout(T::DTYPE.bits(), 8, &val_layout);
    let (tiler_mn, tv_layout) = make_layout_tv(&thr_layout, &val_layout);
    let tile = tiler_mn.flat_leaves();
    let (rest_m, rest_n) = (M / tile[0], N / tile[1]);
    // CuTe DSL's remap_block: make_ordered_layout(select(shape[1], [1, 0]),
    // order=(1, 0)), so that block x walks the rest of N.
    let remap_block = make_ordered_layout((rest_n, rest_m), &[1, 0]);
    let kernel = Kernel::builder(name).block_dim([THREADS, 1, 1]).build(|| {
        let ins: Vec<Tensor<T>> =
            (0..inputs).map(|i| param_tensor::<T>(&format!("in{i}"), Layout::row_major((M, N)))).collect();
        let c = param_tensor::<T>("c", Layout::row_major((M, N)));
        let tidx = thread_idx().x;
        let bid = block_idx();
        let view = |t: &Tensor<T>| thread_view(t, &tiler_mn, &remap_block, &tv_layout, bid, tidx);
        let thr_inputs: Vec<Tensor<T>> = ins.iter().map(view).collect();
        let thr_c = view(&c);
        let thr_crd = thread_view(&make_identity_tensor((M, N)), &tiler_mn, &remap_block, &tv_layout, bid, tidx);
        // CuTe DSL builds this predicate and leaves it unused.
        let _frg_pred: Vec<Boolean> = (0..thr_crd.size()).map(|i| elem_less(thr_crd.coord(i), (M, N))).collect();
        let loads: Vec<TensorSsa<T>> = thr_inputs.iter().map(|t| t.load_ssa()).collect();
        let result = op(&loads);
        thr_c.store_ssa(&result);
    });
    Apply { kernel, cfg: LaunchConfig::new([rest_n as u32, rest_m as u32, 1], [THREADS, 1, 1]) }
}

/// The Float32 kernel of `op`.
pub fn f32_kernel(op: ApplyOp) -> Apply {
    let name = format!("elementwise_apply_{}_f32", op.name());
    elementwise_apply::<f32>(&name, op.inputs(), move |x| match op {
        ApplyOp::Add => &x[0] + &x[1],
        ApplyOp::Sub => &x[0] - &x[1],
        ApplyOp::Mul => &x[0] * &x[1],
        ApplyOp::TrueDiv => &x[0] / &x[1],
        ApplyOp::FloorDiv => x[0].floor_div(&x[1]),
        ApplyOp::LeakyRelu => where_(&x[0].gt(0.0), &x[0], &(x[0].clone() * ALPHA)),
    })
}

macro_rules! half_kernel {
    ($name:ident, $t:ty, $suffix:literal) => {
        /// The kernel of `op` (`Add` or `Mul`) on this half type.
        pub fn $name(op: ApplyOp) -> Apply {
            let name = format!("elementwise_apply_{}_{}", op.name(), $suffix);
            elementwise_apply::<$t>(&name, 2, move |x| match op {
                ApplyOp::Add => &x[0] + &x[1],
                ApplyOp::Mul => &x[0] * &x[1],
                _ => panic!("elementwise_apply: the {} test runs add and mul", $suffix),
            })
        }
    };
}
half_kernel!(f16_kernel, F16, "f16");
half_kernel!(bf16_kernel, BF16, "bf16");

/// Deterministic inputs in [-4, 4) on a 1/64 grid (sums and products of
/// two of them are exact in Float32, so rounding the Float32 reference to
/// a half type rounds once), with the second input's zeros replaced by 1.2
/// for the divisions, as the example does.
pub fn inputs(op: ApplyOp) -> Vec<Vec<f32>> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..op.inputs())
        .map(|k| {
            (0..M * N)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let v = ((state >> 40) % 512) as f32 / 64.0 - 4.0;
                    if k == 1 && v == 0.0 && matches!(op, ApplyOp::TrueDiv | ApplyOp::FloorDiv) { 1.2 } else { v }
                })
                .collect()
        })
        .collect()
}

/// The CPU reference of `op` in Float32.
pub fn reference(op: ApplyOp, x: &[f32]) -> f32 {
    match op {
        ApplyOp::Add => x[0] + x[1],
        ApplyOp::Sub => x[0] - x[1],
        ApplyOp::Mul => x[0] * x[1],
        ApplyOp::TrueDiv => x[0] / x[1],
        ApplyOp::FloorDiv => (x[0] / x[1]).floor(),
        ApplyOp::LeakyRelu => {
            if x[0] > 0.0 {
                x[0]
            } else {
                x[0] * ALPHA
            }
        }
    }
}

/// Simulate `apply` on `inputs` (converted to `T` by `to`) and check every
/// output bit for bit against the Float32 reference rounded by `to`.
pub fn run<T: Numeric + Copy>(
    apply: &Apply,
    op: ApplyOp,
    inputs: &[Vec<f32>],
    to: impl Fn(f32) -> T,
    bits: impl Fn(T) -> u64,
) -> SimReport {
    let typed: Vec<Vec<T>> = inputs.iter().map(|v| v.iter().map(|&x| to(x)).collect()).collect();
    let mut args: Vec<KernelArg> = typed.iter().map(|v| KernelArg::buffer(v)).collect();
    args.push(KernelArg::buffer(&std::iter::repeat_n(to(0.0), M * N).collect::<Vec<T>>()));
    let report = apply.kernel.simulate(apply.cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let c = args.last().unwrap().to_vec::<T>();
    let as_f32 = |v: T| -> f32 {
        match T::DTYPE {
            DType::F32 => f32::from_bits(bits(v) as u32),
            DType::F16 => F16(bits(v) as u16).to_f32(),
            _ => BF16(bits(v) as u16).to_f32(),
        }
    };
    for i in 0..M * N {
        let x: Vec<f32> = typed.iter().map(|v| as_f32(v[i])).collect();
        let want = to(reference(op, &x));
        assert_eq!(bits(c[i]), bits(want), "{}: element {i} of {x:?}", apply.kernel.name());
    }
    report
}

/// [`run`] for each type's kernels.
pub fn run_f32(op: ApplyOp) -> SimReport {
    run(&f32_kernel(op), op, &inputs(op), |x| x, |v: f32| v.to_bits() as u64)
}

pub fn run_f16(op: ApplyOp) -> SimReport {
    run(&f16_kernel(op), op, &inputs(op), F16::from_f32, |v: F16| v.0 as u64)
}

pub fn run_bf16(op: ApplyOp) -> SimReport {
    run(&bf16_kernel(op), op, &inputs(op), BF16::from_f32, |v: BF16| v.0 as u64)
}

fn main() {
    for op in ApplyOp::F32 {
        let report = run_f32(op);
        println!("{} on Float32: correct, global efficiency {:.0}%", op.name(), report.global_efficiency() * 100.0);
    }
    for op in ApplyOp::HALF {
        run_f16(op);
        run_bf16(op);
        println!("{} on Float16 and BFloat16: correct", op.name());
    }
    match Toolchain::discover() {
        Some(tc) => {
            let ptx = f32_kernel(ApplyOp::Add).kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
            let v4 = ptx.lines().filter(|l| l.contains("ld.global.v4") || l.contains("st.global.v4")).count();
            println!("PTX of add on Float32: {v4} 128-bit global loads and stores");
        }
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

//! Kernels that print, with their arguments and the text they print, for
//! the simulator tests (tests/printf.rs) and the GPU (tests/gpu.rs).
//!
//! `hello` ports the kernel of the notebook
//! `examples/python/CuTeDSL/notebooks/1_dsl_features/01_hello_world.ipynb`
//! of CUTLASS 4.8.1, Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES.
//! All rights reserved. BSD-3-Clause; see THIRD-PARTY-NOTICES.md.

use cute::prelude::*;

pub struct PrintCase {
    pub kernel: Kernel,
    pub cfg: LaunchConfig,
    pub args: fn() -> Vec<KernelArg>,
    /// What every thread prints, in order (the simulator's text).
    pub expected: String,
    /// Elements of the hello kernel's arrays.
    pub n: usize,
}

/// The line that prints a global pointer, whose address differs between
/// the simulator and the GPU; it starts with this.
pub const GLOBAL_POINTER_LINE: &str = "global pointer ";

/// One thread printing every kind of operand.
pub fn everything() -> PrintCase {
    let kernel = Kernel::builder("print_everything").block_dim([1, 1, 1]).build(|| {
        let m = param_scalar::<i32>("m");
        let big = param_scalar::<i64>("big");
        let neg = param_scalar::<i32>("neg");
        let u = param_scalar::<u32>("u");
        let h = param_scalar::<F16>("h");
        let bf = param_scalar::<BF16>("bf");
        let x = param_ptr::<f32>("x");

        let fixed = Layout::col_major((4, 8));
        printf("static layout {}\n", &[(&fixed).into()]);
        printf("dynamic layout {}\n", &[(&make_layout((m, 8), (8, 1))).into()]);
        printf("i64 leaves {}\n", &[(&make_layout((IndexExpr::from(big), 2), (1, IndexExpr::from(big)))).into()]);
        let composed = make_composed_layout(make_swizzle(3, 4, 3), m, Layout::row_major((8, 64)));
        printf("composed {}\n", &[(&composed).into()]);
        let identity = make_identity_layout((2, m));
        printf("basis {} stride {}\n", &[(&identity).into(), identity.stride().clone().into()]);
        let coord: SliceCoord = Tree::Branch(vec![Tree::Leaf(Some(m.into())), Tree::Leaf(None)]);
        printf("coord {}\n", &[coord.into()]);
        let tile: Tiler = Tree::Branch(vec![Tree::Leaf(None), Tree::Leaf(Some(Layout::col_major(2)))]);
        printf("tile {} shape {}\n", &[tile.into(), IntTuple::from((3, (4, 5))).into()]);
        printf(
            "ints {} {} {} {} {}\n",
            &[
                neg.into(),
                u.into(),
                Uint8::constant(200).into(),
                Int8::constant(-100).into(),
                Boolean::constant(true).into(),
            ],
        );
        printf("i64 {} u64 {}\n", &[big.into(), Uint64::constant(u64::MAX).into()]);
        printf("halves {} {} %.3f\n", &[h.into(), bf.into(), h.into()]);
        printf(
            "mixed a={}, b=%.2f, c=%5d|%-4u|%+.1e\n",
            &[(&fixed).into(), 1.23456f32.into(), neg.into(), u.into(), 12345.678f64.into()],
        );
        printf_values(&[1.into(), 2.5f32.into(), (&fixed).into()]);
        printf_end("end={}", &[m.into()], " 100%\n");
        let s = alloc_smem::<f32>(8, 16);
        s.store_at(2, 0.0);
        printf("shared pointer {} %p\n", &[(&(s.clone() + 2)).into(), (&s).into()]);
        printf("global pointer {}\n", &[(&x).into()]);
    });
    let expected = format!(
        "static layout (4,8):(1,4)
dynamic layout (5,8):(8,1)
i64 leaves (5000000000,2):(1,5000000000)
composed S<3,4,3> o 5 o (8,64):(64,1)
basis (2,5):(1@0,1@1) stride (1@0,1@1)
coord (5,_)
tile (_,2:1) shape (3,(4,5))
ints -7 4000000000 200 -100 1
i64 5000000000 u64 18446744073709551615
halves 1.500000 -0.250000 1.500
mixed a=(4,8):(1,4), b=1.23, c=   -7|4000000000|+1.2e+04
1, 2.500000, (4,8):(1,4)
end=5 100%
shared pointer 0x8 (nil)
global pointer {:#x}
",
        cute::sim::SIM_GLOBAL_BASE
    );
    PrintCase {
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [1, 1, 1]),
        args: || {
            vec![
                KernelArg::scalar(5i32),
                KernelArg::scalar(5_000_000_000i64),
                KernelArg::scalar(-7i32),
                KernelArg::scalar(4_000_000_000u32),
                KernelArg::scalar(F16::from_f32(1.5)),
                KernelArg::scalar(BF16::from_f32(-0.25)),
                KernelArg::f32(vec![0.0; 4]),
            ]
        },
        expected,
        n: 0,
    }
}

/// The hello-world kernel: thread 0 greets, every thread adds an element.
pub fn hello() -> PrintCase {
    const N: usize = 300;
    let kernel = Kernel::builder("hello_kernel").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major(N));
        let b = param_tensor::<f32>("b", Layout::col_major(N));
        let c = param_tensor::<f32>("c", Layout::col_major(N));
        let n = param_scalar::<i32>("N");
        let tx = thread_idx().x;
        let bx = block_idx().x;
        let bdx = block_dim().x;
        let i = bx * bdx + tx;
        if_(i.eq(0), || printf_end("hello from the GPU -- block {}, thread {}\n", &[bx.into(), tx.into()], "\n"));
        if_(i.lt(n), || c.store(i, a.load(i) + b.load(i)));
    });
    PrintCase {
        kernel,
        cfg: LaunchConfig::new([N.div_ceil(128) as u32, 1, 1], [128, 1, 1]),
        args: || {
            vec![
                KernelArg::f32((0..N).map(|i| i as f32).collect()),
                KernelArg::f32((0..N).map(|i| 2.0 * i as f32).collect()),
                KernelArg::f32(vec![0.0; N]),
                KernelArg::scalar(N as i32),
            ]
        },
        expected: "hello from the GPU -- block 0, thread 0\n\n".into(),
        n: N,
    }
}

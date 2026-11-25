// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kernels declared through the kernel argument protocol (`KernelParam`,
//! `TensorSpec`, `SymInt`, layouts, `Option`s, tuples, arrays, structs and
//! native structs), each compiled once and run at several problem sizes
//! from host values: `tests/kernel_params.rs` runs them on the simulator,
//! `tests/gpu.rs` on the GPU, both against plain Rust.

use cute::prelude::*;

/// Runs a kernel on its flat argument list (the simulator or the GPU).
pub type Runner<'a> = &'a dyn Fn(&Kernel, LaunchConfig, &mut Vec<KernelArg>);

pub struct ParamCase {
    pub name: &'static str,
    /// The kernels the case launches (to compile each once).
    pub kernels: Vec<Kernel>,
    /// Launches every problem size through the runner and checks the
    /// outputs against plain Rust.
    pub check: Box<dyn Fn(Runner)>,
}

const T: u32 = 64;

/// Pack `host` for `k`, run it, and take the outputs back.
fn launch<P: KernelParam>(k: &TypedKernel<P>, blocks: usize, host: &mut P::Host, run: Runner) {
    let mut args = k.pack(host).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    run(k, LaunchConfig::new([blocks.max(1) as u32, 1, 1], [T, 1, 1]), &mut args);
    k.write_back(host, args);
}

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n).map(|i| ((i as u32 * 37 + seed * 11) % 23) as f32 - 11.0).collect()
}

fn tid() -> Int32 {
    block_idx().x * T as i32 + thread_idx().x
}

/// A `TensorSpec` tensor with dynamic extents and a dynamic 64-bit leading
/// stride, written to a column-major tensor sharing its extents.
pub fn spec_tensor() -> ParamCase {
    let (m, n, ld) = (sym_int32(1, Some("M")), sym_int32(4, None), sym_int64(4, None));
    let x = make_fake_tensor::<f32>((m, n), (ld, 1));
    let y = make_fake_tensor::<f32>((m, n), (1, m));
    let k = Kernel::builder("param_spec_tensor").block_dim([T, 1, 1]).jit((x, y), |x: Tensor<f32>, y: Tensor<f32>| {
        let r = tid();
        if_(r.lt(size(&x, &[0])), || {
            for_range(0, size(&x, &[1]), 1, |j| y.store((r, j), x.load((r, j)) * 2.0 + 1.0));
        });
    });
    ParamCase {
        name: "param_spec_tensor",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for (m, n, ld) in [(5, 8, 12), (37, 4, 4), (130, 12, 16)] {
                let xv = values(m * ld, 1);
                let x = HostTensor::from_slice(&xv, Layout::new((m, n), (ld, 1)));
                let y = HostTensor::from_slice(&vec![0f32; m * n], Layout::new((m, n), (1, m)));
                let mut host = (x, y);
                launch(&k, m.div_ceil(T as usize), &mut host, run);
                let y = host.1.to_vec::<f32>();
                for r in 0..m {
                    for c in 0..n {
                        assert_eq!(y[r + c * m], xv[r * ld + c] * 2.0 + 1.0, "{m}x{n} (ld {ld}): y[{r},{c}]");
                    }
                }
            }
        }),
    }
}

/// A layout parameter with an `i32` extent and an `i64` stride, gathering
/// through it from a pointer.
pub fn layout_param() -> ParamCase {
    let spec =
        (PtrSpec::default(), LayoutSpec::new((sym_int32(1, None), 4), (sym_int64(4, None), 1)), PtrSpec::default());
    let k =
        Kernel::builder("param_layout").block_dim([T, 1, 1]).jit(spec, |src: Ptr<f32>, l: Layout, dst: Ptr<f32>| {
            let i = tid();
            if_(i.lt(size(&l, &[])), || {
                let s = make_tensor(&src, l.clone());
                let d = make_tensor(&dst, Layout::col_major(size(&l, &[])));
                d.store(i, s.load((i / 4, i % 4)) + 0.5);
            });
        });
    ParamCase {
        name: "param_layout",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for (rows, ld) in [(3, 4), (17, 8), (40, 12)] {
                let src = values(rows * ld, 2);
                let mut host = (
                    KernelArg::buffer(&src),
                    Layout::new((rows, 4), (ld, 1)),
                    KernelArg::buffer(&vec![0f32; rows * 4]),
                );
                launch(&k, (rows * 4).div_ceil(T as usize), &mut host, run);
                let dst = host.2.to_vec::<f32>();
                for i in 0..rows * 4 {
                    assert_eq!(dst[i], src[(i / 4) * ld + i % 4] + 0.5, "{rows} rows (ld {ld}): dst[{i}]");
                }
            }
        }),
    }
}

/// An optional bias: built with and without it.
pub fn option_param() -> ParamCase {
    let n = sym_int32(4, None);
    let t = make_fake_compact_tensor::<f32>((n,), None, false);
    let body = |x: Tensor<f32>, bias: Option<Tensor<f32>>, y: Tensor<f32>| {
        let i = tid();
        if_(i.lt(size(&x, &[0])), || {
            let v = match &bias {
                Some(b) => x.load(i) + b.load(i),
                None => x.load(i),
            };
            y.store(i, v * 3.0);
        });
    };
    let with =
        Kernel::builder("param_option_some").block_dim([T, 1, 1]).jit((t.clone(), Some(t.clone()), t.clone()), body);
    let without = Kernel::builder("param_option_none").block_dim([T, 1, 1]).jit((t.clone(), None, t), body);
    assert_eq!(with.ir().args.len(), 4, "x, its extent, the bias and y");
    assert_eq!(without.ir().args.len(), 3, "x, its extent and y");
    ParamCase {
        name: "param_option",
        kernels: vec![with.kernel().clone(), without.kernel().clone()],
        check: Box::new(move |run| {
            for n in [4usize, 68, 200] {
                let (x, b) = (values(n, 3), values(n, 4));
                let ht = |v: &[f32]| HostTensor::from_slice(v, Layout::col_major((n,)));
                let mut some = (ht(&x), Some(ht(&b)), ht(&vec![0.0; n]));
                launch(&with, n.div_ceil(T as usize), &mut some, run);
                let mut none = (ht(&x), None, ht(&vec![0.0; n]));
                launch(&without, n.div_ceil(T as usize), &mut none, run);
                let (ys, yn) = (some.2.to_vec::<f32>(), none.2.to_vec::<f32>());
                for i in 0..n {
                    assert_eq!(ys[i], (x[i] + b[i]) * 3.0, "n = {n}: with the bias, y[{i}]");
                    assert_eq!(yn[i], x[i] * 3.0, "n = {n}: without it, y[{i}]");
                }
            }
        }),
    }
}

/// A tuple of scalars (`Int32`, `Float32`, `Int64`) next to two tensors.
pub fn scalar_tuple() -> ParamCase {
    let n = sym_int32(1, None);
    let spec = (
        (ScalarSpec::Plain, ScalarSpec::Plain, ScalarSpec::Plain),
        make_fake_compact_tensor::<f32>((n,), None, false),
        make_fake_compact_tensor::<i64>((n,), None, false),
    );
    let k = Kernel::builder("param_scalar_tuple").block_dim([T, 1, 1]).jit(
        spec,
        |s: (Int32, Float32, Int64), y: Tensor<f32>, z: Tensor<i64>| {
            let (a, b, c) = s;
            let i = tid();
            if_(i.lt(size(&y, &[0])), || {
                y.store(i, (a + i).to::<f32>() * b);
                z.store(i, c + i.to::<i64>());
            });
        },
    );
    ParamCase {
        name: "param_scalar_tuple",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for (n, a, b, c) in [(10usize, 3i32, 0.5f32, 1i64 << 40), (100, -7, 2.0, -5)] {
                let y = HostTensor::from_slice(&vec![0f32; n], Layout::col_major((n,)));
                let z = HostTensor::from_slice(&vec![0i64; n], Layout::col_major((n,)));
                let mut host = ((a, b, c), y, z);
                launch(&k, n.div_ceil(T as usize), &mut host, run);
                let (y, z) = (host.1.to_vec::<f32>(), host.2.to_vec::<i64>());
                for i in 0..n {
                    assert_eq!(y[i], (a + i as i32) as f32 * b, "y[{i}]");
                    assert_eq!(z[i], c + i as i64, "z[{i}]");
                }
            }
        }),
    }
}

/// A `FastDivmod` prepared on the host.
pub fn host_divmod() -> ParamCase {
    let n = sym_int32(1, None);
    let k = Kernel::builder("param_fast_divmod").block_dim([T, 1, 1]).jit(
        ((), make_fake_compact_tensor::<i32>((n,), None, false)),
        |d: FastDivmod, out: Tensor<i32>| {
            let i = tid();
            if_(i.lt(size(&out, &[0])), || {
                let (q, r) = d.divmod(i * 977);
                out.store(i, q * 1000 + r);
            });
        },
    );
    ParamCase {
        name: "param_fast_divmod",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for d in [1, 3, 7, 64, 1000] {
                let n = 150;
                let mut host = (FastDivmod::host(d), HostTensor::from_slice(&vec![0i32; n], Layout::col_major((n,))));
                launch(&k, n.div_ceil(T as usize), &mut host, run);
                let out = host.1.to_vec::<i32>();
                for (i, &o) in out.iter().enumerate() {
                    let v = i as i32 * 977;
                    assert_eq!(o, v / d * 1000 + v % d, "divisor {d}: out[{i}]");
                }
            }
        }),
    }
}

/// Tensors from host tensors marked dynamic: the kernel takes its spec from
/// the first host values (call-site typing) and runs at other sizes.
pub fn marked_tensors() -> ParamCase {
    let host = |m: usize, n: usize, seed: u32| {
        let a =
            HostTensor::from_slice(&values(m * n, seed), Layout::row_major((m, n))).mark_layout_dynamic(None).unwrap();
        let b = HostTensor::from_slice(&values(m * n, seed + 1), Layout::row_major((m, n)))
            .mark_compact_shape_dynamic(0, None, 1)
            .unwrap()
            .mark_compact_shape_dynamic(1, None, 4)
            .unwrap();
        let c =
            HostTensor::from_slice(&vec![0f32; m * n], Layout::row_major((m, n))).mark_layout_dynamic(None).unwrap();
        (a, b, c)
    };
    let first = host(4, 8, 5);
    assert_eq!(first.0.layout_string(), "(?,?):(?{i64},1)");
    assert_eq!(first.1.layout_string(), "(?,?{div=4}):(?{i64 div=4},1)");
    let k = Kernel::builder("param_marked").block_dim([T, 1, 1]).jit_for(
        &first,
        |a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>| {
            let r = tid();
            if_(r.lt(size(&a, &[0])), || {
                for_range(0, size(&a, &[1]), 1, |j| c.store((r, j), a.load((r, j)) - b.load((r, j)) * 2.0));
            });
        },
    );
    ParamCase {
        name: "param_marked",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for (m, n) in [(4, 8), (70, 4), (9, 20)] {
                let mut h = host(m, n, 9);
                let (a, b) = (h.0.to_vec::<f32>(), h.1.to_vec::<f32>());
                launch(&k, m.div_ceil(T as usize), &mut h, run);
                let c = h.2.to_vec::<f32>();
                for i in 0..m * n {
                    assert_eq!(c[i], a[i] - b[i] * 2.0, "{m}x{n}: c[{i}]");
                }
            }
        }),
    }
}

/// One 64-bit `SymInt` as the extent of two tensors and as a scalar: one
/// parameter.
pub fn shared_symint() -> ParamCase {
    let n = sym_int64(1, Some("n"));
    let t = make_fake_compact_tensor::<f32>((n,), Some(&[0]), false);
    let k = Kernel::builder("param_shared_symint").block_dim([T, 1, 1]).jit(
        (t.clone(), t, ScalarSpec::Sym(n)),
        |src: Tensor<f32>, dst: Tensor<f32>, len: Int64| {
            let i = tid();
            if_(i.to::<i64>().lt(len), || dst.store(i, src.load(i) * 3.0));
        },
    );
    let names = k.param_names();
    assert_eq!(names, ["arg0", "arg0_shape_0", "arg1"], "the SymInt is one parameter");
    ParamCase {
        name: "param_shared_symint",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for n in [1usize, 64, 1024, 333] {
                let src = values(n, 6);
                let ht = |v: &[f32]| HostTensor::from_slice(v, Layout::col_major((n,)));
                let mut host = (ht(&src), ht(&vec![0.0; n]), n as i64);
                launch(&k, n.div_ceil(T as usize), &mut host, run);
                let dst = host.1.to_vec::<f32>();
                assert!(dst.iter().zip(&src).all(|(d, s)| *d == s * 3.0), "n = {n}");
            }
        }),
    }
}

/// An array of tensors.
pub fn tensor_array() -> ParamCase {
    let n = sym_int32(2, None);
    let t = make_fake_compact_tensor::<f32>((n,), None, false);
    let k = Kernel::builder("param_tensor_array").block_dim([T, 1, 1]).jit(
        ([t.clone(), t.clone(), t.clone()], t),
        |ins: [Tensor<f32>; 3], out: Tensor<f32>| {
            let i = tid();
            if_(i.lt(size(&out, &[0])), || out.store(i, ins[0].load(i) + ins[1].load(i) * 2.0 - ins[2].load(i)));
        },
    );
    ParamCase {
        name: "param_tensor_array",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for n in [2usize, 66, 130] {
                let v: Vec<Vec<f32>> = (0..3).map(|s| values(n, 10 + s)).collect();
                let ht = |v: &[f32]| HostTensor::from_slice(v, Layout::col_major((n,)));
                let mut host = ([ht(&v[0]), ht(&v[1]), ht(&v[2])], ht(&vec![0.0; n]));
                launch(&k, n.div_ceil(T as usize), &mut host, run);
                let out = host.1.to_vec::<f32>();
                for i in 0..n {
                    assert_eq!(out[i], v[0][i] + v[1][i] * 2.0 - v[2][i], "n = {n}: out[{i}]");
                }
            }
        }),
    }
}

kernel_params! {
    /// A struct argument with a build-time constant, carried through a loop.
    pub struct ScaleParams(ScaleParamsSpec, ScaleParamsHost) {
        pub src: Tensor<f32>,
        pub dst: Tensor<f32>,
        pub scale: Float32,
        pub steps: usize,
    }
}

pub fn struct_param() -> ParamCase {
    let n = sym_int32(1, None);
    let t = make_fake_compact_tensor::<f32>((n,), None, false);
    let spec = ScaleParamsSpec { src: t.clone(), dst: t, scale: ScalarSpec::Plain, steps: 3 };
    let k = Kernel::builder("param_struct").block_dim([T, 1, 1]).jit(spec, |p: ScaleParams| {
        let i = tid();
        // The struct flows through the loop: `scale` is carried, the tensors
        // and `steps` stay as they came in.
        let p = for_range_carry(0, p.steps as i32, 1, p, |_, mut p| {
            p.scale = p.scale * 2.0;
            p
        });
        if_(i.lt(size(&p.src, &[0])), || p.dst.store(i, p.src.load(i) * p.scale));
    });
    ParamCase {
        name: "param_struct",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for n in [5usize, 100] {
                let src = values(n, 20);
                let ht = |v: &[f32]| HostTensor::from_slice(v, Layout::col_major((n,)));
                let mut host = ScaleParamsHost { src: ht(&src), dst: ht(&vec![0.0; n]), scale: 0.75, steps: 3 };
                launch(&k, n.div_ceil(T as usize), &mut host, run);
                let dst = host.dst.to_vec::<f32>();
                for i in 0..n {
                    assert_eq!(dst[i], src[i] * 6.0, "n = {n}: dst[{i}]");
                }
            }
        }),
    }
}

kernel_params! {
    #[native_struct(zero_init = true, packed = false)]
    /// A native struct (CuTe DSL's `@cute.native_struct`).
    pub struct Affine(AffineSpec, AffineHost) {
        pub a: Float32,
        pub b: Float32,
        pub shift: Int32,
        pub tag: Constexpr<u32>,
    }
}

/// A native struct argument, next to a native struct made in the kernel.
pub fn native_struct_param() -> ParamCase {
    let n = sym_int32(1, None);
    let spec = (
        AffineSpec { a: ScalarSpec::Plain, b: ScalarSpec::Plain, shift: ScalarSpec::Plain, tag: 7 },
        make_fake_compact_tensor::<f32>((n,), None, false),
    );
    let k = Kernel::builder("param_native_struct").block_dim([T, 1, 1]).jit(spec, |f: Affine, out: Tensor<f32>| {
        let i = tid();
        let spec = AffineSpec { a: ScalarSpec::Plain, b: ScalarSpec::Plain, shift: ScalarSpec::Plain, tag: *f.tag };
        let mut z = Affine::make(&spec);
        z.b = z.b + 1.0;
        if_(i.lt(size(&out, &[0])), || {
            out.store(i, f.a * (i + f.shift).to::<f32>() + f.b + z.a + z.b * (*f.tag as f32));
        });
    });
    ParamCase {
        name: "param_native_struct",
        kernels: vec![k.kernel().clone()],
        check: Box::new(move |run| {
            for (n, a, b, shift) in [(7usize, 2.0f32, -1.0f32, 3i32), (90, 0.5, 4.0, -10)] {
                let mut host = (
                    AffineHost { a, b, shift, tag: 7 },
                    HostTensor::from_slice(&vec![0f32; n], Layout::col_major((n,))),
                );
                launch(&k, n.div_ceil(T as usize), &mut host, run);
                let out = host.1.to_vec::<f32>();
                for (i, &o) in out.iter().enumerate() {
                    assert_eq!(o, a * (i as i32 + shift) as f32 + b + 7.0, "n = {n}: out[{i}]");
                }
            }
        }),
    }
}

pub fn param_cases() -> Vec<ParamCase> {
    vec![
        spec_tensor(),
        layout_param(),
        option_param(),
        scalar_tuple(),
        host_divmod(),
        marked_tensors(),
        shared_symint(),
        tensor_array(),
        struct_param(),
        native_struct_param(),
    ]
}

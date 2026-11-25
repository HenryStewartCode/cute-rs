//! Scalar semantics: every arithmetic, comparison, cast, math and control
//! flow operation, checked in the simulator against the same computation in
//! Rust, then compiled to PTX (when `mlir-opt` is available) so that every
//! path of the MLIR emitter is exercised.

use cute::prelude::*;

const T: usize = 64;
const NI: usize = 22;
const NF: usize = 19;

fn host_ints(t: i32, x: f32, y: i32) -> [i32; NI] {
    [
        y + 3,
        y - t,
        y * 7,
        y / 3,
        y % 5,
        y.min(t),
        y.max(t),
        y & 6,
        y | 9,
        y ^ t,
        y << 2,
        (y & 0xff) >> 1,
        -y,
        !y,
        y.abs(),
        (x * 10.0) as i32,
        if y > 0 { t } else { 7 },
        (x < 0.5) as i32,
        if t < 32 { t * 2 } else { t + 100 },
        (0..t % 5).map(|i| i * i).sum(),
        if (t % 3 == 0) & (y != 0) | (t == 1) { 1 } else { 0 },
        (t / 2) % 32,
    ]
}

fn host_floats(t: i32, x: f32, y: i32, scale: f32) -> [f32; NF] {
    let h = |v: f32| F16::from_f32(v).to_f32();
    [
        x + 1.5,
        x - 0.25,
        x * 2.0,
        x / 3.0,
        -x,
        x.abs(),
        x.min(0.5),
        x.max(0.5),
        x.abs().sqrt(),
        x.mul_add(2.0, 1.0),
        if x > 0.0 { x } else { 0.0 },
        y as f32,
        h(x),
        h(h(x) * 2.0),
        x * scale,
        1.0 / (x.abs() + 1.0).sqrt(),
        (x * 0.1).exp2(),
        (x.abs() + 1.0).log2(),
        if t % 2 == 0 { x } else { (0..t % 4).fold(x, |acc, _| acc * 0.5) },
    ]
}

fn ops_kernel() -> Kernel {
    Kernel::builder("ops").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let int_out = param_tensor::<i32>("int_out", Layout::row_major((T, NI)));
        let float_out = param_tensor::<f32>("float_out", Layout::row_major((T, NF)));
        let scale = param_scalar::<f32>("scale");
        let t = thread_idx().x;
        let (x, y) = (xs.load(t), ys.load(t));

        let ints: [Int32; NI] = [
            y + 3,
            y - t,
            y * 7,
            y / 3,
            y % 5,
            y.min(t),
            y.max(t),
            y & 6,
            y | 9,
            y ^ t,
            y << 2,
            (y & 0xff) >> 1,
            -y,
            !y,
            y.abs(),
            (x * 10.0).to::<i32>(),
            select(y.gt(0), t, 7),
            x.lt(0.5).to::<i32>(),
            if_else_yield(t.lt(32), || t * 2, || t + 100),
            for_range_carry(0, t % 5, 1, Int32::from(0), |i, acc| acc + i * i),
            ((t % 3).eq(0) & y.ne(0) | t.eq(1)).to::<i32>(),
            (t.as_index() / 2usize % 32usize).into_value(),
        ];
        for (j, v) in ints.into_iter().enumerate() {
            int_out.store((t, j), v);
        }

        let half = x.to::<F16>();
        let floats: [Float32; NF] = [
            x + 1.5,
            x - 0.25,
            x * 2.0,
            x / 3.0,
            -x,
            x.abs(),
            x.min(0.5),
            x.max(0.5),
            x.abs().sqrt(),
            fma(x, 2.0, 1.0),
            select(x.gt(0.0), x, 0.0),
            y.to::<f32>(),
            half.to::<f32>(),
            (half * F16::from_f32(2.0)).to::<f32>(),
            x * scale,
            (x.abs() + 1.0).rsqrt(),
            (x * 0.1).exp2(),
            (x.abs() + 1.0).log2(),
            if_else_yield((t % 2).eq(0), || x, || for_range_carry(0, t % 4, 1, x, |_, acc| acc * 0.5)),
        ];
        for (j, v) in floats.into_iter().enumerate() {
            float_out.store((t, j), v);
        }
    })
}

fn inputs() -> (Vec<f32>, Vec<i32>) {
    let xs = (0..T).map(|i| (i as f32 - 20.0) * 0.37).collect();
    let ys = (0..T).map(|i| (i as i32 * 37) % 23 - 11).collect();
    (xs, ys)
}

#[test]
fn simulated_ops_match_rust() {
    let (xs, ys) = inputs();
    let scale = 1.25f32;
    let mut args = vec![
        KernelArg::f32(xs.clone()),
        KernelArg::i32(ys.clone()),
        KernelArg::i32(vec![0; T * NI]),
        KernelArg::f32(vec![0.0; T * NF]),
        KernelArg::ScalarF32(scale),
    ];
    ops_kernel().simulate(LaunchConfig::new([1, 1, 1], [T as u32, 1, 1]), &mut args).unwrap();
    for t in 0..T {
        let (x, y) = (xs[t], ys[t]);
        for (j, want) in host_ints(t as i32, x, y).into_iter().enumerate() {
            assert_eq!(args[2].as_i32()[t * NI + j], want, "int op {j} for thread {t} (x = {x}, y = {y})");
        }
        for (j, want) in host_floats(t as i32, x, y, scale).into_iter().enumerate() {
            let got = args[3].as_f32()[t * NF + j];
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "float op {j} for thread {t}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn ops_compile_to_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = ops_kernel().compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    for instruction in
        ["sqrt.rn.f32", "rsqrt.approx.f32", "ex2.approx.f32", "lg2.approx.f32", "fma.rn.f32", "cvt.rn.f16.f32"]
    {
        assert!(ptx.contains(instruction), "PTX lacks {instruction}");
    }
}

#[test]
fn ir_dump_lists_control_flow() {
    let dump = ops_kernel().dump();
    assert!(dump.starts_with("kernel ops(xs: gmem<64 x f32>"));
    assert!(dump.contains("for ") && dump.contains("if "), "{dump}");
}

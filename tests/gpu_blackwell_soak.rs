// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! A hardware soak of the datacenter Blackwell GEMM examples on an sm_100 GPU (B200, GB200;
//! sm_103 too). A CPU simulator cannot show a missing wait or fence that matters only
//! sometimes, so each kernel is checked twice, bit for bit, against the CPU reference the
//! functional tests (`tests/gpu_blackwell.rs`, `tests/gpu_blackwell_more.rs`) use:
//!
//! 1. Repeat: the kernel is loaded once and launched `SOAK_LAUNCHES` times (default 200) on one
//!    medium problem, the output buffer reset to a sentinel before each launch.
//! 2. Shapes: about ten problems per kernel inside its constraints (non-square, one K tile,
//!    long K, partial clusters, odd tile counts, the smallest legal one).
//!
//! A mismatch is recorded (kernel, configuration, shape, launch, the count of differing
//! elements, the first few with got/want) and the soak goes on; each test asserts at the end.
//! Every test is skipped without an sm_10x device.
//!
//! ```text
//! SOAK_LAUNCHES=200 cargo test --release --features cuda --test gpu_blackwell_soak -- \
//!     --ignored --nocapture --test-threads=1
//! ```

#[allow(dead_code)]
#[path = "../examples/blackwell_fp16_gemm.rs"]
mod blackwell_fp16_gemm;

#[allow(dead_code)]
#[path = "../examples/blackwell_dense_gemm.rs"]
mod blackwell_dense_gemm;

#[allow(dead_code)]
#[path = "../examples/blackwell_nvfp4_gemm.rs"]
mod blackwell_nvfp4_gemm;

#[allow(dead_code)]
#[path = "../examples/blackwell_mixed_input_gemm.rs"]
mod blackwell_mixed_input_gemm;

#[allow(dead_code)]
#[path = "../examples/custom_epilogue_dense_gemm.rs"]
mod custom;

#[allow(dead_code)]
#[path = "../examples/cute_ext_dense_gemm_persistent.rs"]
mod persistent;

#[allow(dead_code)]
#[path = "../examples/ts_gemm_simple.rs"]
mod ts_gemm_simple;

#[path = "support/gpu_device.rs"]
mod gpu_device;

use custom::efc::dense_gemm::{AbDtype, DenseGemmEfc, EFC_GEMM_1CTA, EFC_GEMM_2CTA_MULTICAST, matmul};
use custom::efc::{HostArgs, Param};
use cute::prelude::*;
use cute::utils::LayoutEnum;
use gpu_device::{Device, Loaded, device};
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn blackwell(what: &str) -> Option<Device> {
    device(10, what)
}

fn launches() -> usize {
    std::env::var("SOAK_LAUNCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(200)
}

/// The problem every kernel is repeated on: ragged in all three dimensions for every tile
/// shape here, with several tiles and a partial cluster.
const MEDIUM: (usize, usize, usize) = (520, 1032, 264);

/// Problems for the GEMMs whose M, N and K need only be multiples of 8 (TMA strides): the
/// smallest, one tile, one K tile, a long K, tall, wide, odd tile counts, partial clusters.
const SHAPES: [(usize, usize, usize); 11] = [
    (8, 8, 8),
    (128, 128, 64),
    (136, 72, 72),
    (264, 264, 136),
    (384, 128, 64),
    (128, 640, 64),
    (256, 256, 8192),
    (4096, 128, 256),
    (128, 4096, 256),
    (520, 1032, 136),
    (1000, 1528, 520),
];

/// Problems for `fp16_gemm_0`, which takes whole tiles only: M a multiple of 128, N of 256, K of
/// 64. The smallest, a long K, odd tile counts in M and N, tall, wide.
const TILED_SHAPES: [(usize, usize, usize); 10] = [
    (128, 256, 64),
    (256, 512, 64),
    (384, 256, 64),
    (128, 768, 64),
    (256, 256, 8192),
    (1024, 1024, 512),
    (4096, 256, 256),
    (128, 4096, 256),
    (640, 1280, 320),
    (2048, 1024, 1024),
];

/// The elements of `got` that differ from `want`, as a message, or none.
fn compare<T: PartialEq + Debug>(got: &[T], want: &[T], cols: usize) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{} elements, the reference {}", got.len(), want.len()));
    }
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if bad.is_empty() {
        return None;
    }
    let first: Vec<String> = bad
        .iter()
        .take(6)
        .map(|&i| format!("[{i}]=({},{}) got {:?} want {:?}", i / cols.max(1), i % cols.max(1), got[i], want[i]))
        .collect();
    Some(format!("{} of {} elements differ: {}", bad.len(), got.len(), first.join("; ")))
}

/// What a soak found, kernel by kernel.
struct Soak {
    test: &'static str,
    failures: Vec<String>,
}

impl Soak {
    fn new(test: &'static str) -> Soak {
        Soak { test, failures: Vec::new() }
    }

    /// Launch `kernel` with `args` `n` times, resetting each `(index, buffer)` of `resets`
    /// before each launch and running `check` on the arguments after it. One summary line.
    #[allow(clippy::too_many_arguments)]
    fn case(
        &mut self,
        kernel: &str,
        config: &str,
        shape: &str,
        loaded: &Loaded,
        launch: LaunchConfig,
        args: &mut [KernelArg],
        resets: &[(usize, KernelArg)],
        n: usize,
        check: &dyn Fn(&[KernelArg]) -> Option<String>,
    ) {
        println!("SOAK START {} {kernel} [{config}] {shape} x{n}", self.test);
        let (mut bad, mut first, mut panicked) = (0usize, None, None);
        for i in 0..n {
            for (at, fresh) in resets {
                args[*at] = fresh.clone();
            }
            let r = catch_unwind(AssertUnwindSafe(|| {
                loaded.launch(launch, args);
                check(args)
            }));
            match r {
                Ok(None) => {}
                Ok(Some(msg)) => {
                    bad += 1;
                    first.get_or_insert((i, msg));
                }
                Err(p) => {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    panicked = Some((i, msg));
                    break;
                }
            }
        }
        if let Some((i, msg)) = panicked {
            let line = format!("FAIL {kernel} [{config}] {shape} launch {i} panicked: {msg}");
            println!("SOAK {line}");
            self.failures.push(line);
        } else if let Some((i, msg)) = first {
            let line = format!("FAIL {kernel} [{config}] {shape} launch {i} ({bad} of {n} launches wrong): {msg}");
            println!("SOAK {line}");
            self.failures.push(line);
        } else {
            println!("SOAK OK {kernel} [{config}] {shape} x{n}");
        }
    }

    fn finish(self) {
        println!("SOAK SUMMARY {}: {} failures", self.test, self.failures.len());
        assert!(self.failures.is_empty(), "{} failures:\n{}", self.failures.len(), self.failures.join("\n"));
    }
}

/// `m`, `n`, `k`, then A, B and C, as every GEMM example here takes them.
fn mnk_abc<T: Numeric, C: Numeric>((m, n, k): (usize, usize, usize), a: &[T], b: &[T], c: &[C]) -> Vec<KernelArg> {
    vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        KernelArg::buffer(a),
        KernelArg::buffer(b),
        KernelArg::buffer(c),
    ]
}

fn shape_str((m, n, k): (usize, usize, usize)) -> String {
    format!("{m}x{n}x{k}")
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_fp16_gemm_0() {
    use blackwell_fp16_gemm::{FP16_GEMM_0, IoDtype, reference, small_ints};
    let Some(dev) = blackwell("soak_fp16_gemm_0") else { return };
    let mut soak = Soak::new("soak_fp16_gemm_0");
    let cfg = FP16_GEMM_0;
    let kernel = cfg.build();
    let loaded = dev.load(&kernel);
    let n_rep = launches();
    for (mnk, n_launch) in std::iter::once(((512, 1024, 256), n_rep)).chain(TILED_SHAPES.iter().map(|&s| (s, 1))) {
        let (m, n, k) = mnk;
        let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
        let want = reference(&a, &b, m, n, k);
        let sentinel = KernelArg::buffer(&vec![IoDtype::from_f32(-7.0); m * n]);
        let mut args = mnk_abc(mnk, &a, &b, &vec![IoDtype::from_f32(-7.0); m * n]);
        soak.case(
            cfg.name,
            "default",
            &shape_str(mnk),
            &loaded,
            cfg.launch(m, n),
            &mut args,
            &[(5, sentinel)],
            n_launch,
            &|args| compare(&args[5].to_vec::<IoDtype>(), &want, n),
        );
    }
    soak.finish();
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_fp16_gemm_ws() {
    use blackwell_fp16_gemm::{FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4, FP16_GEMM_4_STATIC, IoDtype};
    use blackwell_fp16_gemm::{reference, small_ints};
    let Some(dev) = blackwell("soak_fp16_gemm_ws") else { return };
    let mut soak = Soak::new("soak_fp16_gemm_ws");
    let n_rep = launches();
    let problems: Vec<_> = std::iter::once((MEDIUM, n_rep))
        .chain(SHAPES.iter().map(|&s| (s, 1)))
        .map(|(mnk, n_launch)| {
            let (m, n, k) = mnk;
            let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
            let want = reference(&a, &b, m, n, k);
            (mnk, n_launch, a, b, want)
        })
        .collect();
    for cfg in [FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4, FP16_GEMM_4_STATIC] {
        let (cx, cy, cz) = cfg.cluster_shape_mnk;
        let device_clusters = dev.max_active_clusters((cx * cy * cz) as u32);
        let kernel = cfg.build();
        let loaded = dev.load(&kernel);
        for (mnk, n_launch, a, b, want) in &problems {
            let (m, n, _) = *mnk;
            // The repeats run with the device's cluster count; the shapes with that and one.
            let counts: &[u32] = if *n_launch > 1 { &[device_clusters] } else { &[device_clusters, 1] };
            for &clusters in counts {
                let sentinel = KernelArg::buffer(&vec![IoDtype::from_f32(-7.0); m * n]);
                let mut args = mnk_abc(*mnk, a, b, &vec![IoDtype::from_f32(-7.0); m * n]);
                soak.case(
                    cfg.name,
                    &format!("{clusters} clusters"),
                    &shape_str(*mnk),
                    &loaded,
                    cfg.launch(m, n, clusters),
                    &mut args,
                    &[(5, sentinel)],
                    *n_launch,
                    &|args| compare(&args[5].to_vec::<IoDtype>(), want, n),
                );
            }
        }
    }
    soak.finish();
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_dense_gemm() {
    use blackwell_dense_gemm::{CType, DENSE_GEMM_1CTA, DENSE_GEMM_2CTA, DENSE_GEMM_2CTA_MULTICAST, DenseGemm};
    use blackwell_dense_gemm::{reference, small_ints};
    let Some(dev) = blackwell("soak_dense_gemm") else { return };
    let mut soak = Soak::new("soak_dense_gemm");
    let n_rep = launches();

    fn run<C: CType>(soak: &mut Soak, dev: &Device, cfg: &DenseGemm, shapes: &[((usize, usize, usize), usize)]) {
        let kernel = cfg.build::<C>();
        let loaded = dev.load(&kernel);
        for &(mnk, n_launch) in shapes {
            let (m, n, k) = mnk;
            let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
            let want = reference::<C>(&a, &b, mnk, cfg.c_layout, |x| x);
            let sentinel = KernelArg::buffer(&vec![C::from_f(-7.0); m * n]);
            let mut args = mnk_abc(mnk, &a, &b, &vec![C::from_f(-7.0); m * n]);
            // A column-major C is compared flat; only the report's (row, column) is off.
            soak.case(
                cfg.name,
                &format!("C {}", C::DTYPE),
                &shape_str(mnk),
                &loaded,
                cfg.launch(m, n),
                &mut args,
                &[(5, sentinel)],
                n_launch,
                &|args| compare(&args[5].to_vec::<C>(), &want, n),
            );
        }
    }

    let all: Vec<_> = std::iter::once((MEDIUM, n_rep)).chain(SHAPES.iter().map(|&s| (s, 1))).collect();
    for cfg in [DENSE_GEMM_1CTA, DENSE_GEMM_2CTA, DENSE_GEMM_2CTA_MULTICAST] {
        run::<F16>(&mut soak, &dev, &cfg, &all);
        run::<f32>(&mut soak, &dev, &cfg, &all);
    }
    let m_major: Vec<_> = std::iter::once(((520, 264, 264), n_rep))
        .chain(
            [(8, 8, 8), (136, 72, 72), (264, 136, 136), (128, 128, 64), (384, 128, 64), (256, 128, 4096)]
                .map(|s| (s, 1)),
        )
        .chain([(1024, 512, 256), (520, 1032, 136), (1000, 1528, 520)].map(|s| (s, 1)))
        .collect();
    for cfg in [
        DenseGemm { c_layout: LayoutEnum::COL_MAJOR, name: "blackwell_dense_gemm_1cta_m_major", ..DENSE_GEMM_1CTA },
        DenseGemm { c_layout: LayoutEnum::COL_MAJOR, name: "blackwell_dense_gemm_2cta_m_major", ..DENSE_GEMM_2CTA },
    ] {
        run::<F16>(&mut soak, &dev, &cfg, &m_major);
    }
    soak.finish();
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_nvfp4_gemm() {
    use blackwell_nvfp4_gemm::{AbDtype, CDtype, NVFP4_GEMM_0, NVFP4_GEMM_0_UE8M0, Problem, typed};
    let Some(dev) = blackwell("soak_nvfp4_gemm") else { return };
    let mut soak = Soak::new("soak_nvfp4_gemm");
    let n_rep = launches();
    // M a multiple of 128, N of 256, K of 256 (a 128x256x256 tile).
    let shapes: Vec<_> = std::iter::once(((512, 768, 1024), n_rep))
        .chain(
            [
                (128, 256, 256),
                (128, 256, 1280),
                (256, 512, 512),
                (384, 256, 256),
                (128, 768, 256),
                (640, 512, 768),
                (512, 256, 4096),
                (1024, 1024, 2048),
                (256, 1024, 8192),
                (896, 768, 512),
            ]
            .map(|s| (s, 1)),
        )
        .collect();
    for cfg in [NVFP4_GEMM_0, NVFP4_GEMM_0_UE8M0] {
        let kernel = cfg.build();
        let loaded = dev.load(&kernel);
        for &(mnk, n_launch) in &shapes {
            let (m, n, k) = mnk;
            let p = Problem::new(&cfg, m, n, k);
            let (sfa, sfb) = (p.sf_image(&p.sfa, m, cfg.sf_vec_size), p.sf_image(&p.sfb, n, cfg.sf_vec_size));
            let sf = if cfg.sf_dtype == DType::F8E4M3FN { typed::<F8E4M3FN> } else { typed::<F8E8M0FNU> };
            let want = p.reference(&cfg);
            let sentinel = KernelArg::buffer(&vec![CDtype::from_f32(-7.0); m * n]);
            let mut args = vec![
                KernelArg::scalar(m as i32),
                KernelArg::scalar(n as i32),
                KernelArg::scalar(k as i32),
                typed::<AbDtype>(&p.a),
                typed::<AbDtype>(&p.b),
                sf(&sfa),
                sf(&sfb),
                sentinel.clone(),
            ];
            soak.case(
                cfg.name,
                "default",
                &shape_str(mnk),
                &loaded,
                cfg.launch(m, n),
                &mut args,
                &[(7, sentinel)],
                n_launch,
                &|args| compare(&args[7].to_vec::<CDtype>(), &want, n),
            );
        }
    }
    soak.finish();
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_mixed_input_gemm() {
    use blackwell_mixed_input_gemm::*;
    let Some(dev) = blackwell("soak_mixed_input_gemm") else { return };
    let mut soak = Soak::new("soak_mixed_input_gemm");
    let n_rep = launches();
    // Candidates; those a configuration cannot implement (`can_implement`) are dropped, and
    // each kernel is built for its shape, so a few per configuration.
    let candidates = [
        (128, 64, 256),
        (256, 64, 256),
        (128, 128, 512),
        (384, 64, 256),
        (256, 192, 256),
        (128, 64, 4096),
        (512, 256, 1024),
        (640, 320, 768),
        (1024, 128, 512),
        (2048, 1024, 2048),
    ];
    for cfg in [
        MIXED_K_MAJOR_CONVERT,
        MIXED_K_MAJOR_SCALE,
        MIXED_M_MAJOR_CONVERT,
        MIXED_M_MAJOR_SCALE,
        MIXED_M_MAJOR_2CTA,
        MIXED_M_MAJOR_2CTA_SCALE,
    ] {
        let legal: Vec<_> = candidates.iter().copied().filter(|&s| cfg.can_implement(s)).collect();
        println!("SOAK {} legal shapes {legal:?}", cfg.name);
        for (i, &mnk) in legal.iter().enumerate() {
            let (m, n, _) = mnk;
            // The first legal shape of at most 1024x128 columns' worth is repeated.
            let n_launch = if i == legal.len().min(3) - 1 { n_rep } else { 1 };
            let kernel = cfg.build(mnk);
            let loaded = dev.load(&kernel);
            let p = Problem::new(&cfg, mnk, 0x5eed + m as u64);
            let (a, scale) = (p.a_device(&cfg), p.scale_device(&cfg));
            let b: Vec<BF16> = p.b.iter().map(|&x| BF16::from_f32(x)).collect();
            let want = p.reference(&cfg);
            let sentinel = KernelArg::buffer(&vec![-7.0f32; m * n]);
            let mut args = vec![KernelArg::buffer(&a)];
            if !scale.is_empty() {
                args.push(KernelArg::buffer(&scale));
            }
            args.push(KernelArg::buffer(&b));
            args.push(sentinel.clone());
            let out = args.len() - 1;
            soak.case(
                cfg.name,
                "default",
                &shape_str(mnk),
                &loaded,
                cfg.launch(m, n),
                &mut args,
                &[(out, sentinel)],
                n_launch,
                &|args| compare(&args[out].to_vec::<f32>(), &want, n),
            );
        }
    }
    soak.finish();
}

fn to_buffer(dtype: DType, v: &[f64]) -> KernelArg {
    match dtype {
        DType::F16 => KernelArg::buffer(&v.iter().map(|&x| F16::from_f32(x as f32)).collect::<Vec<_>>()),
        DType::BF16 => KernelArg::buffer(&v.iter().map(|&x| BF16::from_f32(x as f32)).collect::<Vec<_>>()),
        DType::F32 => KernelArg::buffer(&v.iter().map(|&x| x as f32).collect::<Vec<_>>()),
        _ => panic!("tensors of {dtype} are not supported"),
    }
}

fn from_buffer(dtype: DType, arg: &KernelArg) -> Vec<f64> {
    match dtype {
        DType::F16 => arg.to_vec::<F16>().iter().map(|x| x.to_f32() as f64).collect(),
        DType::BF16 => arg.to_vec::<BF16>().iter().map(|x| x.to_f32() as f64).collect(),
        DType::F32 => arg.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
        _ => unreachable!(),
    }
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_custom_efc_gemm() {
    use custom::efc::dense_gemm::small_ints;
    let Some(dev) = blackwell("soak_custom_efc_gemm") else { return };
    let mut soak = Soak::new("soak_custom_efc_gemm");
    let n_rep = launches();
    let mixed = custom::Dtypes { c: DType::F16, d: DType::F32, x: DType::BF16, y: DType::F16 };
    let f32s = custom::FLOAT32;
    let mut shapes: Vec<((usize, usize, usize), custom::Dtypes, usize)> =
        vec![(MEDIUM, f32s, n_rep), (MEDIUM, mixed, n_rep)];
    shapes.extend(
        [
            (8, 8, 8),
            (128, 128, 64),
            (136, 72, 72),
            (200, 136, 192),
            (264, 264, 136),
            (384, 128, 64),
            (128, 640, 64),
            (256, 256, 4096),
            (1024, 512, 256),
            (1000, 1528, 520),
        ]
        .map(|s| (s, f32s, 1)),
    );
    shapes.extend([(200, 136, 192), (264, 264, 136), (520, 1032, 136)].map(|s| (s, mixed, 1)));
    for cfg in [EFC_GEMM_1CTA, EFC_GEMM_2CTA_MULTICAST] {
        let (cm, cn) = cfg.cluster_shape_mn;
        let clusters = dev.max_active_clusters((cm * cn) as u32);
        for t in [f32s, mixed] {
            // One kernel per dtype set (the kernel does not depend on the shape).
            let gemm = DenseGemmEfc::new(cfg, custom::epilogue(t, LayoutEnum::ROW_MAJOR).unwrap());
            let kernel = gemm.build();
            let loaded = dev.load(&kernel);
            for &(mnk, _, n_launch) in shapes.iter().filter(|s| s.1.c == t.c) {
                let (m, n, k) = mnk;
                let roles = gemm.tensor_roles().unwrap();
                let mut inputs = HostArgs::default();
                for (name, v) in [("alpha", 1.0), ("beta", 2.0), ("x_factor", 3.0)] {
                    inputs.scalars.insert(name.into(), v);
                }
                inputs.tensors.insert("C".into(), small_ints(m * n, 44));
                inputs.tensors.insert("X".into(), small_ints(m * n, 55));
                inputs.tensors.insert("D".into(), vec![-7.0; m * n]);
                inputs.tensors.insert("Y".into(), vec![-7.0; m * n]);
                let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
                let mut args = vec![
                    KernelArg::scalar(m as i32),
                    KernelArg::scalar(n as i32),
                    KernelArg::scalar(k as i32),
                    to_buffer(AbDtype::DTYPE, &a),
                    to_buffer(AbDtype::DTYPE, &b),
                ];
                let mut resets = Vec::new();
                for (i, p) in gemm.efc.params().iter().enumerate() {
                    match p {
                        Param::Scalar(s) => args.push(KernelArg::scalar(inputs.scalars[s] as f32)),
                        Param::Tensor { name, dtype, .. } => {
                            let buf = to_buffer(*dtype, &inputs.tensors[name]);
                            // The written tensors are reset before each launch.
                            if name == "D" || name == "Y" {
                                resets.push((5 + i, buf.clone()));
                            }
                            args.push(buf);
                        }
                        Param::Config => unreachable!(),
                    }
                }
                let mut want = inputs.clone();
                gemm.efc.evaluate_on_host(&matmul(&a, &b, mnk, roles.d_layout), &mut want);
                let check = |args: &[KernelArg]| -> Option<String> {
                    let mut msgs = Vec::new();
                    for (i, p) in gemm.efc.params().iter().enumerate() {
                        let Param::Tensor { name, dtype, .. } = p else { continue };
                        let got = from_buffer(*dtype, &args[5 + i]);
                        if let Some(msg) = compare(&got, &want.tensors[name], n) {
                            msgs.push(format!("tensor `{name}`: {msg}"));
                        }
                    }
                    (!msgs.is_empty()).then(|| msgs.join(" | "))
                };
                soak.case(
                    cfg.name,
                    &format!("c={} d={} x={} y={}, {clusters} clusters", t.c, t.d, t.x, t.y),
                    &shape_str(mnk),
                    &loaded,
                    gemm.launch(m, n, clusters),
                    &mut args,
                    &resets,
                    n_launch,
                    &check,
                );
            }
        }
    }
    soak.finish();
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_cute_ext_persistent_gemm() {
    use persistent::{ONE_SM, TWO_SM, TWO_SM_MULTICAST, reference, small_ints};
    let Some(dev) = blackwell("soak_cute_ext_persistent_gemm") else { return };
    let mut soak = Soak::new("soak_cute_ext_persistent_gemm");
    let n_rep = launches();
    for cfg in [ONE_SM, TWO_SM, TWO_SM_MULTICAST] {
        let (cm, cn) = cfg.cluster_shape_mn;
        let device_clusters = dev.max_active_clusters((cm * cn) as u32);
        let kernel = cfg.build();
        let loaded = dev.load(&kernel);
        for (mnk, n_launch) in std::iter::once((MEDIUM, n_rep)).chain(SHAPES.iter().map(|&s| (s, 1))) {
            let (m, n, k) = mnk;
            let (a, b) = (small_ints(m * k, 1111), small_ints(n * k, 2222));
            let want = reference(&a, &b, m, n, k);
            let counts: &[u32] = if n_launch > 1 { &[device_clusters] } else { &[device_clusters, 1] };
            for &clusters in counts {
                let sentinel = KernelArg::f32(vec![-7.0; m * n]);
                let mut args = vec![
                    KernelArg::scalar(m as i32),
                    KernelArg::scalar(n as i32),
                    KernelArg::scalar(k as i32),
                    KernelArg::buffer(&a),
                    KernelArg::buffer(&b),
                    sentinel.clone(),
                ];
                soak.case(
                    cfg.name,
                    &format!("{clusters} clusters"),
                    &shape_str(mnk),
                    &loaded,
                    cfg.launch(m, n, clusters),
                    &mut args,
                    &[(5, sentinel)],
                    n_launch,
                    &|args| {
                        let got: Vec<f64> = args[5].to_vec::<f32>().iter().map(|&x| x as f64).collect();
                        compare(&got, &want, n)
                    },
                );
            }
        }
    }
    soak.finish();
}

#[test]
#[ignore = "a soak, for the hardware runs"]
fn soak_ts_gemm() {
    use ts_gemm_simple as g;
    let Some(dev) = blackwell("soak_ts_gemm") else { return };
    let mut soak = Soak::new("soak_ts_gemm");
    let n_rep = launches();

    fn run<T: g::IoDtype>(soak: &mut Soak, dev: &Device, shapes: &[((usize, usize, usize), usize)]) {
        for &(mnk, n_launch) in shapes {
            let (m, n, k) = mnk;
            let (av, bv) = (g::small_ints(m * k, 3), g::small_ints(n * k, 7));
            let to_t = |v: &[f32]| v.iter().map(|&x| T::from_f32(x)).collect::<Vec<T>>();
            // The kernel is built for its shape.
            let kernel = g::build::<T>(m, n, k, None);
            let loaded = dev.load(&kernel);
            let (want, _) = g::reference(&av, &bv, m, n, k);
            let want: Vec<f32> = want.iter().map(|&w| g::round::<T>(w as f32)).collect();
            let sentinel = KernelArg::buffer(&vec![T::from_f32(-7.0); m * n]);
            let mut args = vec![KernelArg::buffer(&to_t(&av)), KernelArg::buffer(&to_t(&bv)), sentinel.clone()];
            soak.case(
                "ts_gemm_simple",
                &format!("{}", T::DTYPE),
                &shape_str(mnk),
                &loaded,
                g::launch(m, n),
                &mut args,
                &[(2, sentinel)],
                n_launch,
                &|args| {
                    let got: Vec<f32> = args[2].to_vec::<T>().iter().map(|d| d.to_f32()).collect();
                    compare(&got, &want, n)
                },
            );
        }
    }
    let shapes: Vec<_> = std::iter::once((MEDIUM, n_rep))
        .chain(
            [(8, 8, 8), (128, 256, 64), (136, 72, 72), (200, 300, 136), (384, 128, 64), (256, 256, 4096)]
                .map(|s| (s, 1)),
        )
        .chain([(4096, 128, 256), (520, 1032, 136), (1000, 1528, 520)].map(|s| (s, 1)))
        .collect();
    run::<F16>(&mut soak, &dev, &shapes);
    run::<BF16>(&mut soak, &dev, &shapes);
    soak.finish();
}

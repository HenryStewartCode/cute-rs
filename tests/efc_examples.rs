//! The EFC epilogue-fusion GEMMs (the three
//! `*_custom_epilogue_dense_gemm` examples and their shared `examples/efc` module),
//! simulated and checked against EFC's host evaluation, plus the analysis, the rules and
//! the sm_100a PTX.

// Each example includes `examples/efc` itself, so the test compiles it three times.
#![allow(clippy::duplicate_mod)]
// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/activation_custom_epilogue_dense_gemm.rs"]
mod activation;
#[path = "../examples/custom_epilogue_dense_gemm.rs"]
mod custom;
#[path = "../examples/synthetic_custom_epilogue_dense_gemm.rs"]
mod synthetic;

use cute::prelude::*;
use cute::utils::LayoutEnum;
// Each example compiles its own copy of `examples/efc`, so each one's types are its own.
use activation::efc as ae;
use custom::efc::dense_gemm::{DenseGemmEfc, EFC_GEMM_1CTA, EFC_GEMM_2CTA, EFC_GEMM_2CTA_MULTICAST};
use custom::efc::{Efc, EfcError, HostArgs, Param, ParameterAttributes};
use synthetic::efc as se;

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// The analysis phase's read and written sets of each example, the scalars' attributes,
/// and the tensors that drive the tiling (the widest written and read ones, the first of
/// equals).
#[test]
fn efc_analysis_finds_the_read_and_written_tensors() {
    let c = custom::epilogue(custom::FLOAT32, LayoutEnum::ROW_MAJOR).unwrap();
    assert_eq!(c.read_tensor_names, names(&["C", "X"]));
    assert_eq!(c.written_tensor_names, names(&["D", "Y"]));
    assert_eq!(c.used_tensor_names, names(&["C", "D", "X", "Y"]));
    assert_eq!(c.parameter_attributes("alpha"), ParameterAttributes::default());
    assert_eq!(c.parameter_attributes("C"), ParameterAttributes { is_tensor: true, is_read: true, is_written: false });

    let s = synthetic::meta_epilogue(2, 3, DType::F32).unwrap();
    assert_eq!(s.read_tensor_names, names(&["read_t0", "read_t1"]));
    assert_eq!(s.written_tensor_names, names(&["written_t0", "written_t1", "written_t2"]));
    let s0 = synthetic::meta_epilogue(0, 1, DType::F32).unwrap();
    assert!(s0.read_tensor_names.is_empty());

    let a = activation::create_epilogue_function("gelu", DType::F16, DType::F16, DType::F16).unwrap();
    assert_eq!(a.read_tensor_names, names(&["C"]));
    assert_eq!(a.written_tensor_names, names(&["Aux", "D"]));
    let roles = ae::dense_gemm::DenseGemmEfc::new(ae::dense_gemm::EFC_GEMM_1CTA, a).tensor_roles().unwrap();
    assert_eq!((roles.d_name_bigger.as_str(), roles.c_name_bigger.as_deref()), ("Aux", Some("C")));

    let mixed = custom::Dtypes { c: DType::F16, d: DType::F16, x: DType::F32, y: DType::F32 };
    let roles = DenseGemmEfc::new(EFC_GEMM_1CTA, custom::epilogue(mixed, LayoutEnum::ROW_MAJOR).unwrap())
        .tensor_roles()
        .unwrap();
    assert_eq!((roles.d_name_bigger.as_str(), roles.d_dtype), ("Y", DType::F32));
    assert_eq!(roles.c_name_bigger.as_deref(), Some("X"));
}

/// The rules: `efc_config` first (and only first), unique names, at least one written
/// tensor, and one majorness for every tensor.
#[test]
fn efc_rules_are_diagnosed() {
    let noop = |_: &custom::efc::EfcConfig| {};
    let e = Efc::new(vec![Param::scalar("alpha"), Param::tensor("D", DType::F32)], noop).err();
    assert_eq!(e, Some(EfcError::MissingConfigParameter));
    assert!(e.unwrap().to_string().contains("efc_config"));
    let e = Efc::new(vec![Param::Config, Param::Config], noop).err();
    assert_eq!(e, Some(EfcError::MisplacedConfigParameter));
    let e = Efc::builder().scalar("x").tensor("x", DType::F32).build(noop).err();
    assert_eq!(e, Some(EfcError::BadParameterName("x".into())));
    let e = Efc::builder().tensor("a", DType::F32).build(noop).err();
    assert_eq!(e, Some(EfcError::BadParameterName("a".into())));

    // Reading only.
    assert_eq!(synthetic::meta_epilogue(1, 0, DType::F32).err(), Some(se::EfcError::NoWrittenTensor));
    let e = Efc::builder().tensor("C", DType::F32).build(|c| drop(c.load("C"))).err();
    assert_eq!(e, Some(EfcError::NoWrittenTensor));

    // X is M-major, D N-major.
    let efc = Efc::builder()
        .tensor_with_layout("X", DType::F32, LayoutEnum::COL_MAJOR)
        .tensor("D", DType::F32)
        .build(|c| c.store("D", c.accum() + c.load("X")))
        .unwrap();
    let e = DenseGemmEfc::new(EFC_GEMM_1CTA, efc).tensor_roles().err().unwrap();
    assert_eq!(
        e,
        EfcError::LayoutMismatch { tensor: "X".into(), layout: LayoutEnum::COL_MAJOR, d_layout: LayoutEnum::ROW_MAJOR }
    );
    assert!(e.to_string().contains("`X`"));
}

/// The host phase on a few values: the custom epilogue, a scalar stored to a tensor, and
/// the rounding to the tensor's type.
#[test]
fn efc_host_evaluation() {
    let efc = custom::epilogue(custom::Dtypes { y: DType::F16, ..custom::FLOAT32 }, LayoutEnum::ROW_MAJOR).unwrap();
    let mut args = HostArgs::default();
    for (n, v) in [("alpha", 1.0), ("beta", 2.0), ("x_factor", 3.0)] {
        args.scalars.insert(n.into(), v);
    }
    args.tensors.insert("C".into(), vec![1.0, -3.0, 0.5]);
    args.tensors.insert("X".into(), vec![1.0, 1.0, -1.0]);
    efc.evaluate_on_host(&[2.0, 1.0, 1000.1], &mut args);
    // relu(acc + 2 C) + 3 X
    assert_eq!(args.tensors["D"], vec![7.0, 3.0, (1000.1f64 + 1.0 - 3.0) as f32 as f64]);
    // Y = acc, in Float16
    assert_eq!(args.tensors["Y"], vec![2.0, 1.0, 1000.0]);
}

/// The synthetic epilogue: two read and three written Float32 tensors (with `where` and
/// `full_like`) on one CTA, and the linear `D = alpha acc + beta + 5000` on a CTA pair;
/// ragged problems, fewer clusters than tiles; exact.
#[test]
fn synthetic_efc_gemm_matches_the_host() {
    let opts = SimOptions::default();
    let run = synthetic::run(se::dense_gemm::EFC_GEMM_1CTA, (200, 136, 64), 2, 3, DType::F32, (2.0, 1.0), 2, &opts);
    run.assert_close("synthetic 1cta 2r3w", 0.0, 0.0);
    let run = synthetic::run(se::dense_gemm::EFC_GEMM_2CTA, (264, 136, 64), 0, 1, DType::F32, (2.0, 1.0), 1, &opts);
    run.assert_close("synthetic 2cta 0r1w", 0.0, 0.0);
}

/// Every activation function on Float16 C, Aux and D (two tiles, N ragged): exact for the
/// piecewise-linear ones, within a Float16 ulp (and GELU's approximation) otherwise.
#[test]
fn activation_efc_gemm_matches_the_host_for_each_activation() {
    for name in ae::ACTIVATION_FUNCTIONS {
        let run = activation::run(ae::dense_gemm::EFC_GEMM_1CTA, name, (128, 136, 64), 0.25, 1, &SimOptions::default());
        let (atol, rtol) = activation::tolerance(name);
        run.assert_close(name, atol, rtol);
    }
}

/// The custom epilogue (two read, two written tensors): on one CTA with mixed element types,
/// on a 2x2 cluster of CTA pairs with multicast, and M-major (Float16: an M-major Float32
/// tile takes the SW128_32B swizzle atom, whose TMA mode cute-rs lacks) on a CTA pair; exact.
#[test]
fn custom_efc_gemm_matches_the_host() {
    let opts = SimOptions::default();
    let mixed = custom::Dtypes { c: DType::F16, d: DType::F32, x: DType::BF16, y: DType::F16 };
    let run = custom::run(EFC_GEMM_1CTA, (200, 136, 64), mixed, LayoutEnum::ROW_MAJOR, 2, &opts, &[]);
    run.assert_close("custom 1cta mixed", 0.0, 0.0);
    let run =
        custom::run(EFC_GEMM_2CTA_MULTICAST, (264, 264, 64), custom::FLOAT32, LayoutEnum::ROW_MAJOR, 2, &opts, &[]);
    run.assert_close("custom 2cta multicast", 0.0, 0.0);
    let f16 = custom::Dtypes { c: DType::F16, d: DType::F16, x: DType::F16, y: DType::F16 };
    let run = custom::run(EFC_GEMM_2CTA, (264, 72, 64), f16, LayoutEnum::COL_MAJOR, 1, &opts, &[]);
    run.assert_close("custom 2cta m-major", 0.0, 0.0);
}

/// The race detector stays clean and the results do not depend on the schedule.
#[test]
fn custom_efc_gemm_is_race_free_under_several_schedules() {
    let opts = SimOptions { detect_races: true, ..Default::default() };
    let run = custom::run(EFC_GEMM_1CTA, (136, 136, 128), custom::FLOAT32, LayoutEnum::ROW_MAJOR, 1, &opts, &[3, 11]);
    run.assert_close("custom race", 0.0, 0.0);
}

/// The MLIR and sm_100a PTX of each example: one TMA load per read tensor and one TMA store
/// per written tensor for each epilogue subtile (the subtile loop is unrolled), beside the
/// two loads of A and B; assembled by `ptxas -arch=sm_100a` when it is on PATH.
#[test]
fn efc_gemms_in_mlir_and_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    // (name, kernel, subtiles per CTA tile, read tensors, written tensors)
    let probe = |f: &dyn Fn() -> usize| {
        let mut subtiles = 0;
        Kernel::build("probe", || subtiles = f());
        subtiles
    };
    let s = se::dense_gemm::DenseGemmEfc::new(
        se::dense_gemm::EFC_GEMM_1CTA,
        synthetic::meta_epilogue(2, 3, DType::F32).unwrap(),
    );
    let a = ae::dense_gemm::DenseGemmEfc::new(
        ae::dense_gemm::EFC_GEMM_2CTA,
        activation::create_epilogue_function("gelu", DType::F16, DType::F16, DType::F16).unwrap(),
    );
    let c =
        DenseGemmEfc::new(EFC_GEMM_2CTA_MULTICAST, custom::epilogue(custom::FLOAT32, LayoutEnum::ROW_MAJOR).unwrap());
    let gemms = [
        (s.cfg.name, s.build(), probe(&|| se::dense_gemm::DenseGemmEfc::subtile_count(&s.setup())), 2, 3),
        (a.cfg.name, a.build(), probe(&|| ae::dense_gemm::DenseGemmEfc::subtile_count(&a.setup())), 1, 2),
        (c.cfg.name, c.build(), probe(&|| DenseGemmEfc::subtile_count(&c.setup())), 2, 2),
    ];
    for (name, kernel, subtiles, reads, writes) in gemms {
        assert!(kernel.to_mlir().contains("tcgen05"), "{name}: no tcgen05 in the MLIR");
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
        let ptx = compiled.ptx();
        let tma: Vec<&str> = ptx.lines().filter(|l| l.contains("cp.async.bulk.tensor")).collect();
        let loads =
            tma.iter().filter(|l| l.contains(".shared::cluster.global") || l.contains(".shared::cta.global")).count();
        let stores = tma.iter().filter(|l| l.contains(".global.shared::cta")).count();
        assert_eq!(loads, 2 + reads * subtiles, "{name}: TMA loads in the PTX:\n{}", tma.join("\n"));
        assert_eq!(stores, writes * subtiles, "{name}: TMA stores in the PTX:\n{}", tma.join("\n"));
        let ptxas = std::env::var_os("PATH")
            .and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
        let Some(ptxas) = ptxas else { continue };
        let base = std::env::temp_dir().join(format!("cute-efc-{}-{name}", std::process::id()));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

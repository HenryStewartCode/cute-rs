//! Tensor maps and grid-constant parameters on a real GPU (roadmap part
//! m7-tma-descriptor): the driver's `cuTensorMapEncodeTiled` against cute-rs's validator,
//! the by-value grid-constant ABI read through the parameter's address, and hidden and
//! host-bound descriptors through the launcher. Needs the `cuda` feature; skipped (with a
//! note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_tma
//! ```

mod tma_atom_cases;
mod tma_kernels;

use cute::cuda::{CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED, CudaContext, DeviceArg};
use cute::prelude::*;
use tma_kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn one_block() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [32, 1, 1])
}

/// Descriptors over `base`, each valid or breaking one rule, with a name.
/// Cases CUDA's documentation forbids but some drivers encode anyway; cute-rs keeps refusing them.
const DOCUMENTED_STRICTER: &[&str] = &["i32b interleave with s64b"];

fn cases(base: u64) -> Vec<(&'static str, TmaDesc)> {
    let mut ok = TmaDesc::new(TensorMapDataType::float16, 2);
    ok.global_address = base;
    ok.global_dims[..2].copy_from_slice(&[128, 64]);
    ok.global_strides[0] = 256;
    ok.box_dims[..2].copy_from_slice(&[64, 8]);
    let mut out = vec![("valid 2-D f16", ok.clone())];
    let mut push = |name, f: &dyn Fn(&mut TmaDesc)| {
        let mut d = ok.clone();
        f(&mut d);
        out.push((name, d));
    };
    push("valid s128b swizzle", &|d| d.swizzle = TensorMapSwizzle::s128b);
    push("valid element stride 2", &|d| d.element_strides[1] = 2);
    push("valid L2 256B and NaN fill", &|d| {
        d.l2_promotion = TensorMapL2Promotion::l2_256b;
        d.oob_fill = TensorMapFloatOOBFill::nan_request_zero_fma;
    });
    push("valid rank 1", &|d| {
        d.rank = 1;
        d.global_dims[1] = 0;
        d.box_dims[1] = 0;
    });
    push("valid 2^32 extent", &|d| d.global_dims[1] = 1 << 32);
    push("valid box 256", &|d| d.box_dims[1] = 256);
    push("misaligned base (8 bytes)", &|d| d.global_address += 8);
    push("stride 40 bytes", &|d| d.global_strides[0] = 40);
    push("stride 2^40", &|d| d.global_strides[0] = 1 << 40);
    push("extent 0", &|d| d.global_dims[1] = 0);
    push("extent 2^32 + 1", &|d| d.global_dims[1] = (1 << 32) + 1);
    push("box 257", &|d| d.box_dims[1] = 257);
    push("box 0", &|d| d.box_dims[1] = 0);
    push("inner box 8 bytes", &|d| d.box_dims[0] = 4);
    push("inner box 128 bytes with s64b", &|d| d.swizzle = TensorMapSwizzle::s64b);
    push("inner box 128 bytes with s32b", &|d| d.swizzle = TensorMapSwizzle::s32b);
    push("element stride 9", &|d| d.element_strides[1] = 9);
    push("element stride 0", &|d| d.element_strides[1] = 0);
    push("rank 6", &|d| d.rank = 6);
    push("rank 0", &|d| d.rank = 0);
    push("interleave at rank 2", &|d| d.interleave = TensorMapInterleave::i16b);
    push("NaN fill on int32", &|d| {
        d.data_type = TensorMapDataType::int32;
        d.box_dims[0] = 32;
        d.oob_fill = TensorMapFloatOOBFill::nan_request_zero_fma;
    });
    let mut i3 = TmaDesc::new(TensorMapDataType::float16, 3);
    i3.global_address = base;
    i3.global_dims[..3].copy_from_slice(&[16, 16, 16]);
    i3.global_strides[..2].copy_from_slice(&[32, 512]);
    i3.box_dims[..3].copy_from_slice(&[16, 8, 8]);
    i3.interleave = TensorMapInterleave::i32b;
    i3.swizzle = TensorMapSwizzle::s32b;
    out.push(("valid i32b interleave with s32b", i3.clone()));
    let mut d = i3.clone();
    d.swizzle = TensorMapSwizzle::s64b;
    out.push(("i32b interleave with s64b", d));
    let mut d = i3.clone();
    d.global_address += 16;
    out.push(("i32b interleave, 16-byte base", d));
    let mut d = i3.clone();
    d.global_strides[0] = 48;
    out.push(("i32b interleave, stride 48", d));
    out
}

#[test]
fn driver_encoding_agrees_with_the_validator() {
    let Some((ctx, _tc)) = device() else { return };
    if !ctx.has_tensor_map_encode() {
        eprintln!("skipping: the driver has no cuTensorMapEncodeTiled");
        return;
    }
    let buf = ctx.alloc_bytes(1 << 20).unwrap();
    let base = buf.as_raw();
    let (major, minor) = ctx.compute_capability();
    // Below sm_90 the driver encodes nothing (CUDA_ERROR_NOT_SUPPORTED) but still checks a few
    // fields first (CUDA_ERROR_INVALID_VALUE): those it rejects, cute-rs must reject too.
    // From sm_90 the two must agree on every case.
    let full = major >= 9;
    let mut disagree = Vec::new();
    for (name, d) in cases(base) {
        let ours = d.validate();
        let driver = ctx.encode_tensor_map_unchecked(&d);
        let code = driver.as_ref().err().and_then(|e| e.code);
        println!(
            "sm_{major}{minor} {name:40} cute-rs {:8} driver {}",
            if ours.is_ok() { "accepts" } else { "rejects" },
            match &driver {
                Ok(_) => "accepts".to_string(),
                Err(e) => format!("rejects ({:?})", e.code),
            }
        );
        let bad = if full {
            // cute-rs may refuse what CUDA's documentation forbids even where a driver accepts
            // it: cuda.h says the i32b interleave needs the s32b swizzle, yet driver 595 on
            // sm_120 encodes i32b with s64b.
            let documented_stricter = DOCUMENTED_STRICTER.contains(&name) && ours.is_err();
            ours.is_ok() != driver.is_ok() && !documented_stricter
        } else {
            assert!(driver.is_err(), "{name}: sm_{major}{minor} encoded a tensor map");
            code == Some(CUDA_ERROR_INVALID_VALUE) && ours.is_ok()
                || !matches!(code, Some(CUDA_ERROR_INVALID_VALUE | CUDA_ERROR_NOT_SUPPORTED))
        };
        if bad {
            disagree.push(format!("{name}: cute-rs {ours:?}, driver {:?}", driver.map(|_| ()).map_err(|e| e.message)));
        }
    }
    assert!(disagree.is_empty(), "the driver and the validator disagree:\n{}", disagree.join("\n"));
    if full {
        // A valid descriptor encodes to 128 bytes that are not all zero.
        let enc = ctx.encode_tensor_map(&cases(base)[0].1).unwrap();
        assert!(enc.0.iter().any(|&b| b != 0));
    } else {
        let e = ctx.encode_tensor_map(&cases(base)[0].1).unwrap_err();
        assert!(e.message.contains("compute capability 9.0 or newer"), "{e}");
    }
}

#[test]
fn grid_constant_bytes_reach_the_kernel_in_place() {
    let Some((ctx, tc)) = device() else { return };
    let k = grid_constant_copy_kernel();
    let blob = grid_constant_blob();
    let mut args = vec![KernelArg::by_value(blob.clone()), KernelArg::buffer(&[0u64; 16])];
    k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got: Vec<u8> = args[1].to_vec::<u64>().iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(got, blob);
    let ptx = k.compile_ptx(&tc, ctx.best_target(&tc).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains(".param .align 64 .b8 grid_constant_copy_param_0[128]"), "{ptx}");
    assert!(ptx.contains("cvta.param.u64"), "{ptx}");
}

#[test]
fn hidden_and_host_bound_tensor_maps_launch() {
    let Some((ctx, tc)) = device() else { return };
    if !ctx.has_tensor_map_encode() {
        eprintln!("skipping: the driver has no cuTensorMapEncodeTiled");
        return;
    }
    // Hidden: encoded by the launcher from `a`'s device address and the scalars.
    let k = hidden_desc_kernel();
    let mut args = hidden_desc_args(100, 72);
    if ctx.compute_capability().0 < 9 {
        // The descriptor is checked by cute-rs and refused by the driver, before the launch.
        let e = k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_err().to_string();
        assert!(e.contains("a_tma_desc_0") && e.contains("compute capability 9.0 or newer"), "{e}");
        let mut args = hidden_desc_args(0, 64);
        let e = k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_err().to_string();
        assert!(e.contains("a_tma_desc_0") && e.contains("dimension 1 has 0 elements"), "{e}");
        eprintln!(
            "sm_{}{}: tensor-map launches need sm_90; checked the refusals only",
            ctx.compute_capability().0,
            ctx.compute_capability().1
        );
        return;
    }
    k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let words = args[3].to_vec::<u64>();
    assert!(words.iter().any(|&w| w != 0), "the driver's descriptor reached the kernel: {words:x?}");
    let addr = args[4].to_vec::<i64>()[0] as u64;
    assert!(addr.is_multiple_of(64), "the grid constant's generic address {addr:#x} is 64-byte aligned");
    // A launch-time violation is reported before the launch, naming the parameter.
    let mut args = hidden_desc_args(0, 64);
    let e = k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_err().to_string();
    assert!(e.contains("a_tma_desc_0") && e.contains("dimension 1 has 0 elements"), "{e}");

    // Host-bound: the driver's encoding of a host-built map over a device buffer.
    let k = explicit_desc_kernel();
    let buf = ctx.alloc_bytes(256 * 64 * 2).unwrap();
    let map = create_tensor_map_tiled(
        buf.as_raw(),
        cute::ir::DType::F16,
        &[256, 64],
        &[32],
        &[64, 16],
        &TensorMapOptions::new().swizzle(TensorMapSwizzle::s128b),
    )
    .unwrap();
    let expected = ctx.encode_tensor_map(map.desc()).unwrap();
    let compiled = k
        .compile_with(&tc, &CompileOptions::new().gpu_arch(ctx.best_target(&tc).unwrap()))
        .unwrap_or_else(|e| panic!("{e}"));
    let f = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    let out = ctx.alloc_bytes(128).unwrap();
    let arg = KernelArg::tensor_map(&map);
    ctx.launch_on_device(&k, &f, one_block(), &[DeviceArg::ByValue(arg_bytes(&arg)), DeviceArg::Buffer(&out)])
        .unwrap_or_else(|e| panic!("{e}"));
    let got: Vec<u8> = out.to_vec::<u8>().unwrap();
    assert_eq!(got, expected.0.to_vec(), "the kernel sees the driver's encoding, not the simulator's");
}

fn arg_bytes(arg: &KernelArg) -> Vec<u8> {
    match arg {
        KernelArg::Bytes { data, .. } => data.clone(),
        _ => unreachable!(),
    }
}

/// The descriptor kernels assemble for sm_90 with the toolkit's ptxas: the 128-byte,
/// 64-byte-aligned grid constant and its in-place address (`cvta.param`, PTX 7.7).
#[test]
fn descriptor_kernels_assemble_for_sm90() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    if tc.cuda_tools().ptxas.is_none() {
        eprintln!("skipping: no ptxas");
        return;
    }
    for k in [
        hidden_desc_kernel(),
        explicit_desc_kernel(),
        grid_constant_copy_kernel(),
        desc_address_to_inline_ptx_kernel(),
        tma_atom_desc_kernel(),
    ] {
        for arch in [Arch::Sm90, Arch::Sm90a] {
            let options = CompileOptions::new().gpu_arch(arch).use_ptxas(true);
            let compiled = k.compile_with(&tc, &options).unwrap_or_else(|e| panic!("{} for {arch}: {e}", k.name()));
            assert!(compiled.cubin().is_some(), "{} for {arch}: no cubin", k.name());
            assert!(compiled.ptx().contains(".param .align 64 .b8"), "{}", compiled.ptx());
        }
    }
}

/// The descriptors of TMA atoms (roadmap part m7-tma-atom) over a device buffer: the
/// K- and M-major GMMA tiles of f16, bf16 and f32. From sm_90 the driver encodes every one;
/// below it encodes nothing (`CUDA_ERROR_NOT_SUPPORTED`) but must not find a field invalid
/// (`CUDA_ERROR_INVALID_VALUE`). The hidden descriptor of an atom kernel goes through the
/// launcher as `tma_descriptor`'s does.
#[test]
fn tma_atom_descriptors_encode() {
    let Some((ctx, tc)) = device() else { return };
    if !ctx.has_tensor_map_encode() {
        eprintln!("skipping: the driver has no cuTensorMapEncodeTiled");
        return;
    }
    let buf = ctx.alloc_bytes(512 * 256 * 4).unwrap();
    let (major, minor) = ctx.compute_capability();
    let cases = tma_atom_cases::gmma_desc_cases();
    assert_eq!(cases.len(), 48);
    for case in &cases {
        let d = case.spec.evaluate(buf.as_raw(), &|v| case.env.get(v)).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        match ctx.encode_tensor_map_unchecked(&d) {
            Ok(enc) => assert!(enc.0.iter().any(|&b| b != 0), "{}", case.name),
            Err(e) => {
                assert!(major < 9, "sm_{major}{minor} refused {}: {e}", case.name);
                assert_eq!(e.code, Some(CUDA_ERROR_NOT_SUPPORTED), "sm_{major}{minor} {}: {e}", case.name);
            }
        }
    }
    println!("sm_{major}{minor}: {} TMA atom descriptors checked by the driver", cases.len());
    let k = tma_atom_desc_kernel();
    let mut args = tma_atom_desc_args(300, 192);
    if major < 9 {
        let e = k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_err().to_string();
        assert!(e.contains("a_tma_desc_0") && e.contains("compute capability 9.0 or newer"), "{e}");
        return;
    }
    k.run_cuda(&ctx, &tc, one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert!(args[3].to_vec::<u64>().iter().any(|&w| w != 0));
}

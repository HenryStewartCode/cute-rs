//! TMA descriptors as kernel parameters (roadmap part m7-tma-descriptor) on the simulator
//! and in the emitted code: hidden descriptors computed at launch from a tensor parameter,
//! host-bound ones, grid-constant parameters read through their generic address, and the
//! build- and launch-time checks.

mod tma_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;
use cute::sim::SIM_GLOBAL_BASE;
use tma_kernels::*;

fn one_block() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [32, 1, 1])
}

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn words_to_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
fn hidden_descriptor_is_appended_and_encoded_per_launch() {
    let k = hidden_desc_kernel();
    let sig = k.signature();
    // The hidden parameter comes last; launches pass the kernel's own five arguments.
    assert_eq!(sig.params.len(), 6);
    assert_eq!(sig.user_param_count(), 5);
    assert_eq!(sig.params[5].name, "a_tma_desc_0");
    assert_eq!(sig.params[5].kind, ParamKind::ByValue { size: 128, align: 64, grid_constant: true });
    assert_eq!(sig.tensor_maps.len(), 1);
    assert_eq!(sig.tensor_maps[0].source.as_ref().unwrap().tensor_param, 2);
    assert!(sig.to_string().ends_with("a_tma_desc_0: byval[128, align 64] grid_constant)"), "{sig}");

    let mut args = hidden_desc_args(100, 72);
    k.simulate(one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let desc = TmaDesc::from_sim_bytes(&words_to_bytes(&args[3].to_vec::<u64>())).unwrap();
    // `a` is the first global buffer of the launch.
    assert_eq!(desc.global_address, SIM_GLOBAL_BASE);
    assert_eq!(desc.data_type, TensorMapDataType::float16);
    assert_eq!(desc.rank, 2);
    assert_eq!(&desc.global_dims[..2], &[64, 100]);
    assert_eq!(desc.global_strides[0], 144);
    assert_eq!(&desc.box_dims[..2], &[64, 8]);
    assert_eq!(desc.swizzle, TensorMapSwizzle::s128b);
    // The generic address of the grid constant is the parameter's own.
    let addr = args[4].to_vec::<i64>()[0] as u64;
    assert!(addr >= SIM_GLOBAL_BASE && addr.is_multiple_of(64), "{addr:#x}");

    // Another launch, other extents: the descriptor follows.
    let mut args = hidden_desc_args(7, 64);
    k.simulate(one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let desc = TmaDesc::from_sim_bytes(&words_to_bytes(&args[3].to_vec::<u64>())).unwrap();
    assert_eq!((desc.global_dims[1], desc.global_strides[0]), (7, 128));
}

#[test]
fn launch_time_descriptor_errors_name_the_parameter() {
    let k = hidden_desc_kernel();
    // No rows: a zero extent.
    let mut args = hidden_desc_args(0, 64);
    let e = k.simulate(one_block(), &mut args).unwrap_err();
    assert!(e.message.contains("the tensor map of `a` (hidden parameter 5, `a_tma_desc_0`)"), "{e}");
    assert!(e.message.contains("dimension 1 has 0 elements"), "{e}");
    // A leading dimension with no divisibility promise: 66 halves are 132 bytes.
    let loose = Kernel::builder("loose_ld").block_dim([32, 1, 1]).build(|| {
        let ld = param_extent("ld", 1);
        let a = param_tensor::<F16>("a", Layout::new((16, 64), (ld, 1)));
        let _ = tma_descriptor(&a, &[8, 64], None, &TensorMapOptions::new());
    });
    let mut args = vec![KernelArg::scalar(66i32), KernelArg::f16_from_f32(&[0.0; 16 * 66])];
    let e = loose.simulate(one_block(), &mut args).unwrap_err();
    assert!(e.message.contains("`a_tma_desc_0`") && e.message.contains("132 bytes, not a multiple of 16"), "{e}");
    let mut args = vec![KernelArg::scalar(72i32), KernelArg::f16_from_f32(&[0.0; 16 * 72])];
    loose.simulate(one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    // Too many arguments: the hidden parameter is not passed.
    let mut args = hidden_desc_args(4, 64);
    args.push(KernelArg::by_value(vec![0; 128]));
    let e = k.simulate(one_block(), &mut args).unwrap_err();
    assert!(e.message.contains("takes 5 arguments, 6 were given"), "{e}");
}

#[test]
fn host_bound_tensor_map_reaches_the_kernel() {
    let k = explicit_desc_kernel();
    let sig = k.signature();
    assert_eq!(sig.user_param_count(), 2);
    assert!(sig.tensor_maps[0].source.is_none());
    let opts = TensorMapOptions::new().swizzle(TensorMapSwizzle::s64b).l2_promotion(TensorMapL2Promotion::l2_128b);
    let map =
        create_tensor_map_tiled(0x7f00_0000_1000, cute::ir::DType::F16, &[256, 64], &[32], &[32, 16], &opts).unwrap();
    let mut args = vec![KernelArg::tensor_map(&map), KernelArg::buffer(&[0u64; 16])];
    k.simulate(one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got = TmaDesc::from_sim_bytes(&words_to_bytes(&args[1].to_vec::<u64>())).unwrap();
    assert_eq!(&got, map.desc());
    // Plain bytes are not a tensor map.
    let mut args = vec![KernelArg::by_value(vec![0; 128]), KernelArg::buffer(&[0u64; 16])];
    let e = k.simulate(one_block(), &mut args).unwrap_err();
    assert!(e.message.contains("argument 0 (`desc`) is a tensor map"), "{e}");
}

#[test]
fn grid_constant_bytes_read_through_their_address() {
    let k = grid_constant_copy_kernel();
    let blob = grid_constant_blob();
    let mut args = vec![KernelArg::by_value(blob.clone()), KernelArg::buffer(&[0u64; 16])];
    k.simulate(one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(words_to_bytes(&args[1].to_vec::<u64>()), blob);
}

#[test]
fn grid_constants_are_read_only_in_the_simulator() {
    let k = Kernel::builder("write_grid_constant").block_dim([32, 1, 1]).build(|| {
        let p = param_bytes("blob", 128, 64, true);
        if_(thread_idx().x.eq(0), || {
            let at = make_ptr::<u64>(p.address(), AddressSpace::Generic, Some(8));
            at.store(Uint64::from(1u64));
        });
    });
    let mut args = vec![KernelArg::by_value(vec![0; 128])];
    let e = k.simulate(one_block(), &mut args).unwrap_err();
    assert!(e.message.contains("read-only"), "{e}");
}

#[test]
fn descriptor_build_diagnostics() {
    // A start that is not 16-byte aligned (a tensor offset by one float).
    let msg = build_panic(|| {
        Kernel::build("misaligned", || {
            let a = param_tensor::<f32>("a", Layout::row_major((64, 32)));
            let shifted = domain_offset((0, 1), &a);
            let _ = tma_descriptor(&shifted, &[8, 32], None, &TensorMapOptions::new());
        });
    });
    assert!(msg.contains("not 16-byte aligned"), "{msg}");
    assert!(msg.contains("tests/tma_desc.rs"), "{msg}");
    // A row stride of 36 bytes (9 floats).
    let msg = build_panic(|| {
        Kernel::build("stride", || {
            let a = param_tensor::<f32>("a", Layout::new((64, 8), (9, 1)));
            let _ = tma_descriptor(&a, &[8, 8], None, &TensorMapOptions::new());
        });
    });
    assert!(msg.contains("the stride of dimension 1 is 36 bytes, not a multiple of 16"), "{msg}");
    // A box of 512 rows.
    let msg = build_panic(|| {
        Kernel::build("big_box", || {
            let a = param_tensor::<f32>("a", Layout::row_major((1024, 32)));
            let _ = tma_descriptor(&a, &[512, 32], None, &TensorMapOptions::new());
        });
    });
    assert!(msg.contains("box dimension 1 is 512 elements, outside 1 to 256"), "{msg}");
    // 64 floats are 256 bytes, wider than the 128-byte swizzle.
    let msg = build_panic(|| {
        Kernel::build("wide_box", || {
            let a = param_tensor::<f32>("a", Layout::row_major((64, 64)));
            let opts = TensorMapOptions::new().swizzle(TensorMapSwizzle::s128b);
            let _ = tma_descriptor(&a, &[8, 64], None, &opts);
        });
    });
    assert!(
        msg.contains("the s128b swizzle needs the box's inner dimension to be at most 128 bytes, got 256"),
        "{msg}"
    );
    // Six modes.
    let msg = build_panic(|| {
        Kernel::build("rank6", || {
            let a = param_tensor::<f32>("a", Layout::col_major((4, 2, 2, 2, 2, 2)));
            let _ = tma_descriptor(&a, &[4, 2, 2, 2, 2, 2], None, &TensorMapOptions::new());
        });
    });
    assert!(msg.contains("rank 6 is outside 1 to 5"), "{msg}");
    // An extent computed in the kernel.
    let msg = build_panic(|| {
        Kernel::build("in_kernel", || {
            let m = param_extent("m", 1);
            let rows = m * 2 + thread_idx().x;
            let a = param_tensor::<f32>("a", Layout::new((rows, 32), (32, 1)));
            let _ = tma_descriptor(&a, &[8, 32], None, &TensorMapOptions::new());
        });
    });
    assert!(msg.contains("a value computed in the kernel"), "{msg}");
    // A shared-memory tensor is not a parameter.
    let msg = build_panic(|| {
        Kernel::build("smem", || {
            let s = make_smem_tensor::<f32>(Layout::row_major((8, 32)));
            let _ = tma_descriptor(&s, &[8, 32], None, &TensorMapOptions::new());
        });
    });
    assert!(msg.contains("not a tensor parameter"), "{msg}");
}

#[test]
fn hidden_descriptor_lowers_to_a_grid_constant() {
    let k = desc_address_to_inline_ptx_kernel();
    let mlir = k.to_mlir();
    assert!(mlir.contains("%arg_a_tma_desc_0: memref<128xi8> {llvm.align = 64 : i64}"), "{mlir}");
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping the PTX checks: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = k.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains(".param .align 64 .b8 desc_inline_ptx_param_2[128]"), "{ptx}");
    // The address of the grid constant is taken in place (cvta.param) and reaches the PTX.
    assert!(ptx.contains("cvta.param.u64"), "{ptx}");
    assert!(ptx.contains("mov.b64"), "{ptx}");
    // The simulator agrees: the address is the hidden parameter's.
    let mut args = vec![KernelArg::f32(vec![0.0; 128 * 32]), KernelArg::buffer(&[0u64; 1])];
    k.simulate(one_block(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert!(args[1].to_vec::<u64>()[0] > SIM_GLOBAL_BASE);
}

/// An export's signature text has no form for tensor-map parameters, so exporting a kernel
/// with one fails instead of writing a manifest that would treat the hidden descriptor as
/// an argument and pass the simulator's encoding to the driver.
#[test]
fn kernels_with_tensor_maps_are_not_exported() {
    for k in [hidden_desc_kernel(), explicit_desc_kernel()] {
        let e = k.signature().text_error().expect("a tensor-map kernel has no signature text");
        assert!(e.contains("tensor-map parameters"), "{e}");
    }
    assert!(grid_constant_copy_kernel().signature().text_error().is_none());
}

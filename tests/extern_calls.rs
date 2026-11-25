//! External device-function calls and standalone device functions (roadmap part
//! m11-extern-ffi-device-fn): mangling, the simulator, MLIR, PTX with linked bitcode
//! (`tests/data/extern_funcs.ll`, assembled with `llvm-as` at test time; skipped without it
//! or without `mlir-opt`), and diagnostics. The GPU runs are in `tests/gpu_extern.rs`.

#[path = "extern_kernels/mod.rs"]
mod extern_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::ffi::{AddressSpace, CallArg, ExternOverloads, MangledParam, mangle, mangle_symbol};
use cute::prelude::*;
use extern_kernels::*;

fn panic_message(f: impl FnOnce()) -> String {
    let e = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn mangling_follows_cute_dsl() {
    assert_eq!(mangle("f32"), "f32");
    assert_eq!(mangle("!llvm.ptr<1>"), "_21llvm_2Eptr_3C1_3E");
    assert_eq!(mangle("a b"), "a_20b");
    assert_eq!(mangle("-12"), "_2D12");
    let t = |s: &str| MangledParam::Type(s.to_string());
    assert_eq!(mangle_symbol("add", &[t("f32"), t("f32")]), "add_f32_f32");
    assert_eq!(mangle_symbol("add", &[t("i32"), MangledParam::Const("i32".into(), 4)]), "add_i32_i32_4");
    assert_eq!(mangle_symbol("nop", &[]), "nop_");
    assert_eq!(mangle_symbol("ld", &[t("!llvm.ptr<3>"), t("i64")]), "ld__21llvm_2Eptr_3C3_3E_i64");
}

#[test]
fn extern_calls_simulate_with_rust_sims() {
    for inline in [true, false] {
        let k = bitcode_kernel("unused.bc".as_ref(), inline);
        let (x, y) = inputs();
        let mut args = vec![KernelArg::f32(x.clone()), KernelArg::f32(y.clone())];
        k.simulate(LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]), &mut args).unwrap();
        assert_eq!(args[0].as_f32(), expected(&x, &y), "inline {inline}");
    }
}

#[test]
fn extern_mlir_has_prototypes_and_calls() {
    let k = bitcode_kernel("unused.bc".as_ref(), false);
    let mlir = k.to_mlir();
    assert!(
        mlir.contains("func.func private @ext_scale(f32) -> f32 attributes {passthrough = [\"convergent\"]}"),
        "{mlir}"
    );
    assert!(mlir.contains("func.func private @ext_load(!llvm.ptr<1>, i32) -> f32 attributes"), "{mlir}");
    assert!(mlir.contains("func.call @ext_scale("), "{mlir}");
    assert!(mlir.contains("{no_inline} : (!llvm.ptr<1>, i32) -> f32"), "{mlir}");
    assert!(mlir.contains("llvm.inttoptr"), "{mlir}");
    assert_eq!(k.ir().link_libraries, vec![std::path::PathBuf::from("unused.bc")]);
    // The call without convergence.
    let k = Kernel::build("ext_free", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let f = ExternFn::new("ext_scale").param::<f32>().returns::<f32>().convergent(false);
        let t = thread_idx().x;
        x.store(t, f.call_value::<f32>(&[x.load(t).into()]));
    });
    assert!(k.to_mlir().contains("func.func private @ext_scale(f32) -> f32\n"), "{}", k.to_mlir());
    assert!(k.dump().contains("call @ext_scale("), "{}", k.dump());
}

#[test]
fn shared_pointer_to_generic_parameter_is_cast() {
    let k = Kernel::build("ext_generic", || {
        let s = alloc_smem::<f32>(32, 16);
        let f = ExternFn::new("touch").param_ptr(AddressSpace::Generic).sim_body(|args| {
            let p = args[0].ptr::<f32>();
            p.store(5.0f32);
            None
        });
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        f.call(&[(&s.offset(t)).into()]);
        sync_threads();
        x.store(t, s.offset(t).load());
    });
    let mlir = k.to_mlir();
    assert!(mlir.contains("llvm.addrspacecast") && mlir.contains("!llvm.ptr<3> to !llvm.ptr"), "{mlir}");
    let mut args = vec![KernelArg::f32(vec![0.0; 32])];
    k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].as_f32(), vec![5.0; 32]);
}

#[test]
fn overloads_generics_and_constants_mangle() {
    let k = Kernel::build("ext_overloads", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let n = param_tensor::<i32>("n", Layout::col_major(32));
        let add = ExternOverloads::new()
            .variant(
                ExternFn::new("add")
                    .param::<f32>()
                    .param::<f32>()
                    .returns::<f32>()
                    .sim(|a| vec![(f32::from_bits(a[0] as u32) + f32::from_bits(a[1] as u32)).to_bits() as u64]),
            )
            .variant(
                ExternFn::new("add")
                    .param::<i32>()
                    .param::<i32>()
                    .returns::<i32>()
                    .sim(|a| vec![(a[0] as u32).wrapping_add(a[1] as u32) as u64]),
            );
        let generic = ExternFn::new("twice").param_generic().returns_same_as(0).sim(|a| vec![a[0].wrapping_mul(2)]);
        let scaled = ExternFn::new("scaled").param::<i32>().param_const::<i32>().returns::<i32>().sim(|a| vec![a[0]]);
        let t = thread_idx().x;
        let xf = add.call(&[x.load(t).into(), x.load(t).into()]).unwrap();
        x.store(t, CallArg::Value(xf).value::<f32>());
        let ni = add.call(&[n.load(t).into(), n.load(t).into()]).unwrap();
        let ni = CallArg::Value(ni).value::<i32>();
        let ni = generic.call_value::<i32>(&[ni.into()]);
        let ni = scaled.call_value::<i32>(&[ni.into(), CallArg::Const(-3)]);
        n.store(t, ni);
    });
    let mlir = k.to_mlir();
    for sym in [
        "@add_f32_f32(f32, f32) -> f32",
        "@add_i32_i32(i32, i32) -> i32",
        "@twice_i32(i32) -> i32",
        "@scaled_i32_i32__2D3(i32) -> i32",
    ] {
        assert!(mlir.contains(sym), "{sym} in {mlir}");
    }
    let mut args = vec![KernelArg::f32(vec![1.5; 32]), KernelArg::i32((0..32).collect())];
    k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].as_f32(), vec![3.0; 32]);
    assert_eq!(args[1].as_i32(), (0..32).map(|i| 4 * i).collect::<Vec<i32>>());
}

#[test]
fn device_function_ir_and_mlir() {
    let f = axpy();
    let mlir = f.to_mlir();
    assert!(mlir.contains("gpu.func @axpy(%v0: f32, %v1: f32, %v2: f32) -> f32 {"), "{mlir}");
    assert!(!mlir.contains(" kernel"), "{mlir}");
    assert!(mlir.contains("gpu.return %v"), "{mlir}");
    let g = scale_in_place();
    assert!(g.to_mlir().contains("gpu.func @scale_in_place(%arg_arg0: memref<"), "{}", g.to_mlir());
}

#[test]
fn kernels_call_device_functions_in_the_simulator() {
    let (k, _) = (axpy_caller(), axpy());
    let (x, y) = inputs();
    let mut args = vec![KernelArg::f32(x.clone()), KernelArg::f32(y.clone()), KernelArg::ScalarF32(2.0)];
    k.simulate(LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].as_f32(), axpy_expected(2.0, &x, &y));
    let mlir = k.to_mlir();
    assert!(mlir.contains("func.func private @axpy(f32, f32, f32) -> f32"), "{mlir}");
    assert!(mlir.contains("func.func private @scale_in_place(!llvm.ptr, f32)"), "{mlir}");
    assert!(k.dump().contains("simulated as"), "{}", k.dump());
}

#[test]
fn ptx_inlines_linked_bitcode_or_calls_it() {
    let Some((toolchain, bc)) = toolchain_and_bitcode("extern-calls") else { return };
    for arch in [Arch::Sm80, Arch::Sm90a] {
        // ext_scale inlined (its 3.0 in the kernel); ext_load, never inlined, a linked `.func`.
        let inlined = bitcode_kernel(&bc, true).compile_ptx(&toolchain, arch).unwrap();
        assert!(!inlined.contains("ext_scale"), "{inlined}");
        assert!(inlined.contains("mul.rn.f32") && inlined.contains("0f40400000"), "the inlined 3.0: {inlined}");
        assert!(inlined.contains("), ext_load, (param0, param1)"), "{inlined}");
        let called = bitcode_kernel(&bc, false).compile_ptx(&toolchain, arch).unwrap();
        for f in ["ext_scale", "ext_load"] {
            assert!(called.contains(&format!(".func  (.param .b32 func_retval0) {f}(")), "{called}");
            assert!(called.contains(&format!("(retval0), {f}, (param0")), "{called}");
        }
        assert!(!called.contains(".extern"), "linked, not external: {called}");
    }
}

#[test]
fn device_function_ptx_has_the_c_abi() {
    let Some(toolchain) = Toolchain::discover() else { return };
    for arch in [Arch::Sm80, Arch::Sm90a] {
        let ptx = axpy().compile_ptx(&toolchain, arch).unwrap();
        assert!(ptx.contains(".visible .func  (.param .b32 func_retval0) axpy("), "{ptx}");
        assert!(ptx.contains(".param .b32 axpy_param_2"), "{ptx}");
        let ptx = scale_in_place().compile_ptx(&toolchain, arch).unwrap();
        assert!(
            ptx.contains(".visible .func scale_in_place(") && ptx.contains(".param .b64 scale_in_place_param_0"),
            "{ptx}"
        );
        let caller = axpy_caller().compile_ptx(&toolchain, arch).unwrap();
        assert!(caller.contains(".extern .func  (.param .b32 func_retval0) axpy"), "{caller}");
        assert!(caller.contains(".extern .func scale_in_place"), "{caller}");
        assert!(caller.contains("call.uni (retval0), axpy") || caller.contains("call (retval0), axpy"), "{caller}");
    }
}

#[test]
fn diagnostics() {
    // A wrong argument count names the function and the caller's line.
    let msg = panic_message(|| {
        Kernel::build("ext_count", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let f = ExternFn::new("f").param::<f32>().param::<f32>().returns::<f32>();
            let _ = f.call(&[x.load(0).into()]);
        });
    });
    assert!(msg.contains("`f` takes 2 arguments, but the call passes 1") && msg.contains("extern_calls.rs"), "{msg}");
    // A wrong argument type.
    let msg = panic_message(|| {
        Kernel::build("ext_type", || {
            let n = param_tensor::<i32>("n", Layout::col_major(32));
            let f = ExternFn::new("f").param::<f32>().returns::<f32>();
            let _ = f.call(&[n.load(0).into()]);
        });
    });
    assert!(msg.contains("argument 0 of `f` must be a f32 value") || msg.contains("must be a Float32"), "{msg}");
    // Conflicting prototypes.
    let msg = panic_message(|| {
        Kernel::build("ext_conflict", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let _ = ExternFn::new("g").param::<f32>().returns::<f32>().call(&[x.load(0).into()]);
            let _ = ExternFn::new("g").param::<f32>().call(&[x.load(1).into()]);
        });
    });
    assert!(msg.contains("conflicting prototypes for external function `g`"), "{msg}");
    // One symbol called both inlined and not: the declaration decides, so it is refused.
    let msg = panic_message(|| {
        Kernel::build("ext_inline_mix", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let f = ExternFn::new("k").param::<f32>().returns::<f32>();
            let _ = f.clone().call(&[x.load(0).into()]);
            let _ = f.inline(false).call(&[x.load(1).into()]);
        });
    });
    assert!(msg.contains("`k` is called both inlined and not inlined"), "{msg}");
    // Declarations of one symbol share convergence (any) and a raw simulation (the first given).
    let k = Kernel::build("ext_merge", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let plain = ExternFn::new("twice").param::<f32>().returns::<f32>().convergent(false);
        let a = plain.call_value::<f32>(&[x.load(t).into()]);
        let simulated = ExternFn::new("twice")
            .param::<f32>()
            .returns::<f32>()
            .sim(|a| vec![(f32::from_bits(a[0] as u32) * 2.0).to_bits() as u64]);
        x.store(t, simulated.call_value::<f32>(&[a.into()]));
    });
    assert!(k.to_mlir().contains("func.func private @twice(f32) -> f32 attributes {passthrough = [\"convergent\"]}"));
    let mut args = [KernelArg::f32((0..32).map(|i| i as f32).collect())];
    k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].as_f32(), (0..32).map(|i| 4.0 * i as f32).collect::<Vec<f32>>());
    // A shared pointer to a global parameter.
    let msg = panic_message(|| {
        Kernel::build("ext_space", || {
            let s = alloc_smem::<f32>(32, 16);
            ExternFn::new("h").param_ptr(AddressSpace::Gmem).call(&[(&s.offset(0)).into()]);
        });
    });
    assert!(msg.contains("argument 0 of `h` must be a pointer to Gmem memory"), "{msg}");
    // Simulating a call with no simulation.
    let k = Kernel::build("ext_nosim", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, ExternFn::new("opaque").param::<f32>().returns::<f32>().call_value::<f32>(&[x.load(t).into()]));
    });
    let err = k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut [KernelArg::f32(vec![0.0; 32])]).unwrap_err();
    assert!(err.to_string().contains("extern function `opaque` has no simulator implementation"), "{err}");
    // A device function that allocates shared memory.
    let msg = panic_message(|| {
        DeviceFunction::builder("bad").param::<f32>().returns::<f32>().build(|args| {
            let s = alloc_smem::<f32>(32, 16);
            s.offset(0).store(args[0].value::<f32>());
            Some(s.offset(0).load().into())
        });
    });
    assert!(msg.contains("device function `bad` allocates shared memory"), "{msg}");
    // A device function whose body returns another type.
    let msg = panic_message(|| {
        DeviceFunction::builder("bad_ret")
            .param::<i32>()
            .returns::<f32>()
            .build(|args| Some(args[0].value::<i32>().into()));
    });
    assert!(msg.contains("declared to return"), "{msg}");
}

#[test]
fn missing_bitcode_is_a_compile_error() {
    let Some(toolchain) = Toolchain::discover() else { return };
    let k = bitcode_kernel("/nonexistent/cute-rs/missing.bc".as_ref(), true);
    let err = k.compile_ptx(&toolchain, Arch::Sm80).unwrap_err();
    assert!(err.to_string().contains("bitcode file /nonexistent/cute-rs/missing.bc does not exist"), "{err}");
    // The compile option too.
    let opts = CompileOptions::new().gpu_arch(Arch::Sm80).use_ptxas(false).link_libraries(["/nonexistent/x.bc"]);
    let err = axpy().compile_with(&toolchain, &opts).unwrap_err();
    assert!(err.to_string().contains("CompileOptions::link_libraries names it"), "{err}");
}

#[test]
fn modules_refuse_conflicting_externs() {
    let a = Kernel::build("mk_ext_a", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let _ = ExternFn::new("shared_fn").param::<f32>().returns::<f32>().call(&[x.load(0).into()]);
    });
    let b = Kernel::build("mk_ext_b", || {
        let x = param_tensor::<i32>("x", Layout::col_major(32));
        let _ = ExternFn::new("shared_fn").param::<i32>().returns::<i32>().call(&[x.load(0).into()]);
    });
    let err = KernelModule::new("mk_ext", [a.clone(), b]).unwrap_err();
    assert!(err.message().contains("declare external function `shared_fn` differently"), "{err}");
    // Alike, they share one prototype.
    let c = Kernel::build("mk_ext_c", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let _ = ExternFn::new("shared_fn").param::<f32>().returns::<f32>().call(&[x.load(1).into()]);
    });
    let m = KernelModule::new("mk_ext", [a, c]).unwrap();
    assert_eq!(m.to_mlir().matches("func.func private @shared_fn").count(), 1, "{}", m.to_mlir());
}

#[test]
fn exports_refuse_unresolved_calls_and_device_functions() {
    let Some(toolchain) = Toolchain::discover() else { return };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm80).use_ptxas(false);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("extern-export-{}", std::process::id()));
    let no_object = cute::aot::ExportOptions { compile_object: false, cc: None, cuda_include: None };
    // A kernel calling device functions linked at load time: its PTX alone cannot be loaded.
    let caller = axpy_caller().compile_with(&toolchain, &opts).unwrap();
    let err = caller.export_c_with(&dir, "caller", "", &no_object).unwrap_err();
    assert!(err.to_string().contains("calls external function `axpy`"), "{err}");
    // A device function is linked, not launched.
    let f = axpy().compile_with(&toolchain, &opts).unwrap();
    let err = f.export_c_with(&dir, "axpy", "", &no_object).unwrap_err();
    assert!(err.to_string().contains("has no kernel"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

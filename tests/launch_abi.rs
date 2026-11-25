//! The kernel parameter ABI and launch attributes on the simulator and in the emitted code:
//! by-value structs, the kernel signature and its launch checks, module globals, launch-time
//! shared memory, overlays, preferred and fallback clusters, and launch bounds.
//!
//! `module_globals_persist_across_launches` ports the module-global part of CUTLASS 4.8.1's
//! `examples/python/CuTeDSL/notebooks/2_primitives/01_array_concepts.ipynb`, and
//! `launch_smem_covers_the_allocations` `examples/python/CuTeDSL/dsl_tutorials/dynamic_smem_size.py`,
//! Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! BSD-3-Clause; see THIRD-PARTY-NOTICES.md.

mod abi_kernels;

use abi_kernels::*;
use cute::prelude::*;

fn build_panic(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).expect_err("the build should fail");
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

fn one_block(threads: u32) -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [threads, 1, 1])
}

#[test]
fn by_value_struct_fields_read_back() {
    let k = struct_kernel();
    let mut args = struct_args();
    k.simulate(one_block(32), &mut args).unwrap_or_else(|e| panic!("{e}"));
    check_struct(&args);
    let sig = k.signature();
    assert_eq!(sig.params[0].kind, ParamKind::ByValue { size: 32, align: 8, grid_constant: false });
    assert_eq!(
        sig.to_string(),
        "byval_struct(s: byval[32, align 8], ints: ptr<i64, align 16>, floats: ptr<f64, align 16>)"
    );
    // The struct's C layout and its packing.
    let layout = struct_layout();
    let offsets: Vec<usize> = layout.fields.iter().map(|f| f.offset).collect();
    assert_eq!((offsets, layout.size, layout.align), (vec![0, 4, 8, 16, 24], 32, 8));
    assert!(layout.pack(&[KernelArg::scalar(1i32)]).is_err());
}

#[test]
fn by_value_parameter_misuse() {
    let msg = build_panic(|| {
        Kernel::build("store_byval", || {
            let s = param_bytes("s", 16, 8, false);
            let _ = s.load::<i32>(0);
            // Fields are read at multiples of their size.
            let _ = s.load::<i64>(4);
        });
    });
    assert!(msg.contains("a i64 at byte 4 is not naturally aligned in the parameter's 16 bytes"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bad_size", || {
            let _ = param_bytes("s", 12, 8, false);
        });
    });
    assert!(msg.contains("the size must be a nonzero multiple of the alignment"), "{msg}");
}

#[test]
fn signature_checks_name_the_leaf() {
    let k = Kernel::builder("leaves").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((64,)));
        let n = param_extent("n", 4);
        let wide = param_extent_i64("wide", 1);
        let s = param_bytes("s", 16, 8, false);
        let i = thread_idx().x;
        if_(i.lt(n), || x.store(i, s.load::<f32>(0) + wide.to::<f32>()));
    });
    let good = || {
        vec![
            KernelArg::f32(vec![0.0; 64]),
            KernelArg::ScalarI32(8),
            KernelArg::scalar(3i64),
            KernelArg::by_value(vec![0; 16]),
        ]
    };
    k.simulate(one_block(32), &mut good()).unwrap_or_else(|e| panic!("{e}"));
    let fail = |i: usize, arg: KernelArg| {
        let mut args = good();
        args[i] = arg;
        k.simulate(one_block(32), &mut args).unwrap_err().message
    };
    let e = fail(0, KernelArg::i32(vec![0; 64]));
    assert!(e.contains("argument 0 (`x`) must be a f32 buffer, got an i32 buffer of 64"), "{e}");
    let e = fail(1, KernelArg::ScalarI32(6));
    assert!(e.contains("argument 1 (`n`) must be a multiple of 4, got 6"), "{e}");
    let e = fail(2, KernelArg::ScalarI32(3));
    assert!(e.contains("argument 2 (`wide`) is a 64-bit leaf (i64), got a i32 scalar"), "{e}");
    let e = fail(3, KernelArg::by_value(vec![0; 12]));
    assert!(e.contains("argument 3 (`s`) is passed by value as 16 bytes, got 12"), "{e}");
    let e = fail(3, KernelArg::ScalarI32(1));
    assert!(e.contains("argument 3 (`s`) is passed by value (16 bytes"), "{e}");

    // The launcher's packing checks pointers too: alignment and the bytes behind them.
    let sig = k.signature();
    let (n, w, b) = (KernelArg::ScalarI32(8), KernelArg::scalar(3i64), [0u8; 16]);
    let values = |address: u64, bytes: usize| {
        vec![
            LaunchValue::Pointer { address, bytes: Some(bytes) },
            LaunchValue::Scalar(&n),
            LaunchValue::Scalar(&w),
            LaunchValue::Bytes(&b),
        ]
    };
    let packed = sig.pack(&values(0x1000, 256)).unwrap();
    assert_eq!(
        (packed.len(), packed.bytes(0), packed.bytes(1)),
        (4, 0x1000u64.to_le_bytes().to_vec(), vec![8, 0, 0, 0])
    );
    assert_eq!(packed.bytes(2), 3i64.to_le_bytes());
    let e = sig.pack(&values(0x1004, 256)).unwrap_err();
    assert!(e.contains("argument 0: `x` is at 0x1004, which is not 16-byte aligned"), "{e}");
    let e = sig.pack(&values(0x1000, 128)).unwrap_err();
    assert!(e.contains("argument 0 (`x`) needs 64 f32 elements (256 bytes), the pointer reaches 128"), "{e}");
}

#[test]
fn module_globals_persist_across_launches() {
    let k = globals_kernel();
    // A fresh module per Kernel::simulate: the counter starts at 0 each time.
    let mut args = globals_args();
    k.simulate(one_block(4), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].as_f32(), [2.0, 6.0, 12.0, 20.0]);
    assert_eq!(args[2].as_i32(), [1]);
    // One module across launches: the counter climbs, and a host write is seen next launch.
    let mut module = k.sim_module();
    for want in 1..=3 {
        let mut args = globals_args();
        module.simulate(one_block(4), &mut args).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(args[2].as_i32(), [want]);
    }
    assert_eq!(module.read_global("run_count").unwrap().to_vec::<i32>(), [3]);
    module.write_global("run_count", &KernelArg::i32(vec![41])).unwrap();
    module.write_global("gamma", &KernelArg::f32(vec![2.0; 4])).unwrap();
    let mut args = globals_args();
    module.simulate(one_block(4), &mut args).unwrap();
    assert_eq!((args[1].as_f32(), args[2].as_i32()), ([4.0, 8.0, 12.0, 16.0].as_slice(), [42].as_slice()));
    let e = module.write_global("run_count", &KernelArg::f32(vec![1.0])).unwrap_err();
    assert!(e.contains("holds 1 i32 elements, got an f32 buffer of 1"), "{e}");
    assert!(module.write_global("nothing", &KernelArg::i32(vec![1])).unwrap_err().contains("no module global"));

    // A read-only global is folded into the kernel: host writes are refused.
    let k = readonly_kernel();
    let mut args = vec![KernelArg::i32(vec![0; 4])];
    k.simulate(one_block(4), &mut args).unwrap();
    assert_eq!(args[0].as_i32(), [10, 21, 32, 43]);
    let e = k.sim_module().write_global("offset", &KernelArg::i32(vec![0; 4])).unwrap_err();
    assert!(e.contains("module global `offset` is read-only"), "{e}");
    assert!(k.signature().global("offset").is_some_and(|g| g.readonly && !g.constant));
}

#[test]
fn module_global_misuse_is_diagnosed() {
    let msg = build_panic(|| {
        Kernel::build("store_constant", || {
            let c = module_global::<f32>(Some("table"), 4, AddressSpace::Cmem, None, false);
            c.store(0, Float32::from(1.0));
        });
    });
    assert!(msg.contains("a store to `table`, a global in constant memory, which the device cannot write"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("store_readonly", || {
            let c = module_global::<i32>(Some("fixed"), 1, AddressSpace::Gmem, Some(&[1]), true);
            c.store(0, Int32::from(2));
        });
    });
    assert!(msg.contains("a store to `fixed`, a read-only module global"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("too_much_constant", || {
            let a = module_global::<f32>(Some("a"), 10_000, AddressSpace::Cmem, None, false);
            let b = module_global::<f32>(Some("b"), 8_000, AddressSpace::Cmem, None, false);
            let out = param_tensor::<f32>("out", Layout::col_major((1,)));
            out.store(0, a.load(0) + b.load(0));
        });
    });
    assert!(msg.contains("constant-memory globals hold 72000 bytes; a module has at most 65536"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("readonly_no_init", || {
            let _ = module_global::<f32>(Some("x"), 1, AddressSpace::Gmem, None, true);
        });
    });
    assert!(msg.contains("a readonly module global needs init"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("duplicate", || {
            let _ = module_global::<f32>(Some("x"), 1, AddressSpace::Gmem, None, false);
            let _ = param_tensor::<f32>("x", Layout::col_major((1,)));
        });
    });
    assert!(msg.contains("duplicate parameter name \"x\""), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("clash", || {
            let _ = module_global::<f32>(Some("clash"), 1, AddressSpace::Gmem, None, false);
        });
    });
    assert!(msg.contains("module global `clash` has the name of another symbol"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("extern", || {
            let _ = extern_global::<f32>("elsewhere", 4, AddressSpace::Gmem);
        });
    });
    assert!(msg.contains("extern globals are defined in another module and need device code linking"), "{msg}");
    // Generated names are C identifiers.
    let k = Kernel::build("anonymous", || {
        let g = module_global::<i32>(None, 2, AddressSpace::Gmem, None, false);
        g.store(0, Int32::from(1));
    });
    assert_eq!(k.signature().globals[0].name, "__cute_global_0");
}

#[test]
fn launch_smem_covers_the_allocations() {
    // The tutorial's kernel_static: its allocations need 2368 bytes, which CuTe DSL infers for
    // smem=None. They are static shared memory here, so the launch adds none.
    let k = static_smem_kernel();
    let need = k.ir().shared_memory_bytes() as u32;
    assert_eq!(need, 2368);
    let mut out = vec![KernelArg::i32(vec![-1])];
    k.simulate(one_block(1), &mut out).unwrap();
    assert_eq!(out[0].as_i32(), [0]);
    // smem=16 is below the allocations: CuTe DSL warns, cute-rs refuses.
    let e = k.simulate(one_block(1).with_smem(16), &mut out).unwrap_err().message;
    assert!(
        e.contains(
            "kernel `kernel_static` allocates 2368 bytes of shared memory per block, more than the launch's smem of 16"
        ),
        "{e}"
    );
    // More than the allocations: the rest is dynamic shared memory.
    k.simulate(one_block(1).with_smem(need + 1024), &mut out).unwrap();
    assert_eq!(out[0].as_i32(), [1024]);
    let e = k.simulate(one_block(1).with_smem(need).with_dynamic_smem(4), &mut out).unwrap_err().message;
    assert!(e.contains("sets both smem_bytes (2368) and dynamic_smem_bytes (4): set one"), "{e}");

    // With a dynamic region, the bytes past the build-time buffers are the region's.
    let k = region_kernel();
    assert_eq!(k.ir().shared_memory_layout().dynamic_start, REGION_START as usize);
    for total in [REGION_START, REGION_START + 64, REGION_START + 256] {
        let mut out = vec![KernelArg::i32(vec![-1; 65])];
        k.simulate(one_block(64).with_smem(total), &mut out).unwrap_or_else(|e| panic!("{total}: {e}"));
        let words = ((total - REGION_START) / 4) as usize;
        let mut want = vec![(total - REGION_START) as i32];
        want.extend((0..64).map(|i| if i < words { i as i32 * 5 } else { -1 }));
        assert_eq!(out[0].as_i32(), want, "smem {total}");
    }
    let e = k.simulate(one_block(64).with_smem(100), &mut [KernelArg::i32(vec![0; 65])]).unwrap_err().message;
    assert!(e.contains("allocates 160 bytes of shared memory per block, more than the launch's smem of 100"), "{e}");
    let e = k.simulate(one_block(64).with_smem(400 * 1024), &mut [KernelArg::i32(vec![0; 65])]).unwrap_err().message;
    assert!(
        e.contains("needs 409600 bytes of shared memory per block")
            && e.contains("no GPU gives a block more than 334848"),
        "{e}"
    );
}

#[test]
fn overlays_share_bytes_under_the_race_checker() {
    let k = overlay_kernel(false);
    let layout = k.ir().shared_memory_layout();
    let shared: Vec<usize> = layout.starts.iter().flatten().copied().collect();
    assert_eq!(shared, [0, 0], "both alternatives start at the group's offset");
    assert!(layout.in_window && layout.static_bytes == 256);
    let mut args = vec![KernelArg::f32(vec![0.0; 64]), KernelArg::i32(vec![0; 32])];
    k.simulate_all_orders(one_block(64), &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].as_f32(), (0..64).rev().map(|i| i as f32).collect::<Vec<_>>());
    assert_eq!(args[1].as_i32(), (0..32).rev().map(|i| i * 3).collect::<Vec<_>>());
    // Without the barrier between the phases, the second alternative's writes race with the
    // first's reads of the same bytes.
    let k = overlay_kernel(true);
    let mut args = vec![KernelArg::f32(vec![0.0; 64]), KernelArg::i32(vec![0; 32])];
    let e = k.simulate(one_block(64), &mut args).unwrap_err();
    assert!(e.message.contains("race"), "{e}");
    let msg = build_panic(|| {
        Kernel::build("overlay_two", || {
            let _ = smem_overlay(
                || (make_smem_tensor::<f32>(Layout::col_major((4,))), make_smem_tensor::<f32>(Layout::col_major((4,)))),
                || make_smem_tensor::<i32>(Layout::col_major((4,))),
            );
        });
    });
    assert!(msg.contains("each alternative of smem_overlay allocates exactly one shared buffer"), "{msg}");
}

/// Cluster coordinates per block: (cluster index, rank, size).
fn coords(out: &KernelArg) -> Vec<(i32, i32, i32)> {
    out.as_i32().chunks(3).map(|c| (c[0], c[1], c[2])).collect()
}

#[test]
fn preferred_and_fallback_clusters() {
    let k = cluster_kernel();
    let cfg = LaunchConfig::new([8, 1, 1], [32, 1, 1]).with_fallback_cluster([4, 1, 1], [2, 1, 1]);
    let run = |shapes: ClusterShapes| {
        let mut out = vec![KernelArg::i32(vec![-1; 24])];
        let opts = SimOptions { cluster_shapes: shapes, ..SimOptions::default() };
        k.simulate_with(cfg, &mut out, &opts).unwrap_or_else(|e| panic!("{shapes:?}: {e}"));
        coords(&out[0])
    };
    let preferred = run(ClusterShapes::Preferred);
    assert_eq!(preferred, (0..8).map(|b| (b / 4, b % 4, 4)).collect::<Vec<_>>());
    let fallback = run(ClusterShapes::Fallback);
    assert_eq!(fallback, (0..8).map(|b| (b / 2, b % 2, 2)).collect::<Vec<_>>());
    // A mix: each preferred-sized tile is one cluster of 4 or two of 2.
    let mut both = false;
    for seed in 0..8 {
        let mixed = run(ClusterShapes::Mixed { seed });
        for tile in mixed.chunks(4) {
            let size = tile[0].2;
            assert!(size == 4 || size == 2, "{mixed:?}");
            for (r, c) in tile.iter().enumerate() {
                assert_eq!((c.1, c.2), ((r as i32) % size, size), "seed {seed}: {mixed:?}");
            }
        }
        both |= mixed.iter().any(|c| c.2 == 4) && mixed.iter().any(|c| c.2 == 2);
    }
    assert!(both, "some seed mixes the shapes");

    let fails = |cfg: LaunchConfig| k.simulate(cfg, &mut [KernelArg::i32(vec![0; 24])]).unwrap_err().message;
    let base = LaunchConfig::new([8, 1, 1], [32, 1, 1]);
    let e = fails(LaunchConfig { fallback_cluster: Some([2, 1, 1]), ..base });
    assert!(e.contains("a fallback cluster ([2, 1, 1]) needs a preferred cluster"), "{e}");
    let e = fails(base.with_fallback_cluster([4, 1, 1], [3, 1, 1]));
    assert!(
        e.contains("the preferred cluster [4, 1, 1] is not a whole number of fallback clusters of [3, 1, 1]"),
        "{e}"
    );
    let e = fails(LaunchConfig::new([6, 1, 1], [32, 1, 1]).with_fallback_cluster([4, 1, 1], [2, 1, 1]));
    assert!(e.contains("grid [6, 1, 1] is not a whole number of clusters of [4, 1, 1]"), "{e}");
}

#[test]
fn launch_bounds_in_the_ir_and_launches() {
    let reqntid = bounds_kernel(None, None).to_mlir();
    assert!(reqntid.contains("attributes {nvvm.reqntid = array<i32: 128, 1, 1>}"), "{reqntid}");
    assert!(!reqntid.contains("known_block_size") && !reqntid.contains("maxntid"));
    let maxntid = bounds_kernel(Some([256, 1, 1]), Some(2)).to_mlir();
    assert!(
        maxntid.contains("attributes {nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i32}"),
        "{maxntid}"
    );
    assert!(!maxntid.contains("reqntid"));

    let k = bounds_kernel(Some([128, 1, 1]), None);
    let mut out = vec![KernelArg::i32(vec![0; 128])];
    k.simulate(one_block(96), &mut out).unwrap();
    let e = k.simulate(one_block(256), &mut out).unwrap_err().message;
    assert!(
        e.contains("kernel `bounds` takes blocks of at most [128, 1, 1] (max_threads), launched with [256, 1, 1]"),
        "{e}"
    );
    let msg = build_panic(|| {
        Kernel::builder("both").block_dim([256, 1, 1]).max_threads([128, 1, 1]).build(|| {});
    });
    assert!(msg.contains("declares blocks of [256, 1, 1], larger than its max_threads [128, 1, 1]"), "{msg}");

    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping the PTX checks: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = |k: Kernel| k.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    let a = ptx(bounds_kernel(None, Some(4)));
    assert!(a.contains(".reqntid 128, 1, 1") && a.contains(".minnctapersm 4") && !a.contains(".maxntid"), "{a}");
    let b = ptx(bounds_kernel(Some([256, 1, 1]), None));
    assert!(b.contains(".maxntid 256, 1, 1") && !b.contains(".reqntid"), "{b}");
}

#[test]
fn abi_kernels_compile() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = struct_kernel().compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains(".param .align 8 .b8 byval_struct_param_0[32]"), "{ptx}");
    let ptx = grid_constant_kernel().compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains(".param .align 64 .b8 grid_constant_desc_param_0[128]"), "{ptx}");
    let mlir = globals_kernel().to_mlir();
    assert!(
        mlir.contains("memref.global @gamma : memref<4xf32, 4> = dense<\"0x0000803F0000C03F0000004000002040\">"),
        "{mlir}"
    );
    assert!(mlir.contains("memref.global @run_count : memref<1xi32, 1> = dense<\"0x00000000\">"), "{mlir}");
    let ptx = globals_kernel().compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        ptx.contains(".visible .const .align 4 .b8 gamma[16]")
            || ptx.contains(".visible .const .align 4 .f32 gamma[4]"),
        "{ptx}"
    );
    assert!(ptx.contains(".visible .global .align 4 .b32 run_count[1]") || ptx.contains("run_count"), "{ptx}");
    assert!(ptx.contains("ld.const"), "{ptx}");
    let ptx = readonly_kernel().compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains("offset"), "{ptx}");
    for k in [overlay_kernel(false), region_kernel(), static_smem_kernel()] {
        k.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    }
    cluster_kernel().compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
}

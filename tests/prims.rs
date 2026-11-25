//! The typed MMA descriptors built in a
//! kernel from a runtime shared-memory address, and the base DSL's byte-address swizzle
//! (`cutlass.Swizzle`, `apply_swizzle`, `load_swizzled`, `store_swizzled`) on the
//! simulator, in the PTX, and (with `--features cuda`) on a GPU. The bit-level field
//! tests are the unit tests in `src/prims/descriptors.rs`.

use cute::prelude::*;
use cute::prims::*;

fn panic_text(f: impl FnOnce()) -> String {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

/// Compile `kernel` for `arch` and assemble it with `ptxas` when one is on PATH; the PTX,
/// or `None` without an `mlir-opt`.
fn ptx_for(kernel: &Kernel, arch: Arch) -> Option<String> {
    let tc = Toolchain::discover()?;
    let name = kernel.ir().name.clone();
    let ptx = kernel
        .compile_with(&tc, &CompileOptions::new().gpu_arch(arch))
        .unwrap_or_else(|e| panic!("{name} for {arch:?}: {e}"))
        .ptx()
        .to_string();
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    if let Some(ptxas) = ptxas {
        let base = std::env::temp_dir().join(format!("cute-prims-{}-{name}-{arch:?}", std::process::id()));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, &ptx).unwrap();
        let sm = format!("-arch={}", arch.chip());
        let out = std::process::Command::new(ptxas).arg(&sm).arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas {sm}: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{name}: ptxas {sm} ok");
    }
    Some(ptx)
}

// ---------------------------------------------------------------------------------------
// Descriptors from a runtime shared-memory address.
// ---------------------------------------------------------------------------------------

fn tc_fields() -> Tcgen05SmemDescFields {
    Tcgen05SmemDescFields {
        leading_byte_offset: 16,
        stride_byte_offset: 1024,
        swizzle: Tcgen05SmemSwizzle::Swizzle128B,
        ..Default::default()
    }
}

fn descriptor_kernel(arch: Arch) -> Kernel {
    Kernel::builder("prims_descriptors").block_dim([32, 1, 1]).target(arch).build(|| {
        let out = param_tensor::<u64>("out", Layout::col_major((6,)));
        let out32 = param_tensor::<i32>("out32", Layout::col_major((2,)));
        let _pad = alloc_smem::<u8>(1024, 1024);
        let smem = alloc_smem::<F16>(4096, 1024);
        let t = thread_idx().x;
        if_(t.eq(0), || {
            let w = WgmmaSmemDesc::build(&smem, 16, 1024, 0, WgmmaSwizzle::Swizzle128B).unwrap();
            out.store(0, w.value());
            out.store(1, w.advance_start_address(64u64).unwrap().value());
            let d = Tcgen05SmemDesc::build(&smem, &tc_fields()).unwrap();
            out.store(2, d.value());
            // A runtime increment: (t + 3) * 16 bytes, and a getter of a runtime descriptor.
            let inc = (t.to::<u64>() + 3u64) * 16u64;
            out.store(3, d.advance_start_address(inc).unwrap().value());
            out.store(4, d.start_address().value());
            let i = Tcgen05InstrDesc::for_mma(
                cute::ir::UmmaDataKind::F16,
                DType::F16,
                DType::F16,
                DType::F32,
                (128, 256),
                cpasync::CtaGroup::ONE,
                OperandMajorMode::K,
                OperandMajorMode::K,
            )
            .unwrap();
            out.store(5, i.value().to::<u64>());
            let tm = TmemAddr::from_row_col(t + 32, 0x40).unwrap();
            out32.store(0, tm.value());
            out32.store(1, tm.row_id().value());
        });
    })
}

#[test]
fn descriptors_from_a_runtime_smem_address() {
    let kernel = descriptor_kernel(Arch::Sm100a);
    let mut args = [KernelArg::buffer::<u64>(&[0; 6]), KernelArg::buffer::<i32>(&[0; 2])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got = args[0].to_vec::<u64>();
    let start = kernel.ir().shared_memory_layout().starts.iter().flatten().copied().max().unwrap() as u32;
    assert_eq!(start % 1024, 0);
    let w = WgmmaSmemDesc::build(start, 16, 1024, 0, WgmmaSwizzle::Swizzle128B).unwrap();
    assert_eq!(got[0], w.bits().as_static().unwrap());
    assert_eq!(got[1], w.bits().as_static().unwrap() + 4);
    let d = Tcgen05SmemDesc::build(start, &tc_fields()).unwrap();
    assert_eq!(got[2], d.bits().as_static().unwrap());
    assert_eq!(got[3], d.bits().as_static().unwrap() + 3);
    assert_eq!(got[4], u64::from(start >> 4));
    assert_eq!(got[5], 1 << 4 | 32 << 17 | 8 << 24);
    assert_eq!(args[1].to_vec::<i32>(), vec![32 << 16 | 0x40, 32]);

    // The same kernel through the PTX backend for both descriptor families' targets.
    for arch in [Arch::Sm90a, Arch::Sm100a] {
        if ptx_for(&descriptor_kernel(arch), arch).is_none() {
            eprintln!("skipping the PTX: no mlir-opt found (set CUTE_MLIR_OPT)");
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------
// The byte-address swizzle.
// ---------------------------------------------------------------------------------------

#[test]
fn byte_swizzle_presets_and_conversions() {
    assert_eq!(ByteSwizzle::from_name("none"), Ok(ByteSwizzle::new(0, 0, 0)));
    assert_eq!(ByteSwizzle::from_name("s32b"), Ok(ByteSwizzle::new(1, 4, 3)));
    assert_eq!(ByteSwizzle::from_name("s64b"), Ok(ByteSwizzle::new(2, 4, 3)));
    assert_eq!(ByteSwizzle::from_name("s128b"), Ok(ByteSwizzle::new(3, 4, 3)));
    let e = ByteSwizzle::from_name("s256b").unwrap_err();
    assert!(e.contains("s256b") && e.contains("s128b"), "{e}");

    let presets = [ByteSwizzle::NONE, ByteSwizzle::S32B, ByteSwizzle::S64B, ByteSwizzle::S128B];
    let maps = [TensorMapSwizzle::none, TensorMapSwizzle::s32b, TensorMapSwizzle::s64b, TensorMapSwizzle::s128b];
    let umma = [
        Tcgen05SmemSwizzle::SwizzleNone,
        Tcgen05SmemSwizzle::Swizzle32B,
        Tcgen05SmemSwizzle::Swizzle64B,
        Tcgen05SmemSwizzle::Swizzle128B,
    ];
    for i in 0..4 {
        assert_eq!(presets[i].to_tensor_map_swizzle(), Ok(maps[i]));
        assert_eq!(TensorMapSwizzle::try_from(presets[i]), Ok(maps[i]));
        assert_eq!(presets[i].to_tcgen05_smem_swizzle(), Ok(umma[i]));
        assert_eq!(Tcgen05SmemSwizzle::try_from(presets[i]), Ok(umma[i]));
    }
    for odd in [ByteSwizzle::new(2, 5, 2), ByteSwizzle::new(3, 4, 4), ByteSwizzle::new(0, 4, 3)] {
        assert!(odd.to_tensor_map_swizzle().unwrap_err().contains("tensor-map"));
        assert!(odd.to_tcgen05_smem_swizzle().unwrap_err().contains("tcgen05"));
    }

    // A permutation of every period-aligned block of byte addresses.
    for sw in presets.into_iter().chain([ByteSwizzle::new(2, 5, 2), ByteSwizzle::new(1, 3, 5)]) {
        let p = sw.period() as u64;
        for block in [0, p, 7 * p] {
            let mut seen = vec![false; p as usize];
            for a in block..block + p {
                let s = sw.apply(a);
                assert!((block..block + p).contains(&s), "{sw}: {a} -> {s}");
                assert!(!std::mem::replace(&mut seen[(s - block) as usize], true), "{sw}: {s} twice");
            }
        }
        // The formula, written out.
        for a in 0..4096u64 {
            let yyy = (a >> (sw.mbase + sw.sshift)) & ((1 << sw.bbits) - 1);
            assert_eq!(sw.apply(a), a ^ (yyy << sw.mbase));
        }
    }
    assert_eq!(ByteSwizzle::S128B.period(), 1024);
    assert_eq!(ByteSwizzle::NONE.period(), 1);

    // The element-unit swizzle of 1-, 2- and 4-byte elements moves element i where the
    // byte swizzle moves its bytes; s128b is S<3,3,3> on Float16, and cute-rs's identity
    // S<0,4,3> is NONE.
    for sw in presets {
        for bytes in [1usize, 2, 4] {
            let e = sw.to_elements(8 * bytes).unwrap();
            assert_eq!(ByteSwizzle::from_elements(e, 8 * bytes), Ok(sw));
            for i in 0..2048usize {
                assert_eq!(e.apply(i) * bytes, sw.apply((i * bytes) as u64) as usize, "{sw} on {bytes}-byte elements");
            }
        }
    }
    assert_eq!(ByteSwizzle::S128B.to_elements(16), Ok(Swizzle::new(3, 3, 3)));
    assert_eq!(ByteSwizzle::from_elements(Swizzle::identity(), 16), Ok(ByteSwizzle::NONE));
    assert_eq!(Swizzle::from(ByteSwizzle::NONE), Swizzle::identity());
    assert!(ByteSwizzle::new(3, 2, 3).to_elements(64).is_err());
}

const N: usize = 1024;

/// Each of `N / c` threads stores its `c` elements of `src` (`c` = `count`, 1 for a
/// scalar) into an s128b-swizzled shared buffer with `store_swizzled`, the buffer is
/// copied out as it lies in memory (`raw`), and each thread reads its elements back with
/// `load_swizzled` into `dst`. Without `copy_raw` the `raw` copy is left out (with one
/// thread, LLVM would otherwise serve the swizzled loads from the copy's loads of the same
/// constant addresses, splitting the vectors).
fn swizzle_kernel(count: Option<usize>, copy_raw: bool) -> Kernel {
    let c = count.unwrap_or(1);
    let threads = N / c;
    let name = match count {
        None => "swizzled_scalar".to_string(),
        Some(c) => format!("swizzled_x{c}"),
    };
    Kernel::builder(name).block_dim([threads as u32, 1, 1]).build(move || {
        let src = param_tensor::<F16>("src", Layout::col_major((N,)));
        let raw = param_tensor::<F16>("raw", Layout::col_major((N,)));
        let dst = param_tensor::<F16>("dst", Layout::col_major((N,)));
        let smem = alloc_smem::<F16>(N, 1024);
        let plain = make_tensor(&smem, Layout::col_major((N,)));
        let sw = ByteSwizzle::S128B;
        let t = thread_idx().x.as_index();
        let base = t.clone() * c;
        let p = smem.offset(base.clone());
        match count {
            None => store_swizzled(&p, src.load(base.clone()), sw, None),
            Some(_) => {
                let v = Vector::new(c, (0..c).map(|j| src.load(base.clone() + j)).collect::<Vec<_>>());
                store_swizzled(&p, v, sw, None);
            }
        }
        sync_threads();
        for j in (0..c).filter(|_| copy_raw) {
            raw.store(base.clone() + j, plain.load(base.clone() + j));
        }
        match load_swizzled(&p, sw, None, count) {
            ScalarOrVector::Scalar(v) => dst.store(base.clone(), v),
            ScalarOrVector::Vector(v) => {
                assert_eq!(v.size(), c);
                for (j, x) in v.values().iter().enumerate() {
                    dst.store(base.clone() + j, *x);
                }
            }
        }
    })
}

fn swizzle_inputs() -> Vec<KernelArg> {
    let src: Vec<F16> = (0..N).map(|i| F16::from_f32(i as f32 - 300.0)).collect();
    vec![
        KernelArg::buffer(&src),
        KernelArg::buffer(&vec![F16::from_f32(0.0); N]),
        KernelArg::buffer(&vec![F16::from_f32(0.0); N]),
    ]
}

fn check_swizzled(name: &str, args: &[KernelArg]) {
    let src = args[0].to_vec::<F16>();
    let raw = args[1].to_vec::<F16>();
    let dst = args[2].to_vec::<F16>();
    assert_eq!(dst, src, "{name}: load_swizzled did not return what store_swizzled wrote");
    for (i, &v) in src.iter().enumerate() {
        let phys = ByteSwizzle::S128B.apply(2 * i as u64) as usize / 2;
        assert_eq!(raw[phys], v, "{name}: element {i} is not at {phys}");
    }
}

const COUNTS: [Option<usize>; 5] = [None, Some(1), Some(8), Some(512), Some(1024)];

#[test]
fn swizzled_loads_and_stores_on_the_simulator() {
    for count in COUNTS {
        let kernel = swizzle_kernel(count, true);
        let mut args = swizzle_inputs();
        let threads = (N / count.unwrap_or(1)) as u32;
        kernel.simulate(LaunchConfig::new([1, 1, 1], [threads, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
        check_swizzled(&kernel.ir().name, &args);
    }
}

/// Whole 16-byte granules move as 4-word vectors; a scalar or a count of 1 does not.
#[test]
fn swizzled_vectors_in_the_ptx() {
    for count in COUNTS {
        let kernel = swizzle_kernel(count, false);
        let Some(ptx) = ptx_for(&kernel, Arch::Sm89) else {
            eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
            return;
        };
        if let Some(dir) = std::env::var_os("CUTE_DUMP_PTX") {
            std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", kernel.ir().name)), &ptx).unwrap();
        }
        let wide = count.is_some_and(|c| c >= 8);
        for op in ["ld.shared.v4", "st.shared.v4"] {
            assert_eq!(ptx.contains(op), wide, "{}: `{op}` in the PTX", kernel.ir().name);
        }
    }
}

#[test]
fn swizzle_alignment_diagnostics() {
    // A buffer aligned below the swizzle's period.
    let text = panic_text(|| {
        drop(Kernel::builder("under_aligned").build(|| {
            let smem = alloc_smem::<F16>(1024, 128);
            let _ = load_swizzled(&smem, ByteSwizzle::S128B, None, Some(8));
        }))
    });
    assert!(text.contains("1024") && text.contains("128-byte aligned"), "{text}");
    // Global memory.
    let text = panic_text(|| {
        drop(Kernel::builder("global").build(|| {
            let p = param_ptr::<F16>("p");
            let _ = apply_swizzle(&p, ByteSwizzle::S64B);
        }))
    });
    assert!(text.contains("shared-memory"), "{text}");
    // apply_swizzle without a swizzle is the pointer itself; with one, the physical address.
    Kernel::builder("apply").build(|| {
        let smem = alloc_smem::<F16>(1024, 1024);
        let p = smem.offset(72);
        let q = apply_swizzle(&p, ByteSwizzle::S128B);
        assert_eq!(q.swizzle(), None);
        assert_eq!(apply_swizzle(&p, ByteSwizzle::NONE).swizzle(), None);
    });
}

// ---------------------------------------------------------------------------------------
// The GPU.
// ---------------------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn swizzled_loads_and_stores_on_the_gpu() {
    use cute::cuda::CudaContext;
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let Some(tc) = Toolchain::discover() else { return eprintln!("skipping: no mlir-opt found") };
    let arch = ctx.arch().expect("SM80 or newer");
    for count in COUNTS {
        let kernel = swizzle_kernel(count, true);
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
        let function = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
        let mut args = swizzle_inputs();
        let threads = (N / count.unwrap_or(1)) as u32;
        ctx.launch(&kernel, &function, LaunchConfig::new([1, 1, 1], [threads, 1, 1]), &mut args)
            .unwrap_or_else(|e| panic!("{e}"));
        check_swizzled(&kernel.ir().name, &args);
        eprintln!("{}: GPU ok", kernel.ir().name);
    }
}

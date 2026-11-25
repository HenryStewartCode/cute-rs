//! Raw-address pointers: what `make_ptr`, `toint`, `tospace`, `align` and
//! masking build, the PTX they compile to, and the mistakes the builder and
//! the simulator catch.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

fn kernel(name: &str, body: impl FnOnce()) -> Kernel {
    Kernel::builder(name).block_dim([32, 1, 1]).build(body)
}

fn sim_err(k: &Kernel, args: &mut [KernelArg]) -> SimError {
    match k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), args) {
        Ok(_) => panic!("expected `{}` to fail in the simulator", k.name()),
        Err(e) => e,
    }
}

/// The address spaces, alignments and integer types of what `make_ptr`,
/// `toint`, `tospace`, `align` and `&` give.
#[test]
fn pointer_properties() {
    let _ = kernel("properties", || {
        let x = param_ptr::<f32>("x");
        assert!(!x.is_raw());
        assert_eq!(x.alignment(), 16);
        assert_eq!((x.clone() + 1).alignment(), 4);
        assert_eq!((x.clone() + 4).alignment(), 16);
        let addr = x.toint();
        let raw = make_ptr::<f32>(addr, AddressSpace::Gmem, Some(64));
        assert!(raw.is_raw());
        assert_eq!((raw.memspace(), raw.alignment()), (AddressSpace::Gmem, 64));
        assert_eq!((raw.clone() + 8).alignment(), 32);
        // The default is the element's size, as inttoptr's is.
        let p = inttoptr::<f64>(x.toint_i64(), AddressSpace::Generic);
        assert_eq!((p.memspace(), p.alignment()), (AddressSpace::Generic, 8));
        // A shared pointer's address is an Int32.
        let s = alloc_smem::<u16>(64, 128);
        let _: Int32 = (s.clone() + 3).toint_i32();
        let back = make_ptr::<u16>((s.clone() + 8).toint_i32(), AddressSpace::Smem, Some(16));
        assert_eq!(back.memspace(), AddressSpace::Smem);
        // align: a static offset in an aligned buffer stays in the buffer.
        let up = (s.clone() + 3).align(16);
        assert!(!up.is_raw());
        assert_eq!(up.alignment(), 16);
        // tospace keeps the alignment.
        let g = raw.tospace(AddressSpace::Generic);
        assert_eq!((g.memspace(), g.alignment()), (AddressSpace::Generic, 64));
        assert_eq!(g.tospace(AddressSpace::Generic).memspace(), AddressSpace::Generic);
        // A mask clearing the low 7 bits proves 128-byte alignment.
        let m = &(x.clone() + thread_idx().x) & !127i64;
        assert_eq!(m.alignment(), 128);
        // Tensor memory takes an Int32 address.
        let t = make_ptr::<f32>(Int32::from(0), AddressSpace::Tmem, Some(16));
        assert_eq!(t.memspace(), AddressSpace::Tmem);
        // parameter alignment.
        assert_eq!(param_ptr_aligned::<f32>("y", 4).alignment(), 4);
    });
}

/// The MLIR and PTX: `llvm.inttoptr` and `llvm.load` through a raw pointer,
/// the assumed alignment as an `llvm.intr.assume`, generic accesses
/// without a state space, `addrspacecast` for `tospace`, and a 16-byte
/// vector through `assumed_align=16`.
#[test]
fn raw_pointer_lowering() {
    let k = kernel("lowering", || {
        let x = param_ptr::<f32>("x");
        let y = param_ptr::<f32>("y");
        let t = thread_idx().x;
        let p = make_ptr::<f32>((x + t).toint_i64(), AddressSpace::Generic, None);
        let q = make_ptr::<f32>((y + t).toint_i64(), AddressSpace::Gmem, None).tospace(AddressSpace::Generic);
        q.store(p.load() + 1.0);
    });
    let mlir = k.to_mlir();
    assert!(mlir.contains("llvm.inttoptr"), "{mlir}");
    assert!(mlir.contains("llvm.load") && mlir.contains("-> f32"), "{mlir}");
    assert!(mlir.contains("llvm.addrspacecast"), "{mlir}");
    assert!(mlir.contains("llvm.intr.assume"), "{mlir}");
    if let Some(tc) = Toolchain::discover() {
        let ptx = k.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
        // The generic load has no state space; LLVM sees that the generic
        // store's address came from a global one and stores to global.
        assert!(ptx.contains("ld.f32") || ptx.contains("ld.b32"), "{ptx}");
        assert!(ptx.contains("st.global.f32") || ptx.contains("st.global.b32"), "{ptx}");
    }
}

#[test]
fn raw_vector_copy_is_one_vector_each_way() {
    use cute::ir::Op;
    let case = feature_kernels_raw_vector_copy();
    let wide = |load: bool| {
        case.ir().count_ops(|op| match op {
            Op::Load { results, .. } if load => results.len() == 4,
            Op::Store { values, .. } if !load => values.len() == 4,
            _ => false,
        })
    };
    assert_eq!((wide(true), wide(false)), (1, 1));
    if let Some(tc) = Toolchain::discover() {
        let ptx = case.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains("ld.global.v4.") && ptx.contains("st.global.v4."), "{ptx}");
    }
}

/// The feature kernel's body (`tests/feature_kernels/raw_ptr.rs`), built here
/// with an assumed alignment of 16 and, to compare, of 4.
fn feature_kernels_raw_vector_copy() -> Kernel {
    vector_copy(16)
}

fn vector_copy(align: usize) -> Kernel {
    kernel("raw_vector_copy", || {
        let x = param_ptr::<f32>("x");
        let y = param_ptr::<f32>("y");
        let t = thread_idx().x;
        let src = make_ptr::<f32>((x + t * 4).toint_i64(), AddressSpace::Gmem, Some(align));
        let dst = make_ptr::<f32>((y + t * 4).toint_i64(), AddressSpace::Gmem, Some(align));
        let (s, d) = (make_tensor(&src, 4), make_tensor(&dst, 4));
        let frag = make_fragment_like::<f32, f32>(&s);
        autovec_copy(&s, &frag);
        autovec_copy(&frag, &d);
    })
}

/// With only 4 bytes assumed, the copy cannot prove a vector aligned: the
/// alignment comes from the pointer, not from the buffer it was made from.
#[test]
fn vector_width_follows_the_assumed_alignment() {
    use cute::ir::Op;
    let k = vector_copy(4);
    assert_eq!(k.ir().count_ops(|op| matches!(op, Op::Load { results, .. } if results.len() > 1)), 0);
    assert_eq!(k.ir().count_ops(|op| matches!(op, Op::Load { .. })), 4);
}

/// An address in no buffer is a wild address, reported at the line of the
/// access.
#[test]
fn wild_address() {
    let k = kernel("wild", || {
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let p = make_ptr::<f32>(Int64::from(0x1000), AddressSpace::Gmem, None);
        out.store(thread_idx().x, p.load());
    });
    let e = sim_err(&k, &mut [KernelArg::f32(vec![0.0; 32])]);
    assert!(e.message.contains("0x1000 is a wild address"), "{e}");
    assert!(e.to_string().contains("tests/raw_pointers.rs"), "{e}");
    // Past the end of a buffer is out of bounds for it.
    let k = kernel("overrun", || {
        let x = param_ptr::<f32>("x");
        let p = make_ptr::<f32>((x + 32).toint_i64(), AddressSpace::Gmem, None);
        p.store_at(thread_idx().x, 1.0);
    });
    let e = sim_err(&k, &mut [KernelArg::f32(vec![0.0; 48])]);
    assert!(e.message.contains("out of bounds for `x`"), "{e}");
    // A shared address in no shared buffer.
    let k = kernel("wild_shared", || {
        let s = alloc_smem::<i32>(32, 16);
        s.store_at(thread_idx().x, 1);
        let p = make_ptr::<i32>((s + 64).toint_i32(), AddressSpace::Smem, None);
        let _ = p.load();
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        out.store(thread_idx().x, p.load());
    });
    let e = sim_err(&k, &mut [KernelArg::i32(vec![0; 32])]);
    assert!(e.message.contains("wild shared address"), "{e}");
}

/// An address that breaks `make_ptr`'s `assumed_align` is caught by the
/// assumption's check.
#[test]
fn broken_assumed_alignment() {
    let k = kernel("misaligned", || {
        let x = param_ptr::<f32>("x");
        let p = make_ptr::<f32>((x.clone() + 1).toint_i64(), AddressSpace::Gmem, Some(16));
        x.store_at(thread_idx().x, p.load());
    });
    let e = sim_err(&k, &mut [KernelArg::f32(vec![0.0; 64])]);
    assert!(e.message.contains("is not a multiple of 16, as the kernel assumes"), "{e}");
}

/// `align` with a non-power of two, and on a tensor-memory pointer; loads
/// through a tensor-memory pointer.
#[test]
fn align_refusals() {
    let e = build_panic(|| {
        kernel("align3", || {
            let x = param_ptr::<f32>("x");
            let _ = (x + thread_idx().x).align(12);
        });
    });
    assert!(e.contains("12 bytes cannot be an alignment, being no power of two"), "{e}");
    let e = build_panic(|| {
        kernel("align_tmem", || {
            let t = make_ptr::<f32>(Int32::from(0), AddressSpace::Tmem, None);
            let _ = t.align(16);
        });
    });
    assert!(e.contains("tensor-memory address is not a byte address"), "{e}");
    let e = build_panic(|| {
        kernel("load_tmem", || {
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            let t = make_ptr::<f32>(Int32::from(0), AddressSpace::Tmem, None);
            out.store(0, t.load());
        });
    });
    assert!(e.contains("tensor memory is not read or written by loads"), "{e}");
    let e = build_panic(|| {
        kernel("bad_assumed", || {
            let x = param_ptr::<f32>("x");
            let _ = make_ptr::<f32>(x.toint(), AddressSpace::Gmem, Some(6));
        });
    });
    assert!(e.contains("assumed alignment of 6 bytes"), "{e}");
}

/// `tospace` between two non-generic spaces is refused when the kernel is
/// built; a generic shared address cast to global is reported by the
/// simulator.
#[test]
fn tospace_refusals() {
    let e = build_panic(|| {
        kernel("gmem_to_smem", || {
            let x = param_ptr::<f32>("x");
            let _ = x.tospace(AddressSpace::Smem);
        });
    });
    assert!(e.contains("a gmem pointer cannot become a smem pointer directly"), "{e}");
    let k = kernel("shared_as_global", || {
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let s = alloc_smem::<f32>(32, 16);
        let t = thread_idx().x;
        s.store_at(t, 2.0);
        let wrong = (s + t).tospace(AddressSpace::Generic).tospace(AddressSpace::Gmem);
        out.store(t, wrong.load());
    });
    let e = sim_err(&k, &mut [KernelArg::f32(vec![0.0; 32])]);
    assert!(e.message.contains("is in the shared window, so it has no global address"), "{e}");
    assert!(e.to_string().contains("tests/raw_pointers.rs"), "{e}");
}

/// Raw pointers in `cp.async` and `ldmatrix` are refused with a message
/// that says why.
#[test]
fn raw_pointers_in_buffer_only_copies() {
    let e = build_panic(|| {
        kernel("raw_cp_async", || {
            let x = param_ptr::<f32>("x");
            let raw = make_ptr::<f32>(x.toint(), AddressSpace::Gmem, Some(16));
            let s = make_smem_tensor::<f32>(Layout::col_major(128));
            let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
            let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
            let thr = tiled.get_slice(thread_idx().x);
            copy(&tiled, &thr.partition_s(&make_tensor(&raw, 128)), &thr.partition_d(&s));
        });
    });
    assert!(e.contains("cp.async: the source is a tensor over a raw gmem pointer"), "{e}");
    let e = build_panic(|| {
        kernel("raw_ldmatrix", || {
            let s = alloc_smem::<F16>(256, 128);
            let raw = make_ptr::<F16>(s.toint_i32(), AddressSpace::Smem, Some(16));
            let src = make_tensor(&raw, Layout::col_major((16, 16)));
            let atom = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
            let tiled = make_tiled_copy_tv(&atom, &Layout::col_major((16, 2)), &Layout::col_major((1, 8)));
            let thr = tiled.get_slice(thread_idx().x);
            let part = thr.partition_s(&src);
            let frag = make_fragment_like::<F16, F16>(&part);
            copy(&tiled, &part, &frag);
        });
    });
    assert!(e.contains("ldmatrix") && e.contains("raw smem pointer"), "{e}");
}

/// The launch's alignment check, which the CUDA launcher applies to every
/// device pointer (cuMemAlloc's and the simulator's are 256-byte aligned,
/// so it is tested here directly).
#[test]
fn pointer_alignment_check() {
    assert!(check_pointer_alignment("x", 0x7f00_0000_1000, 256).is_ok());
    assert!(check_pointer_alignment("x", 0x1010, 16).is_ok());
    let e = check_pointer_alignment("x", 0x1004, 16).unwrap_err();
    assert!(e.contains("`x` is at 0x1004, which is not 16-byte aligned"), "{e}");
    assert!(check_pointer_alignment("y", 0x1004, 4).is_ok());
}

/// A pointer as a `printf` operand prints its address: the simulator's
/// first global buffer is at `SIM_GLOBAL_BASE`, a shared buffer at its start
/// in the block's shared memory, and a generic shared address in the shared
/// window.
#[test]
fn printf_of_pointers() {
    let k = Kernel::builder("print_ptr").block_dim([1, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let s = alloc_smem::<f32>(8, 16);
        s.store_at(2, 0.0);
        printf(
            "%lld %d %lld\n",
            &[(&(x + 2)).into(), (&(s.clone() + 2)).into(), s.tospace(AddressSpace::Generic).into()],
        );
    });
    let mut args = [KernelArg::f32(vec![0.0; 4])];
    let report = k.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    let want = format!("{} 8 {}\n", cute::sim::SIM_GLOBAL_BASE + 8, cute::sim::SIM_SHARED_WINDOW);
    assert_eq!(report.printed, vec![want]);
}

/// An atomic through a generic pointer that reaches a per-thread array is
/// undefined in PTX: the simulator reports it.
#[test]
fn generic_atomic_on_local_memory_is_reported() {
    let k = kernel("generic_atomic_local", || {
        let x = param_ptr::<f32>("x");
        let stack = make_rmem_tensor::<f32>(Layout::col_major(4));
        stack.ptr().store_at(0, 0.0f32);
        let g = stack.ptr().tospace(AddressSpace::Generic);
        let old = atomic_add(&g, 1.0f32);
        x.store_at(thread_idx().x, old);
    });
    let e = sim_err(&k, &mut [KernelArg::f32(vec![0.0; 32])]).to_string();
    assert!(e.contains("per-thread buffer") && e.contains("atomics need global or shared memory"), "{e}");
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-address pointers: `toint` and `make_ptr` round trips, generic
//! pointers and `tospace`, `align` and masking of runtime addresses, a
//! vector copy through an assumed alignment, a pointer as an inline-PTX
//! operand, and a pointer array built on the device. None writes an
//! address to an output the GPU test compares: addresses differ between the
//! simulator and the GPU.

use cute::prelude::*;

use super::Case;

/// Threads of the raw-pointer kernels.
const N: usize = 32;

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [N as u32, 1, 1])
}

/// Global and shared pointers to integers and back: thread `t` reads `x[t]`
/// through `make_ptr(toint(x + t))`, and element `31 - t` of a shared array
/// through `make_ptr(toint(smem + 31 - t), Smem)`, an Int32 address.
pub fn raw_ptr_round_trips() -> Case {
    let kernel = Kernel::builder("raw_ptr_round_trips").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let out = param_tensor::<f32>("out", Layout::col_major((N, 2)));
        let t = thread_idx().x;
        let global = make_ptr::<f32>((x.clone() + t).toint_i64(), AddressSpace::Gmem, Some(4));
        out.store((t, 0), global.load());
        let smem = alloc_smem::<f32>(N, 16);
        (smem.clone() + t).store(x.load_at(t) * 2.0);
        sync_threads();
        let addr: Int32 = (smem + (N as i32 - 1 - t)).toint_i32();
        let shared = make_ptr::<f32>(addr, AddressSpace::Smem, None);
        out.store((t, 1), shared.load());
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..N {
            assert_eq!(out[t], x[t], "raw_ptr_round_trips: global, thread {t}");
            assert_eq!(out[N + t], x[N - 1 - t] * 2.0, "raw_ptr_round_trips: shared, thread {t}");
        }
    }
    Case { name: "raw_ptr_round_trips", kernel, cfg: cfg(), small_only: false, check }
}

/// Generic loads and stores, and `tospace` both ways: thread `t` writes
/// `x[t] + 1` through the generic form of `out + t` and reads it back
/// through that pointer cast to global; it stores `x[t]` into shared memory
/// through a generic pointer and reads element `31 - t` back through a
/// generic pointer cast to shared.
pub fn generic_pointers() -> Case {
    let kernel = Kernel::builder("generic_pointers").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let out = param_tensor::<f32>("out", Layout::col_major((N, 3)));
        let t = thread_idx().x;
        let o = out.ptr() + t;
        let g = o.tospace(AddressSpace::Generic);
        assert_eq!(g.memspace(), AddressSpace::Generic);
        g.store(x.load_at(t) + 1.0);
        let back = g.tospace(AddressSpace::Gmem);
        assert_eq!((back.memspace(), back.alignment()), (AddressSpace::Gmem, 4));
        out.store((t, 1), back.load() * 3.0);
        let smem = alloc_smem::<f32>(N, 16);
        (smem.clone() + t).tospace(AddressSpace::Generic).store(x.load_at(t));
        sync_threads();
        let mirrored = (smem + (N as i32 - 1 - t)).tospace(AddressSpace::Generic).tospace(AddressSpace::Smem);
        out.store((t, 2), mirrored.load());
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..N {
            assert_eq!(out[t], x[t] + 1.0, "generic_pointers: generic store, thread {t}");
            assert_eq!(out[N + t], (x[t] + 1.0) * 3.0, "generic_pointers: generic to global, thread {t}");
            assert_eq!(out[2 * N + t], x[N - 1 - t], "generic_pointers: generic to shared, thread {t}");
        }
    }
    Case { name: "generic_pointers", kernel, cfg: cfg(), small_only: false, check }
}

/// `Ptr::align` of a runtime address and a mask down to a 128-byte
/// boundary: thread `t` reads through `(x + t).align(16)`, element
/// `round_up(t, 4)`, and through `(x + 8t) & !127`, element `32 * (t / 4)`
/// (the buffer starts 256-byte aligned, in the simulator as from
/// `cuMemAlloc`).
pub fn ptr_align_and_mask() -> Case {
    let kernel = Kernel::builder("ptr_align_and_mask").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_ptr::<i32>("x");
        let out = param_tensor::<i32>("out", Layout::col_major((N, 2)));
        let t = thread_idx().x;
        let up = (x.clone() + t).align(16);
        assert_eq!(up.alignment(), 16);
        out.store((t, 0), up.load());
        let down = &(x + t * 8) & !127i64;
        assert_eq!(down.alignment(), 128);
        out.store((t, 1), down.load());
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..N {
            assert_eq!(out[t], x[t.next_multiple_of(4)], "ptr_align_and_mask: align, thread {t}");
            assert_eq!(out[N + t], x[32 * (t / 4)], "ptr_align_and_mask: mask, thread {t}");
        }
    }
    Case { name: "ptr_align_and_mask", kernel, cfg: cfg(), small_only: false, check }
}

/// A 128-bit vector copy through `make_ptr(..., assumed_align=16)`: each
/// thread's four floats move as one 16-byte vector each way, the alignment
/// proved from the pointer's assumed alignment.
pub fn raw_vector_copy() -> Case {
    let kernel = Kernel::builder("raw_vector_copy").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_ptr::<f32>("x");
        let y = param_ptr::<f32>("y");
        let t = thread_idx().x;
        let src = make_ptr::<f32>((x + t * 4).toint_i64(), AddressSpace::Gmem, Some(16));
        let dst = make_ptr::<f32>((y + t * 4).toint_i64(), AddressSpace::Gmem, Some(16));
        let (s, d) = (make_tensor(&src, 4), make_tensor(&dst, 4));
        let frag = make_fragment_like::<f32, f32>(&s);
        autovec_copy(&s, &frag);
        frag.map_in_place(|v| v - 1.0);
        autovec_copy(&frag, &d);
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for i in 0..4 * N {
            assert_eq!(y[i], x[i] - 1.0, "raw_vector_copy: element {i}");
        }
    }
    Case { name: "raw_vector_copy", kernel, cfg: cfg(), small_only: false, check }
}

/// A pointer as an inline-PTX operand: a global pointer is an `l` (64-bit)
/// register and a shared one an `r` (32-bit) register; the PTX adds one
/// element's bytes, and `make_ptr` reads the next element through the
/// result.
pub fn ptr_inline_ptx() -> Case {
    let kernel = Kernel::builder("ptr_inline_ptx").block_dim([N as u32, 1, 1]).build(|| {
        let x = param_ptr::<i32>("x");
        let out = param_tensor::<i32>("out", Layout::col_major((N, 2)));
        let t = thread_idx().x;
        let p = x.clone() + t;
        let next: Int64 = inline_ptx_with_sim("add.s64 $0, $1, 4;", &[(&p).into()], |a| vec![a[0] + 4]);
        out.store((t, 0), make_ptr::<i32>(next, AddressSpace::Gmem, None).load());
        let smem = alloc_smem::<i32>(N + 1, 16);
        (smem.clone() + t).store(x.load_at(t) * 5);
        if_(t.eq(0), || smem.store_at(N, 7));
        sync_threads();
        let s = smem + t;
        let next: Int32 =
            inline_ptx_with_sim("add.s32 $0, $1, 4;", &[s.into()], |a| vec![(a[0] as u32).wrapping_add(4) as u64]);
        out.store((t, 1), make_ptr::<i32>(next, AddressSpace::Smem, None).load());
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..N {
            assert_eq!(out[t], x[t + 1], "ptr_inline_ptx: global, thread {t}");
            let want = if t + 1 < N { x[t + 1].wrapping_mul(5) } else { 7 };
            assert_eq!(out[N + t], want, "ptr_inline_ptx: shared, thread {t}");
        }
    }
    Case { name: "ptr_inline_ptx", kernel, cfg: cfg(), small_only: false, check }
}

/// Pointer parameters in the device-side pointer array.
pub const PTR_ARRAY: usize = 3;

/// A pointer array built on the device, as grouped GEMMs pass operands:
/// thread 0 stores `toint()` of three parameter pointers into a global
/// `i64` array; after `sync_threads` every thread loads each address,
/// dereferences it with `make_ptr` at its own element and adds the three;
/// after another barrier thread 0 clears the array, so that no address
/// reaches an output.
pub fn device_pointer_array() -> Case {
    let kernel = Kernel::builder("device_pointer_array").block_dim([N as u32, 1, 1]).build(|| {
        let a = param_ptr::<i32>("a");
        let b = param_ptr::<i32>("b");
        let c = param_ptr::<i32>("c");
        let ptrs = param_tensor::<i64>("ptrs", Layout::col_major(PTR_ARRAY));
        let out = param_tensor::<i32>("out", Layout::col_major(N));
        let t = thread_idx().x;
        if_(t.eq(0), || {
            for (j, p) in [&a, &b, &c].into_iter().enumerate() {
                ptrs.store(j, p.toint_i64());
            }
        });
        sync_threads();
        let mut sum = Int32::from(0);
        for j in 0..PTR_ARRAY {
            let p = make_ptr::<i32>(ptrs.load(j), AddressSpace::Gmem, Some(4));
            sum = sum + p.load_at(t) * (j as i32 + 1);
        }
        out.store(t, sum);
        sync_threads();
        if_(t.eq(0), || {
            for j in 0..PTR_ARRAY {
                ptrs.store(j, 0i64);
            }
        });
    });
    fn check(args: &[KernelArg]) {
        let (a, b, c) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let (ptrs, out) = (args[3].to_vec::<i64>(), args[4].to_vec::<i32>());
        assert_eq!(ptrs, vec![0; PTR_ARRAY], "device_pointer_array: the array is cleared");
        for t in 0..N {
            let want = a[t].wrapping_add(b[t].wrapping_mul(2)).wrapping_add(c[t].wrapping_mul(3));
            assert_eq!(out[t], want, "device_pointer_array: thread {t}");
        }
    }
    Case { name: "device_pointer_array", kernel, cfg: cfg(), small_only: false, check }
}

/// Every raw-pointer kernel, which the SM89 GPU runs too.
pub fn raw_ptr_cases() -> Vec<Case> {
    vec![
        raw_ptr_round_trips(),
        generic_pointers(),
        ptr_align_and_mask(),
        raw_vector_copy(),
        ptr_inline_ptx(),
        device_pointer_array(),
    ]
}

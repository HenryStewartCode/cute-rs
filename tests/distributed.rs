//! `cutlass.utils.distributed`: multimem load-reduce, store and
//! reduction on multicast buffers in the simulator, their MLIR and PTX (sm_90a, and FP8 on
//! sm_100a; assembled with `ptxas` where it is installed), the flag spin locks (a handshake
//! of co-resident blocks, and a wait that can never succeed), the all-reduce example, and the
//! diagnostics. Multi-GPU execution is untested: one GPU cannot make multicast memory.

#[allow(dead_code)]
#[path = "../examples/all_reduce_two_shot_multimem.rs"]
mod all_reduce;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::ir::MultimemType;
use cute::prelude::*;
use cute::utils::distributed::*;

fn panic_message(f: impl FnOnce()) -> String {
    let e = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

const LEN: usize = 128;

/// Each thread `t` load-reduces 4 Float32 at element `4 t` of `src` and stores the sums to
/// `dst` with `multimem.st`; a 2-register and a 1-register pass do the same on the next
/// halves.
fn f32_kernel() -> Kernel {
    Kernel::builder("mm_f32").block_dim([32, 1, 1]).target(Arch::Sm90a).build(|| {
        let src = param_tensor_multicast::<f32>("src", Layout::col_major(LEN * 2));
        let dst = param_tensor_multicast::<f32>("dst", Layout::col_major(LEN * 2));
        let t = thread_idx().x;
        let regs = multimem_ld_reduce_4xf32(&src.ptr().offset(t * 4));
        multimem_st_4xb32(&dst.ptr().offset(t * 4), regs);
        let two = multimem_ld_reduce_2xf32(&src.ptr().offset(t * 2 + 128));
        multimem_st(&dst.ptr().offset(t * 2 + 128), &two);
        let one = multimem_ld_reduce_1xf32(&src.ptr().offset(t + 192));
        multimem_st_1xb32(&dst.ptr().offset(t + 192), one);
    })
}

fn replicas(world: usize, f: impl Fn(usize, usize) -> f32) -> Vec<KernelArg> {
    (0..world).map(|r| KernelArg::f32((0..LEN * 2).map(|i| f(r, i)).collect())).collect()
}

#[test]
fn multimem_f32_reduces_every_replica_and_stores_to_all() {
    let k = f32_kernel();
    for world in [1, 2, 4] {
        for local in 0..world {
            let value = |r: usize, i: usize| (i as f32) * 0.5 + 10.0 * r as f32;
            let mut args = vec![
                KernelArg::f32((0..LEN * 2).map(|i| value(local, i)).collect()),
                KernelArg::f32(vec![0.0; LEN * 2]),
            ];
            let mut groups = vec![
                MulticastReplicas::new("src", replicas(world, value), local),
                MulticastReplicas::new("dst", replicas(world, |_, _| -1.0), local),
            ];
            k.simulate_multicast(
                LaunchConfig::new([1, 1, 1], [32, 1, 1]),
                &mut args,
                &mut groups,
                &SimOptions::default(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            // 32 threads write 4, 2 and 1 elements: the first 224.
            let want: Vec<f32> = (0..224).map(|i| (0..world).map(|r| value(r, i)).sum()).collect();
            for (r, rep) in groups[1].replicas.iter().enumerate() {
                assert_eq!(rep.as_f32()[..224], want, "world {world}, rank {local}, replica {r}");
                let untouched = if r == local { 0.0 } else { -1.0 };
                assert!(rep.as_f32()[224..].iter().all(|&v| v == untouched));
            }
            assert_eq!(args[1].as_f32()[..224], want, "the local replica is the launch argument");
        }
    }
    // Without replicas a multicast parameter is a group of one.
    let mut args = vec![KernelArg::f32(vec![2.0; LEN * 2]), KernelArg::f32(vec![0.0; LEN * 2])];
    k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].as_f32()[..224], vec![2.0; 224]);
}

#[test]
fn ld_reduce_types_accumulate_and_round() {
    use cute::sim::multimem_ld_reduce_word as ld_reduce_word;
    // Float16 accumulates in Float32: 2048 + 1 + 1 is 2050 (in Float16 steps it would stay 2048).
    let h = |x: f32| F16::from_f32(x).0 as u32;
    let w = |a: u32, b: u32| a | b << 16;
    assert_eq!(
        ld_reduce_word(MultimemType::F16x2, &[w(h(2048.0), h(1.0)), w(h(1.0), h(2.0)), w(h(1.0), h(3.0))]),
        w(h(2050.0), h(6.0))
    );
    // FP8 accumulates in Float16, rounding each step: E4M3 448 + 448 overflows to NaN in E4M3.
    let e4 = |b: u32| b | b << 8 | b << 16 | b << 24;
    let nan = ld_reduce_word(MultimemType::E4m3x4, &[e4(0x7e), e4(0x7e)]);
    assert_eq!(nan & 0x7f, 0x7f, "{nan:#x}");
}

#[test]
fn ld_reduce_dispatch_and_types_in_the_simulator() {
    let k = Kernel::builder("mm_types").block_dim([32, 1, 1]).build(|| {
        let h = param_tensor_multicast::<F16>("h", Layout::col_major(256));
        let b = param_tensor_multicast::<BF16>("b", Layout::col_major(256));
        let out = param_tensor::<i32>("out", Layout::col_major(32 * 8));
        let t = thread_idx().x;
        let hs = multimem_ld_reduce(&h.ptr().offset(t * 8), cute::ir::DType::F16, 8);
        let bs = multimem_ld_reduce_4xbf16(&b.ptr().offset(t * 4 + 128));
        for (k, v) in hs.into_iter().chain(bs).enumerate() {
            out.store(t * 8 + k as i32, v);
        }
    });
    let hv = |r: usize, i: usize| F16::from_f32((i % 16) as f32 + r as f32);
    let bv = |r: usize, i: usize| BF16::from_f32((i % 8) as f32 * 2.0 - r as f32);
    let hreps: Vec<KernelArg> =
        (0..3).map(|r| KernelArg::buffer(&(0..256).map(|i| hv(r, i)).collect::<Vec<_>>())).collect();
    let breps: Vec<KernelArg> =
        (0..3).map(|r| KernelArg::buffer(&(0..256).map(|i| bv(r, i)).collect::<Vec<_>>())).collect();
    let mut args = vec![hreps[1].clone(), breps[1].clone(), KernelArg::i32(vec![0; 256])];
    let mut groups = vec![MulticastReplicas::new("h", hreps, 1), MulticastReplicas::new("b", breps, 1)];
    k.simulate_multicast(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args, &mut groups, &SimOptions::default())
        .unwrap();
    let out = args[2].as_i32();
    for t in 0..32 {
        for k in 0..4 {
            let i = t * 8 + 2 * k;
            let sum = |i: usize| F16::from_f32((0..3).map(|r| hv(r, i).to_f32()).sum());
            let want = sum(i).0 as u32 | (sum(i + 1).0 as u32) << 16;
            assert_eq!(out[t * 8 + k] as u32, want, "f16 thread {t} reg {k}");
            let j = t * 4 + 128 + 2 * (k % 2);
            if k < 2 {
                let bsum = |i: usize| BF16::from_f32((0..3).map(|r| bv(r, i).to_f32()).sum());
                let want = bsum(j).0 as u32 | (bsum(j + 1).0 as u32) << 16;
                assert_eq!(out[t * 8 + 4 + k] as u32, want, "bf16 thread {t} reg {k}");
            }
        }
    }
}

#[test]
fn multimem_red_adds_to_every_replica() {
    let k = Kernel::builder("mm_red").block_dim([64, 1, 1]).build(|| {
        let flags = param_tensor_multicast::<i32>("flags", Layout::col_major(4));
        let t = thread_idx().x;
        multimem_red_relaxed_gpu_add1(&flags.ptr().offset(t % 4));
        if_(t.lt(4), || multimem_red_release_sys_add1(&flags.ptr().offset(t)));
    });
    let reps: Vec<KernelArg> = (0..3i32).map(|r| KernelArg::i32(vec![r * 100; 4])).collect();
    let mut args = vec![reps[2].clone()];
    let mut groups = vec![MulticastReplicas::new("flags", reps, 2)];
    k.simulate_multicast(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &mut groups, &SimOptions::default())
        .unwrap();
    for (r, rep) in groups[0].replicas.iter().enumerate() {
        assert_eq!(rep.as_i32(), vec![r as i32 * 100 + 17; 4], "replica {r}");
    }
}

#[test]
fn multimem_mlir_and_ptx() {
    let mlir = f32_kernel().to_mlir();
    for inst in [
        "multimem.ld_reduce.weak.global.add.v4.f32 {$0, $1, $2, $3}, [$4];",
        "multimem.st.weak.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "multimem.ld_reduce.weak.global.add.v2.f32 {$0, $1}, [$2];",
        "multimem.ld_reduce.weak.global.add.f32 $0, [$1];",
        "multimem.st.weak.global.f32 [$0], $1;",
    ] {
        assert!(mlir.contains(inst), "{inst} in {mlir}");
    }
    let Some(toolchain) = Toolchain::discover() else { return };
    let ptx = f32_kernel().compile_ptx(&toolchain, Arch::Sm90a).unwrap();
    assert!(ptx.contains(".version 8.1") || ptx.contains(".version 8."), "{ptx}");
    assert!(ptx.contains("multimem.ld_reduce.weak.global.add.v4.f32"), "{ptx}");
    // Every kind, assembled where ptxas is installed.
    let all = Kernel::builder("mm_all").block_dim([32, 1, 1]).build(|| {
        let p = param_tensor_multicast::<u32>("p", Layout::col_major(1024));
        let out = param_tensor::<i32>("out", Layout::col_major(1024));
        let t = thread_idx().x;
        let mut regs = Vec::new();
        regs.extend(multimem_ld_reduce_8xf16(&p.ptr().offset(t * 4)));
        regs.extend(multimem_ld_reduce_8xbf16(&p.ptr().offset(t * 4 + 128)));
        regs.extend(multimem_ld_reduce_4xf16(&p.ptr().offset(t * 2 + 256)));
        regs.extend(multimem_ld_reduce_4xbf16(&p.ptr().offset(t * 2 + 320)));
        regs.push(multimem_ld_reduce_2xf16(&p.ptr().offset(t + 384)));
        regs.push(multimem_ld_reduce_2xbf16(&p.ptr().offset(t + 416)));
        for (k, r) in regs.iter().enumerate() {
            out.store(t * 16 + k as i32, *r);
        }
        multimem_st_2xb32(&p.ptr().offset(t * 2 + 512), [regs[0], regs[1]]);
        multimem_red_relaxed_sys_add1(&p.ptr().offset(t + 600));
        multimem_red_release_gpu_add1(&p.ptr().offset(t + 640));
        multimem_red_add1(&p.ptr().offset(t + 680), MemoryOrder::Relaxed, MemoryScope::Gpu);
    });
    let opts = CompileOptions::new().gpu_arch(Arch::Sm90a);
    let c = all.compile_with(&toolchain, &opts).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(c.ptx_version(), 82, "the .acc::f32 forms need PTX ISA 8.2");
    for inst in [
        "multimem.ld_reduce.weak.global.add.acc::f32.v4.f16x2",
        "multimem.ld_reduce.weak.global.add.acc::f32.v4.bf16x2",
        "multimem.ld_reduce.weak.global.add.acc::f32.v2.f16x2",
        "multimem.ld_reduce.weak.global.add.acc::f32.bf16x2",
        "multimem.st.weak.global.v2.f32",
        "multimem.red.relaxed.sys.global.add.s32",
        "multimem.red.release.gpu.global.add.s32",
        "multimem.red.relaxed.gpu.global.add.s32",
    ] {
        assert!(c.ptx().contains(inst), "{inst} in {}", c.ptx());
    }
    if toolchain.cuda_tools().ptxas.is_some() {
        let c = all.compile_with(&toolchain, &opts.clone().use_ptxas(true)).unwrap_or_else(|e| panic!("{e}"));
        assert!(c.cubin().is_some());
        f32_kernel().compile_with(&toolchain, &opts.use_ptxas(true)).unwrap_or_else(|e| panic!("{e}"));
    }
}

fn fp8_kernel(target: Option<Arch>) -> Kernel {
    let mut b = Kernel::builder("mm_fp8").block_dim([32, 1, 1]);
    if let Some(a) = target {
        b = b.target(a);
    }
    b.build(|| {
        let p = param_tensor_multicast::<u8>("p", Layout::col_major(4096));
        let out = param_tensor::<i32>("out", Layout::col_major(32 * 7));
        let t = thread_idx().x;
        let mut regs = Vec::new();
        regs.extend(multimem_ld_reduce_16xe4m3(&p.ptr().offset(t * 16)));
        regs.extend(multimem_ld_reduce_8xe5m2(&p.ptr().offset(t * 8 + 1024)));
        regs.push(multimem_ld_reduce_4xe4m3(&p.ptr().offset(t * 4 + 2048)));
        for (k, r) in regs.iter().enumerate() {
            out.store(t * 7 + k as i32, *r);
        }
    })
}

#[test]
fn fp8_ld_reduce_needs_sm100a() {
    let msg = panic_message(|| {
        fp8_kernel(Some(Arch::Sm90a));
    });
    assert!(msg.contains("an FP8 multimem_ld_reduce") && msg.contains("sm_100a"), "{msg}");
    // The PTX ISA gives the sm_120 family no FP8 multimem: only sm_120a and sm_121a.
    let msg = panic_message(|| {
        fp8_kernel(Some(Arch::Sm120f));
    });
    assert!(msg.contains("an FP8 multimem_ld_reduce"), "{msg}");
    fp8_kernel(Some(Arch::Sm120a));
    fp8_kernel(Some(Arch::Sm100f));
    let k = fp8_kernel(None);
    // In the simulator: E4M3 1.0 (0x38) on three replicas is 3.0 (0x44); E5M2 1.0 (0x3c) is 0x42.
    let reps: Vec<KernelArg> = (0..3)
        .map(|_| {
            KernelArg::buffer(
                &(0..4096).map(|i| if (1024..2048).contains(&i) { 0x3cu8 } else { 0x38 }).collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut args = vec![reps[0].clone(), KernelArg::i32(vec![0; 32 * 7])];
    let mut groups = vec![MulticastReplicas::new("p", reps, 0)];
    k.simulate_multicast(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args, &mut groups, &SimOptions::default())
        .unwrap();
    for t in 0..32 {
        let row = &args[1].as_i32()[t * 7..t * 7 + 7];
        assert_eq!(row[..4], [0x4444_4444; 4]);
        assert_eq!(row[4..6], [0x4242_4242; 2]);
        assert_eq!(row[6], 0x4444_4444);
    }
    let Some(toolchain) = Toolchain::discover() else { return };
    let err = k.compile_ptx(&toolchain, Arch::Sm90a).unwrap_err();
    assert!(err.to_string().contains("sm_100a"), "{err}");
    let opts = CompileOptions::new().gpu_arch(Arch::Sm100a);
    let c = k.compile_with(&toolchain, &opts).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.ptx_version() >= 86);
    assert!(c.ptx().contains("multimem.ld_reduce.weak.global.add.acc::f16.v4.e4m3x4"), "{}", c.ptx());
    assert!(c.ptx().contains("multimem.ld_reduce.weak.global.add.acc::f16.v2.e5m2x4"), "{}", c.ptx());
    if toolchain.cuda_tools().ptxas.is_some() {
        k.compile_with(&toolchain, &opts.use_ptxas(true)).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn diagnostics() {
    // An ordinary load from a multicast buffer.
    let msg = panic_message(|| {
        Kernel::build("mm_plain_load", || {
            let p = param_tensor_multicast::<f32>("p", Layout::col_major(32));
            let q = param_tensor::<f32>("q", Layout::col_major(32));
            q.store(0, p.load(0));
        });
    });
    assert!(msg.contains("load on multicast buffer `p`"), "{msg}");
    // A multimem instruction on a unicast buffer.
    let msg = panic_message(|| {
        Kernel::build("mm_unicast", || {
            let p = param_tensor::<f32>("p", Layout::col_major(32));
            let _ = multimem_ld_reduce_1xf32(&p.ptr());
        });
    });
    assert!(msg.contains("which is not a multicast buffer"), "{msg}");
    // An unsupported dtype or element count in the dispatcher.
    let msg = panic_message(|| {
        Kernel::build("mm_dispatch_type", || {
            let p = param_tensor_multicast::<i32>("p", Layout::col_major(32));
            let _ = multimem_ld_reduce(&p.ptr(), cute::ir::DType::I32, 4);
        });
    });
    assert!(msg.contains("multimem_ld_reduce has no Int32 variant") || msg.contains("has no i32 variant"), "{msg}");
    let msg = panic_message(|| {
        Kernel::build("mm_dispatch_count", || {
            let p = param_tensor_multicast::<f32>("p", Layout::col_major(32));
            let _ = multimem_ld_reduce(&p.ptr(), cute::ir::DType::F32, 3);
        });
    });
    assert!(msg.contains("multimem_ld_reduce of 3"), "{msg}");
    let msg = panic_message(|| {
        Kernel::build("mm_st_count", || {
            let p = param_tensor_multicast::<f32>("p", Layout::col_major(32));
            multimem_st(&p.ptr(), &[thread_idx().x; 3]);
        });
    });
    assert!(msg.contains("multimem_st stores 4, 2 or 1 registers, not 3"), "{msg}");
    // A misaligned multimem address (offset 2 of a 16-byte access).
    let k = Kernel::build("mm_misaligned", || {
        let p = param_tensor_multicast::<f32>("p", Layout::col_major(64));
        let off = param_scalar::<i32>("off");
        let _ = multimem_ld_reduce_4xf32(&p.ptr().offset(off));
    });
    let err = k
        .simulate(
            LaunchConfig::new([1, 1, 1], [1, 1, 1]),
            &mut [KernelArg::f32(vec![0.0; 64]), KernelArg::ScalarI32(2)],
        )
        .unwrap_err();
    assert!(err.to_string().contains("misaligned multimem.ld_reduce"), "{err}");
    // Order and scope.
    let msg = panic_message(|| {
        Kernel::build("mm_order", || {
            let p = param_tensor_multicast::<i32>("p", Layout::col_major(32));
            multimem_red_add1(&p.ptr(), MemoryOrder::Acquire, MemoryScope::Sys);
        });
    });
    assert!(msg.contains("takes order Release or Relaxed"), "{msg}");
}

/// Co-resident blocks, `expected` of them: thread 0 of each adds one to every block's flag
/// (`red_add1`, release), then spins on its own flag until it holds `expected` (with `spin`;
/// the CAS waits reset it to 0, as the all-reduce does), then records its block.
fn handshake(spin: &'static str) -> Kernel {
    Kernel::build(format!("flag_handshake_{spin}"), move || {
        let flags = param_tensor::<i32>("flag", Layout::col_major(8));
        let out = param_tensor::<i32>("out", Layout::col_major(8));
        let expected = param_scalar::<i32>("expected");
        let t = thread_idx().x;
        let b = block_idx().x;
        if_(t.eq(0), || {
            for j in 0..8 {
                if_(expected.gt(j), || red_add1(&flags.ptr().offset(j), MemoryOrder::Release, MemoryScope::Gpu));
            }
            let own = flags.ptr().offset(b);
            match spin {
                "ld" => spin_lock_ld_lt_relaxed_wait(&own, expected, MemoryScope::Gpu),
                "cas_relaxed" => spin_lock_atom_cas_relaxed_wait(&own, expected, 0, MemoryScope::Gpu),
                _ => spin_lock_atom_cas_acquire_wait(&own, expected, 0, MemoryScope::Gpu),
            }
            out.store(b, b + 1);
        });
    })
}

#[test]
fn flag_handshake_of_co_resident_blocks() {
    for spin in ["ld", "cas_relaxed", "cas_acquire"] {
        for blocks in [2u32, 5, 8] {
            let k = handshake(spin);
            let mut args =
                vec![KernelArg::i32(vec![0; 8]), KernelArg::i32(vec![0; 8]), KernelArg::ScalarI32(blocks as i32)];
            let cfg = LaunchConfig::new([blocks, 1, 1], [32, 1, 1]).with_cooperative(true);
            k.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{spin}, {blocks} blocks: {e}"));
            let left = if spin == "ld" { blocks as i32 } else { 0 };
            let flags: Vec<i32> = (0..8).map(|b| if b < blocks as i32 { left } else { 0 }).collect();
            assert_eq!(args[0].as_i32(), flags, "{spin}, {blocks} blocks");
            let want: Vec<i32> = (0..8).map(|b| if b < blocks as i32 { b + 1 } else { 0 }).collect();
            assert_eq!(args[1].as_i32(), want, "{spin}, {blocks} blocks");
        }
    }
}

#[test]
fn a_spin_wait_that_cannot_complete_is_reported() {
    for spin in ["ld", "cas_relaxed"] {
        let k = handshake(spin);
        // Expecting 3 arrivals from 2 blocks: the wait never succeeds.
        let mut args = vec![KernelArg::i32(vec![0; 8]), KernelArg::i32(vec![0; 8]), KernelArg::ScalarI32(3)];
        let opts = SimOptions { max_idle_steps: 20_000, ..SimOptions::default() };
        let cfg = LaunchConfig::new([2, 1, 1], [32, 1, 1]).with_cooperative(true);
        let err = k.simulate_with(cfg, &mut args, &opts).unwrap_err().to_string();
        assert!(err.contains("spins") && err.contains("while_carry"), "{spin}: {err}");
        assert!(err.contains("polling flag["), "{spin}: {err}");
    }
}

#[test]
fn all_reduce_example_on_one_rank_at_a_time() {
    let p = all_reduce::Problem { m: 32, n: 256, rows_per_thread: 2, world_size: 2 };
    for rank in 0..2 {
        let (outs, flags) = all_reduce::simulate_rank(p, rank);
        let want = all_reduce::expected_output(p, rank);
        for (r, out) in outs.iter().enumerate() {
            assert_eq!(*out, want, "rank {rank}, replica {r}");
        }
        assert!(flags.iter().all(|&f| f == 0));
    }
    let p4 = all_reduce::Problem { m: 16, n: 128, rows_per_thread: 1, world_size: 4 };
    let (outs, _) = all_reduce::simulate_rank(p4, 3);
    assert_eq!(outs[0], all_reduce::expected_output(p4, 3));
    let Some(toolchain) = Toolchain::discover() else { return };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm90a);
    let c = all_reduce::all_reduce_kernel(p, 0).compile_with(&toolchain, &opts).unwrap_or_else(|e| panic!("{e}"));
    for inst in [
        "multimem.ld_reduce.weak.global.add.v4.f32",
        "multimem.st.weak.global.v4.f32",
        "multimem.red.release.sys.global.add.s32",
        "atom.relaxed.sys.global.cas.b32",
    ] {
        assert!(c.ptx().contains(inst), "{inst} in {}", c.ptx());
    }
    if toolchain.cuda_tools().ptxas.is_some() {
        all_reduce::all_reduce_kernel(p, 1)
            .compile_with(&toolchain, &opts.use_ptxas(true))
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

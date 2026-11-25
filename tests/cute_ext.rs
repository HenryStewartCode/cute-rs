//! cute_ext foundations (roadmap part m11-cute-ext-core): `cute::experimental`'s allocate,
//! partition, copy with automatic predication, warp_reduce, elementwise and dot, simulated
//! against plain Rust; the pooled TMEM allocation in the simulator and (when `mlir-opt` and
//! `ptxas` are installed) assembled for sm_100a; and the build-time diagnostics.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::experimental::{self, OperationType};
use cute::prelude::*;
use cute::tcgen05::{Ld32x32bOp, Repetition, St32x32bOp, make_tmem_copy};

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn one_warp() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [32, 1, 1])
}

/// A warp_reduce kernel: each lane loads `n_in` values of `input` (lane-major), reduces them
/// with `kind` over `map`, and stores its `n_out` results.
fn reduce_kernel(map: Layout, kind: OperationType, n_in: usize, n_out: usize, max: bool) -> Kernel {
    Kernel::builder("warp_reduce").block_dim([32, 1, 1]).build(move || {
        let input = param_tensor::<i32>("input", Layout::row_major((32, n_in)));
        let output = param_tensor::<i32>("output", Layout::row_major((32, n_out)));
        let lane = thread_idx().x;
        let x = make_fragment::<i32>(n_in);
        let y = make_fragment::<i32>(n_out);
        for i in 0..n_in {
            x.store(i, input.load((lane, i as i32)));
        }
        if max {
            experimental::warp_reduce(&x, &y, &map, kind, |a, b| select(a.gt(b), a, b));
        } else {
            experimental::warp_reduce(&x, &y, &map, kind, |a, b| a + b);
        }
        for i in 0..n_out {
            output.store((lane, i as i32), y.load(i));
        }
    })
}

fn reduce_inputs(n: usize) -> Vec<i32> {
    (0..32 * n).map(|i| ((i * 7919) % 201) as i32 - 100).collect()
}

/// The group (stride-0 coordinates) and logical lane of each physical lane under `map`.
fn lane_roles(map: &[(usize, usize)]) -> Vec<(usize, usize)> {
    (0..32)
        .map(|lane| {
            let (mut rest, mut group, mut gscale, mut logical) = (lane, 0, 1, 0);
            for &(n, d) in map {
                let c = rest % n;
                rest /= n;
                if d == 0 {
                    group += c * gscale;
                    gscale *= n;
                } else {
                    logical += c * d;
                }
            }
            (group, logical)
        })
        .collect()
}

#[test]
fn warp_reduce_down_shuffle_matches_rust() {
    let maps: [(&[(usize, usize)], bool); 3] =
        [(&[(32, 1)], false), (&[(8, 1), (4, 0)], true), (&[(4, 0), (8, 1)], false)];
    for (flat, max) in maps {
        let map = Layout::from_flat(flat);
        let kernel = reduce_kernel(map.clone(), OperationType::ReduceDownShuffle, 3, 3, max);
        let input = reduce_inputs(3);
        let mut args = vec![KernelArg::buffer(&input), KernelArg::buffer(&vec![0i32; 96])];
        kernel.simulate(one_warp(), &mut args).unwrap_or_else(|e| panic!("{map}: {e}"));
        let out = args[1].to_vec::<i32>();
        let roles = lane_roles(flat);
        for lane in 0..32 {
            for i in 0..3 {
                let peers = (0..32).filter(|&p| roles[p].0 == roles[lane].0).map(|p| input[p * 3 + i]);
                let want = if max { peers.max().unwrap() } else { peers.sum() };
                assert_eq!(out[lane * 3 + i], want, "{map}: lane {lane} element {i}");
            }
        }
    }
}

#[test]
fn warp_reduce_swap_shuffle_matches_rust() {
    // One reduction over the warp (32 slices of 1) and four over eight lanes (8 slices of 2).
    for (flat, frag) in [(&[(32usize, 1usize)][..], 1usize), (&[(8, 1), (4, 0)][..], 2)] {
        let map = Layout::from_flat(flat);
        let roles = lane_roles(flat);
        let groups = roles.iter().map(|r| r.0).max().unwrap() + 1;
        let n = 32 / groups; // lanes per group
        let kernel = reduce_kernel(map.clone(), OperationType::ReduceSwapShuffle, n * frag, frag, false);
        let input = reduce_inputs(n * frag);
        let mut args = vec![KernelArg::buffer(&input), KernelArg::buffer(&vec![0i32; 32 * frag])];
        kernel.simulate(one_warp(), &mut args).unwrap_or_else(|e| panic!("{map}: {e}"));
        let out = args[1].to_vec::<i32>();
        for lane in 0..32 {
            let (group, slice) = roles[lane];
            for f in 0..frag {
                let want: i32 =
                    (0..32).filter(|&p| roles[p].0 == group).map(|p| input[p * n * frag + slice * frag + f]).sum();
                assert_eq!(out[lane * frag + f], want, "{map}: lane {lane} element {f}");
            }
        }
    }
}

#[test]
fn elementwise_matches_rust() {
    let kernel = Kernel::builder("elementwise").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::row_major((32, 4)));
        let b = param_tensor::<i32>("b", Layout::row_major((32, 4)));
        let c = param_tensor::<f32>("c", Layout::row_major((32, 4)));
        let d = param_tensor::<i32>("d", Layout::row_major((32, 4)));
        let t = thread_idx().x;
        let (a, b, c, d) = (a.slice((t, ..)), b.slice((t, ..)), c.slice((t, ..)), d.slice((t, ..)));
        // Register outputs not yet written read as zero; then a memory output's current value.
        let acc = make_fragment::<f32>(4);
        let cnt = make_fragment::<i32>(4);
        experimental::elementwise((&a, &b), (&acc, &cnt), |(x, y), (s, k)| (s + x * y.to::<f32>(), k + 1));
        experimental::elementwise((&acc, &cnt), (&c, &d), |(s, k), (old, _)| (s + old, k * 10));
    });
    let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();
    let b: Vec<i32> = (0..128).map(|i| i % 7 - 3).collect();
    let c: Vec<f32> = (0..128).map(|i| 1000.0 + i as f32).collect();
    let mut args = vec![
        KernelArg::f32(a.clone()),
        KernelArg::buffer(&b),
        KernelArg::f32(c.clone()),
        KernelArg::buffer(&[0i32; 128]),
    ];
    kernel.simulate(one_warp(), &mut args).unwrap();
    let (got_c, got_d) = (args[2].as_f32(), args[3].to_vec::<i32>());
    for i in 0..128 {
        assert_eq!(got_c[i], a[i] * b[i] as f32 + c[i], "c[{i}]");
        assert_eq!(got_d[i], 10, "d[{i}]");
    }
}

/// `dot` with a bare MMA atom and rank-1 fragments (which it gives trailing unit modes):
/// one 16x8x8 MMA, `D = A * B^T + D`.
#[test]
fn dot_with_a_bare_atom_matches_rust() {
    let kernel = Kernel::builder("dot").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((16, 8)));
        let b = param_tensor::<F16>("b", Layout::row_major((8, 8)));
        let d = param_tensor::<f32>("d", Layout::row_major((16, 8)));
        let atom = make_mma_atom(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 8)));
        let tiled = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let thr = tiled.get_slice(thread_idx().x);
        let (ga, gb, gd) = (thr.partition_a(&a), thr.partition_b(&b), thr.partition_c(&d));
        let (ra, rb, rc) =
            (make_fragment_like::<F16, _>(&ga), make_fragment_like::<F16, _>(&gb), make_fragment_like::<f32, _>(&gd));
        basic_copy(&ga, &ra);
        basic_copy(&gb, &rb);
        basic_copy(&gd, &rc);
        let rank1 = |t: &Tensor<F16>| Tileable::with_layout(t, Layout::from_modes([t.layout().mode(0)]));
        experimental::dot(&atom, &rank1(&ra), &rank1(&rb), &rc);
        basic_copy(&rc, &gd);
    });
    let a: Vec<f32> = (0..128).map(|i| (i % 9) as f32 - 4.0).collect();
    let b: Vec<f32> = (0..64).map(|i| (i % 5) as f32 - 2.0).collect();
    let d0: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let mut args = vec![KernelArg::f16_from_f32(&a), KernelArg::f16_from_f32(&b), KernelArg::f32(d0.clone())];
    kernel.simulate(one_warp(), &mut args).unwrap();
    let d = args[2].as_f32();
    for i in 0..16 {
        for j in 0..8 {
            let want: f32 = d0[i * 8 + j] + (0..8).map(|k| a[i * 8 + k] * b[j * 8 + k]).sum::<f32>();
            assert_eq!(d[i * 8 + j], want, "D[{i}][{j}]");
        }
    }
}

/// `allocate` in shared memory with a swizzle, `partition` by a TV layout, and
/// `simt_auto_vec_copy` (synchronous and `cp.async`), on a ragged 2-D tensor: the copies
/// from and to global memory are predicated by its shape.
fn smem_round_trip(rows: usize) -> Kernel {
    Kernel::builder("smem_round_trip").block_dim([32, 1, 1]).build(move || {
        let src = param_tensor::<f32>("src", Layout::row_major((rows, 32)));
        let dst = param_tensor::<f32>("dst", Layout::row_major((rows, 32)));
        let tile = Tiler::from(&IntTuple::from((8, 32)));
        let blk = block_idx().x;
        let g_src = local_tile(&src, tile.clone(), (blk, 0));
        let g_dst = local_tile(&dst, tile, (blk, 0));
        let sw = make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32)));
        let s = experimental::allocate::<f32>(AddressSpace::Smem, sw, 512);
        // Thread t owns row t / 4, columns 8 * (t % 4) ..: 8 values of a 8 x 32 tile.
        let tv = Layout::new(((4, 8), 8), ((64, 1), 8));
        let t = thread_idx().x;
        let part = |x: &Tensor<f32>| experimental::partition(x, t, &tv, Tiler::from(&IntTuple::from((8, 32))));
        experimental::simt_auto_vec_copy(&part(&g_src), &part(&s), true);
        cp_async_commit_group();
        cp_async_wait_group(0);
        sync_threads();
        let r = experimental::allocate::<f32>(AddressSpace::Rmem, Layout::col_major((8, 1)), 16);
        experimental::simt_auto_vec_copy(&part(&s), &r, false);
        r.map_in_place(|v| v * 2.0f32);
        experimental::simt_auto_vec_copy(&r, &part(&g_dst), false);
    })
}

#[test]
fn smem_allocate_partition_and_predicated_copies() {
    for rows in [16, 13] {
        let kernel = smem_round_trip(rows);
        let src: Vec<f32> = (0..rows * 32).map(|i| i as f32).collect();
        let mut args = vec![KernelArg::f32(src.clone()), KernelArg::f32(vec![-1.0; rows * 32])];
        let cfg = LaunchConfig::new([rows.div_ceil(8) as u32, 1, 1], [32, 1, 1]);
        kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{rows} rows: {e}"));
        let dst = args[1].as_f32();
        for i in 0..rows * 32 {
            assert_eq!(dst[i], src[i] * 2.0, "{rows} rows: dst[{i}]");
        }
    }
}

/// `allocate` in tensor memory: two buffers of one pooled allocation (32 columns, then 16 at
/// a 32-column alignment: 64 columns allocated), each written and read back through
/// `tcgen05.st`/`tcgen05.ld`.
fn tmem_pool_kernel() -> Kernel {
    Kernel::builder("cute_ext_tmem_pool").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let input = param_tensor::<f32>("input", Layout::new((128, 48), (48, 1)));
        let output = param_tensor::<f32>("output", Layout::new((128, 48), (48, 1)));
        let t = thread_idx().x;
        let ta = experimental::allocate::<f32>(AddressSpace::Tmem, Layout::new((128, 32), (1 << 16, 1)), 32);
        let tb = experimental::allocate::<f32>(AddressSpace::Tmem, Layout::new((128, 16), (1 << 16, 1)), 32);
        let cols = |c0: i32, n: i32| {
            let tile = Tiler::from(&IntTuple::from((128, n as usize)));
            (local_tile(&input, tile.clone(), (0, c0 / n)), local_tile(&output, tile, (0, c0 / n)))
        };
        for (tm, (src, dst), reps) in [(&ta, cols(0, 32), Repetition::X32), (&tb, cols(32, 16), Repetition::X16)] {
            let st = make_tmem_copy(&make_copy_atom::<f32>(St32x32bOp::new(reps)), tm);
            let st_thr = st.get_slice(t);
            let frag = make_fragment_like::<f32, _>(&st_thr.partition_s(&src));
            basic_copy(&st_thr.partition_s(&src), &frag);
            copy(&st, &frag, &st_thr.partition_d(tm));
            fence_view_async_tmem_store();
            let ld = make_tmem_copy(&make_copy_atom::<f32>(Ld32x32bOp::new(reps)), tm);
            let ld_thr = ld.get_slice(t);
            let back = make_fragment_like::<f32, _>(&ld_thr.partition_d(&dst));
            copy(&ld, &ld_thr.partition_s(tm), &back);
            fence_view_async_tmem_load();
            basic_copy(&back, &ld_thr.partition_d(&dst));
        }
    })
}

#[test]
fn tmem_allocate_pools_one_allocation() {
    let kernel = tmem_pool_kernel();
    let input: Vec<f32> = (0..128 * 48).map(|i| i as f32 - 3000.0).collect();
    let mut args = vec![KernelArg::f32(input.clone()), KernelArg::f32(vec![0.0; 128 * 48])];
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    kernel.simulate_all_orders(cfg, &mut args, &SimOptions::default()).unwrap();
    assert_eq!(args[1].as_f32(), input.as_slice());
    let mlir = kernel.to_mlir();
    assert_eq!(mlir.matches("nvvm.tcgen05.alloc").count(), 1, "{mlir}");
    assert_eq!(mlir.matches("nvvm.tcgen05.dealloc").count(), 1, "{mlir}");
    assert!(mlir.contains("nvvm.tcgen05.relinquish_alloc_permit"), "{mlir}");
    // The prologue comes first: the allocation precedes every TMEM store.
    let alloc = mlir.find("nvvm.tcgen05.alloc").unwrap();
    let st = mlir.find("nvvm.tcgen05.st").expect("a TMEM store");
    assert!(alloc < st, "{mlir}");
}

/// The TMEM kernel assembled for sm_100a when `mlir-opt` and `ptxas` are installed.
#[test]
fn tmem_allocate_assembles_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = tmem_pool_kernel().compile_ptx(&tc, Arch::Sm100a).unwrap();
    assert!(ptx.contains("tcgen05.alloc.cta_group::1"), "{ptx}");
    let ptxas = std::env::var("CUTE_PTXAS").unwrap_or_else(|_| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute_ext_tmem_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ptx");
    std::fs::write(&file, &ptx).unwrap();
    match std::process::Command::new(&ptxas).arg("-arch=sm_100a").arg(&file).arg("-o").arg(dir.join("k.cubin")).output()
    {
        Ok(out) => assert!(out.status.success(), "ptxas: {}", String::from_utf8_lossy(&out.stderr)),
        Err(_) => eprintln!("skipping the assembly: no ptxas found"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn diagnostics() {
    let tmem = || Layout::new((128, 32), (1 << 16, 1));
    let msg = build_panic(|| {
        Kernel::builder("k").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
            if_(thread_idx().x.lt(64), || {
                experimental::allocate::<f32>(AddressSpace::Tmem, tmem(), 32);
            });
        });
    });
    assert!(msg.contains("top level") && msg.contains("if_"), "{msg}");

    let msg = build_panic(|| {
        Kernel::builder("k").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
            experimental::allocate::<f32>(AddressSpace::Tmem, Layout::new((128, 512), (1 << 16, 1)), 32);
            experimental::allocate::<f32>(AddressSpace::Tmem, tmem(), 32);
        });
    });
    assert!(msg.contains("544 columns") && msg.contains("512"), "{msg}");

    let period_bytes = make_swizzle(3, 4, 3).period() * 2;
    let msg = build_panic(|| {
        Kernel::build("k", || {
            let sw = make_composed_layout(make_swizzle(3, 4, 3), 0, Layout::row_major((16, 64)));
            experimental::allocate::<F16>(AddressSpace::Smem, sw, 128);
        });
    });
    assert!(msg.contains(&format!("{period_bytes} bytes")) && msg.contains("has 128"), "{msg}");

    for (map, want) in [(Layout::col_major(16), "covers 16"), (Layout::col_major((12, 2)), "12 lanes")] {
        let msg = build_panic(|| {
            Kernel::build("k", || {
                let x = make_fragment::<f32>(1);
                x.fill(1.0f32);
                experimental::warp_reduce(&x, &x, &map, OperationType::ReduceDownShuffle, |a, b| a + b);
            });
        });
        assert!(msg.contains(want), "{map}: {msg}");
    }

    let msg = build_panic(|| {
        Kernel::build("k", || {
            let mma =
                make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 8)), Layout::col_major((1, 1, 1)));
            let a = make_fragment::<F16>(((2, 2), 2, 1));
            let b = make_fragment::<F16>((2, 1, 1));
            let c = make_fragment::<f32>(((2, 2), 1, 1));
            experimental::dot(&mma, &a, &b, &c);
        });
    });
    assert!(msg.contains("operand shapes do not form an MMA"), "{msg}");

    let msg = build_panic(|| {
        Kernel::build("k", || {
            let x = param_tensor::<f32>("x", Layout::col_major((40, 4)));
            let lost = coalesce(&local_tile(&x, Tiler::from(&IntTuple::from((32, 4))), (1, 0)));
            let r = make_fragment::<f32>(128);
            experimental::copy(&lost, &r, &make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }));
        });
    });
    assert!(msg.contains("predicated_tensor_origin"), "{msg}");
}

#[test]
fn operation_type_names_round_trip() {
    for op in OperationType::ALL {
        assert_eq!(OperationType::from_name(op.name()), Some(op));
    }
    assert_eq!(OperationType::ReduceSwapShuffle.to_string(), "REDUCE_SWAP_SHUFFLE");
}

thread_local! {
    static PANIC_AT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// The message and `file:line` of the panic of `f`.
fn panic_location(f: impl FnOnce()) -> (String, String) {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let at = info.location().map(|l| format!("{}:{}", l.file(), l.line()));
            PANIC_AT.with(|p| *p.borrow_mut() = at);
            default(info);
        }));
    });
    let msg = build_panic(f);
    let at = PANIC_AT.with(|p| p.borrow_mut().take()).expect("the panic hook recorded a location");
    (msg, at)
}

/// The view operations that record themselves on the predication origin still report a bad
/// argument at the caller's line, on a global tensor (which has an origin).
#[test]
fn view_ops_on_global_tensors_report_the_callers_line() {
    let bad = || Layout::new(4, 2);
    let bad_tiler = || Tiler::from(Tree::Leaf(Some(Layout::new(4, 2))));
    type Case = Box<dyn FnOnce(&Tensor<f32>)>;
    let cases: Vec<(&str, Case)> = vec![
        ("composition", Box::new(move |t| drop(composition(t, &bad())))),
        ("compose", Box::new(move |t| drop(t.compose(bad())))),
        ("logical_divide", Box::new(move |t| drop(logical_divide(t, bad_tiler())))),
        ("zipped_divide", Box::new(move |t| drop(zipped_divide(t, bad_tiler())))),
        ("tiled_divide", Box::new(move |t| drop(tiled_divide(t, bad_tiler())))),
        ("flat_divide", Box::new(move |t| drop(flat_divide(t, bad_tiler())))),
        ("group_modes", Box::new(|t| drop(t.group_modes(1, 5)))),
        ("slice", Box::new(|t| drop(t.slice((1, 2, 3))))),
        ("domain_offset", Box::new(|t| drop(domain_offset((1, 2, 3), t)))),
        (
            "partition",
            Box::new(|t| {
                drop(experimental::partition(t, 0, &Layout::col_major(4), Tiler::from(&IntTuple::from((2, 2)))))
            }),
        ),
    ];
    for (what, f) in cases {
        let (msg, at) = panic_location(|| {
            Kernel::build("k", || {
                let t = param_tensor::<f32>("t", Layout::new((6, 8), (8, 1)));
                f(&t);
            });
        });
        assert!(at.starts_with("tests/cute_ext.rs:"), "the panic of {what} is reported at {at}: {msg}");
    }
}

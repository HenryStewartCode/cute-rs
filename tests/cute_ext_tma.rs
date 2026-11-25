//! cute_ext's TMA operations with automatic descriptors, TMAStorePipeline and the device
//! workspace (roadmap part m11-cute-ext-tma), simulated against plain Rust: a TMA load ->
//! shared memory -> TMA store round trip over a tensor with ragged edges, a multicast load in
//! a 2-CTA cluster, reduce stores (add, max), a TMAStorePipeline epilogue (2 stages, 4 writer
//! warps and a TMA warp), a pointer-array style kernel whose descriptors the kernel builds in
//! its device workspace, and a generic workspace user (per-CTA scratch of a two-pass
//! reduction, also run on the GPU with the `cuda` feature). The kernels are compiled to MLIR
//! and PTX for sm_90a (sm_100a for the 2SM load) and assembled when `ptxas` is installed.
//! Diagnostics last.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::cpasync::ReductionKind;
use cute::experimental::{self, OperationType, TMAStorePipeline, TmaLoadOptions, WorkspaceSize};
use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// One mbarrier for a single load, initialised by thread 0 and published to the CTA.
fn one_mbarrier() -> Ptr<u64> {
    let mbar = alloc_smem::<u64>(1, 8);
    if_(thread_idx().x.eq(0), || mbarrier_init(&mbar, 1));
    mbarrier_init_fence();
    sync_threads();
    mbar
}

/// Warp 0 loads `src` into `dst` (with the default expect_tx) and arrives; every thread waits.
fn load_and_wait<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, mbar: &Ptr<u64>) {
    if_(warp_idx().eq(0), || {
        experimental::tma_load(src, dst, mbar);
        if_(elect_one(), || mbarrier_arrive(mbar));
    });
    mbarrier_wait(mbar, 0);
}

const RT_ROWS: usize = 100;
const RT_COLS: usize = 200;

fn rt_value(m: usize, n: usize) -> f32 {
    ((m * 7 + n * 3) % 61) as f32 - 30.0
}

/// A `(100, 200)` f16 tensor in 64x64 tiles, one CTA per tile: warp 0 TMA-loads the tile into
/// a 128B-swizzled buffer, every thread copies the buffer (zero-filled past the edges) to
/// `tiles`, a `(128, 256)` image, and warp 0 TMA-stores the buffer to `b` (clipped).
fn round_trip_kernel() -> Kernel {
    Kernel::builder("cute_ext_tma_round_trip").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((RT_ROWS, RT_COLS), (RT_COLS, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((RT_ROWS, RT_COLS), (RT_COLS, 1)), 16);
        let tiles = param_tensor::<F16>("tiles", Layout::new((128, 256), (256, 1)));
        let layout = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64));
        let s = experimental::allocate::<F16>(AddressSpace::Smem, layout, 1024);
        let (bx, by) = (block_idx().x, block_idx().y);
        let mbar = one_mbarrier();
        load_and_wait(&local_tile(&a, (64, 64), (bx, by)), &s, &mbar);
        let t = thread_idx().x;
        for_range(0, 32, 1, |k| {
            let idx = k * 128 + t;
            let (m, n) = (idx / 64, idx % 64);
            tiles.store((bx * 64 + m, by * 64 + n), s.load((m, n)));
        });
        if_(warp_idx().eq(0), || {
            experimental::tma_store(&s, &local_tile(&b, (64, 64), (bx, by)));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    })
}

#[test]
fn round_trip_with_ragged_edges() {
    let kernel = round_trip_kernel();
    // One hidden descriptor for the load and one for the store.
    assert_eq!(kernel.ir().tensor_maps.len(), 2);
    let a: Vec<F16> = (0..RT_ROWS * RT_COLS).map(|i| F16::from_f32(rt_value(i / RT_COLS, i % RT_COLS))).collect();
    let mut args = vec![
        KernelArg::buffer(&a),
        KernelArg::buffer(&vec![F16::from_f32(99.0); RT_ROWS * RT_COLS]),
        KernelArg::buffer(&vec![F16::from_f32(-1.0); 128 * 256]),
    ];
    kernel.simulate(LaunchConfig::new([2, 4, 1], [128, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<F16>(), a);
    let tiles = args[2].to_vec::<F16>();
    for m in 0..128 {
        for n in 0..256 {
            let want = if m < RT_ROWS && n < RT_COLS { rt_value(m, n) } else { 0.0 };
            assert_eq!(tiles[m * 256 + n].to_f32(), want, "({m}, {n})");
        }
    }
}

/// Two CTAs of a cluster share one 64x64 tile of `a` with an SM90 multicast along mode 2 of
/// the `(1, 1, 2, 1)` cluster layout: each loads half and delivers it to both; each copies its
/// buffer to its own slice of `out`.
fn multicast_kernel() -> Kernel {
    Kernel::builder("cute_ext_tma_multicast").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((2, 64, 64), (4096, 64, 1)));
        let layout = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64));
        let s = experimental::allocate::<F16>(AddressSpace::Smem, layout, 1024);
        let mbar = alloc_smem::<u64>(1, 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();
        let vmnk = Layout::col_major((1, 1, 2, 1));
        let opts = TmaLoadOptions::new()
            .tma_operation_type(OperationType::Sm90TmaLoadMulticast)
            .vmnk_layout(&vmnk)
            .multicast_mode(2);
        if_(warp_idx().eq(0), || {
            experimental::tma_load_with(&local_tile(&a, (64, 64), (0, 0)), &s, &mbar, &opts);
            if_(elect_one(), || mbarrier_arrive(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        let (r, t) = (block_idx_in_cluster(), thread_idx().x);
        for_range(0, 32, 1, |k| {
            let idx = k * 128 + t;
            out.store((r, idx / 64, idx % 64), s.load((idx / 64, idx % 64)));
        });
        // No CTA exits while its peer may still write to its shared memory.
        cluster_arrive();
        cluster_wait();
    })
}

#[test]
fn multicast_load_in_a_2cta_cluster() {
    let kernel = multicast_kernel();
    let a: Vec<F16> = (0..4096).map(|i| F16::from_f32(rt_value(i / 64, i % 64))).collect();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![F16::from_f32(0.0); 8192])];
    let cfg = LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1]);
    kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let out = args[1].to_vec::<F16>();
    assert_eq!(&out[..4096], a.as_slice());
    assert_eq!(&out[4096..], a.as_slice());
}

const RED_M: usize = 48;
const RED_N: usize = 40;

fn red_init(m: usize, n: usize) -> i32 {
    ((m * 5 + n) % 23) as i32 - 7
}

fn red_operand(z: usize, m: usize, n: usize) -> i32 {
    ((m * 3 + n * 11 + z * 7) % 29) as i32 - 9
}

/// A `(48, 40)` i32 tensor in 32x32 tiles (ragged), reduced into by two CTAs per tile
/// (`blockIdx.z`): each writes its operand into a buffer and reduce-stores it with `kind`.
fn reduce_kernel(kind: ReductionKind) -> Kernel {
    Kernel::builder(format!("cute_ext_tma_reduce_{kind:?}")).block_dim([128, 1, 1]).build(move || {
        let acc = param_tensor_aligned::<i32>("acc", Layout::new((RED_M, RED_N), (RED_N, 1)), 16);
        let s = experimental::allocate::<i32>(AddressSpace::Smem, Layout::new((32, 32), (32, 1)), 128);
        let (bx, by, bz) = (block_idx().x, block_idx().y, block_idx().z);
        let t = thread_idx().x;
        for_range(0, 8, 1, |k| {
            let idx = k * 128 + t;
            let (m, n) = (idx / 32, idx % 32);
            let (gm, gn) = (bx * 32 + m, by * 32 + n);
            s.store((m, n), (gm * 3 + gn * 11 + bz * 7) % 29 - 9);
        });
        fence_view_async_shared();
        sync_threads();
        if_(warp_idx().eq(0), || {
            experimental::tma_reduce_store(&s, &local_tile(&acc, (32, 32), (bx, by)), kind);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    })
}

#[test]
fn reduce_stores_add_and_max() {
    for kind in [ReductionKind::ADD, ReductionKind::MAX] {
        let kernel = reduce_kernel(kind);
        let init: Vec<i32> = (0..RED_M * RED_N).map(|i| red_init(i / RED_N, i % RED_N)).collect();
        let mut args = vec![KernelArg::buffer(&init)];
        kernel.simulate(LaunchConfig::new([2, 2, 2], [128, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
        let got = args[0].to_vec::<i32>();
        for m in 0..RED_M {
            for n in 0..RED_N {
                let (x, y, z) = (red_init(m, n), red_operand(0, m, n), red_operand(1, m, n));
                let want = match kind {
                    ReductionKind::ADD => x + y + z,
                    _ => x.max(y).max(z),
                };
                assert_eq!(got[m * RED_N + n], want, "{kind:?} ({m}, {n})");
            }
        }
    }
}

const EPI_M: usize = 96;
const EPI_N: usize = 160;

fn epi_value(m: usize, n: usize) -> f32 {
    (m * 1000 + n) as f32
}

/// An epilogue over a `(96, 160)` f32 output in 32x32 tiles (3 x 5) through a TMAStorePipeline
/// of 2 stages: warps 0-3 write each tile into the pipeline's stage, warp 4 TMA-stores it; the
/// 160 threads meet on named barrier 1. The pipeline is carried through the tile loop.
fn epilogue_kernel(arv_count: usize) -> Kernel {
    Kernel::builder("cute_ext_tma_store_pipeline").block_dim([160, 1, 1]).build(move || {
        let out = param_tensor_aligned::<f32>("out", Layout::new((EPI_M, EPI_N), (EPI_N, 1)), 16);
        let s = experimental::allocate::<f32>(AddressSpace::Smem, Layout::new((32, 32, 2), (32, 1, 1024)), 128);
        let t = thread_idx().x;
        let pipe = TMAStorePipeline::new(2, arv_count, 1, 4);
        let tiles_n = (EPI_N / 32) as i32;
        let pipe = for_range_carry(0, (EPI_M / 32 * EPI_N / 32) as i32, 1, pipe, |tile, mut pipe| {
            let (tm, tn) = (tile / tiles_n, tile % tiles_n);
            let stage = s.slice((.., .., pipe.get_index()));
            pipe.acquire_sync();
            if_(warp_idx().lt(4), || {
                for_range(0, 8, 1, |k| {
                    let idx = k * 128 + t;
                    let (m, n) = (idx / 32, idx % 32);
                    stage.store((m, n), ((tm * 32 + m) * 1000 + tn * 32 + n).to::<f32>());
                });
            });
            pipe.commit_sync();
            if_(warp_idx().eq(4), || experimental::tma_store(&stage, &local_tile(&out, (32, 32), (tm, tn))));
            pipe.release_advance();
            pipe
        });
        pipe.tail();
    })
}

#[test]
fn tma_store_pipeline_epilogue() {
    let kernel = epilogue_kernel(160);
    let mut args = vec![KernelArg::f32(vec![-1.0; EPI_M * EPI_N])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [160, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let got = args[0].as_f32();
    for m in 0..EPI_M {
        for n in 0..EPI_N {
            assert_eq!(got[m * EPI_N + n], epi_value(m, n), "({m}, {n})");
        }
    }
}

/// Two groups of f16 matrices in one buffer, `meta` holding each one's `(rows, cols,
/// element offset)`: CTA `g` makes a tensor of group `g` on a pointer computed in the kernel
/// (so its descriptor cannot be encoded on the host) and loads its tiles (0, 0) and (1, 1)
/// (ragged) into two buffers, then copies both to `out[g]`. The descriptor is built in the
/// device workspace.
const GROUPS: [(usize, usize, usize); 2] = [(40, 64, 0), (72, 48, 40 * 64)];

fn ptr_array_kernel() -> Kernel {
    Kernel::builder("cute_ext_tma_ptr_array").block_dim([128, 1, 1]).build(|| {
        let data = param_ptr_aligned::<F16>("data", 16);
        let meta = param_tensor::<i32>("meta", Layout::new((2, 3), (3, 1)));
        let out = param_tensor::<F16>("out", Layout::new((2, 2, 32, 32), (2048, 1024, 32, 1)));
        let g = block_idx().x;
        let (rows, cols, off) = (meta.load((g, 0)), meta.load((g, 1)), meta.load((g, 2)));
        let address = data.toint_i64() + off.to::<i64>() * 2;
        let base = make_ptr::<F16>(address, AddressSpace::Gmem, Some(16));
        // A tensor on a raw pointer is made a predication origin explicitly.
        let tensor = experimental::predicated_tensor_origin(&make_tensor(
            base,
            Layout::new((IndexExpr::from(rows), IndexExpr::from(cols)), (IndexExpr::from(cols), 1)),
        ));
        let s0 = experimental::allocate::<F16>(AddressSpace::Smem, Layout::new((32, 32), (32, 1)), 128);
        let s1 = experimental::allocate::<F16>(AddressSpace::Smem, Layout::new((32, 32), (32, 1)), 128);
        let mbar = alloc_smem::<u64>(2, 8);
        if_(thread_idx().x.eq(0), || {
            mbarrier_init(&mbar, 1);
            mbarrier_init(&mbar.offset(1), 1);
        });
        mbarrier_init_fence();
        sync_threads();
        if_(warp_idx().eq(0), || {
            experimental::tma_load(&local_tile(&tensor, (32, 32), (0, 0)), &s0, &mbar);
            if_(elect_one(), || mbarrier_arrive(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        if_(warp_idx().eq(0), || {
            experimental::tma_load(&local_tile(&tensor, (32, 32), (1, 1)), &s1, &mbar.offset(1));
            if_(elect_one(), || mbarrier_arrive(&mbar.offset(1)));
        });
        mbarrier_wait(&mbar.offset(1), 0);
        let t = thread_idx().x;
        for_range(0, 8, 1, |k| {
            let idx = k * 128 + t;
            let (m, n) = (idx / 32, idx % 32);
            out.store((g, 0, m, n), s0.load((m, n)));
            out.store((g, 1, m, n), s1.load((m, n)));
        });
    })
}

#[test]
fn ptr_array_descriptors_in_the_workspace() {
    let kernel = ptr_array_kernel();
    let total: usize = GROUPS.iter().map(|&(r, c, _)| r * c).sum();
    let data: Vec<F16> = (0..total).map(|i| F16::from_f32((i % 1999) as f32 - 999.0)).collect();
    let meta: Vec<i32> = GROUPS.iter().flat_map(|&(r, c, o)| [r as i32, c as i32, o as i32]).collect();
    let cfg = LaunchConfig::new([2, 1, 1], [128, 1, 1]);
    let mut args =
        vec![KernelArg::buffer(&data), KernelArg::buffer(&meta), KernelArg::buffer(&vec![F16::from_f32(-1.0); 4096])];
    // One 128-byte descriptor per CTA.
    let req = kernel.device_workspace_requirement(cfg, &args).unwrap();
    assert_eq!(req, experimental::AllocationRequirement { size_in_bytes: 256, alignment: 128 });
    args.push(KernelArg::buffer(&vec![0u8; req.size_in_bytes as usize]));
    kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let out = args[2].to_vec::<F16>();
    for (g, &(rows, cols, off)) in GROUPS.iter().enumerate() {
        for (tile, (r0, c0)) in [(0, 0), (32, 32)].into_iter().enumerate() {
            for m in 0..32 {
                for n in 0..32 {
                    let (r, c) = (r0 + m, c0 + n);
                    let want = if r < rows && c < cols { data[off + r * cols + c].to_f32() } else { 0.0 };
                    assert_eq!(
                        out[g * 2048 + tile * 1024 + m * 32 + n].to_f32(),
                        want,
                        "group {g} tile {tile} ({m}, {n})"
                    );
                }
            }
        }
    }
}

/// A two-pass sum per CTA through per-CTA scratch in the device workspace: each thread writes
/// the sum of its two elements to the CTA's slot, then thread 0 sums the slot.
fn scratch_kernel() -> Kernel {
    Kernel::builder("cute_ext_workspace_scratch").block_dim([128, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(4 * 256));
        let out = param_tensor::<f32>("out", Layout::col_major(4));
        let slot = experimental::device_workspace_cta_slot(128 * 4, 16);
        let scratch = make_tensor(recast_ptr::<f32, u8>(&slot, None), Layout::col_major(128));
        let (b, t) = (block_idx().x, thread_idx().x);
        scratch.store(t, x.load(b * 256 + t) + x.load(b * 256 + 128 + t));
        sync_threads();
        if_(t.eq(0), || {
            let sum = for_range_carry(0, 128, 1, Float32::from(0.0f32), |i, acc| acc + scratch.load(i));
            out.store(b, sum);
        });
    })
}

fn scratch_inputs() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..1024).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
    let want = (0..4).map(|b| x[b * 256..(b + 1) * 256].iter().sum()).collect();
    (x, want)
}

#[test]
fn per_cta_scratch_two_pass_reduction() {
    let kernel = scratch_kernel();
    let (x, want) = scratch_inputs();
    let cfg = LaunchConfig::new([4, 1, 1], [128, 1, 1]);
    let mut args = vec![KernelArg::f32(x), KernelArg::f32(vec![0.0; 4])];
    let req = experimental::QueryDeviceWorkspaceFunc.call(&kernel, cfg, &args).unwrap();
    assert_eq!(req, experimental::AllocationRequirement { size_in_bytes: 4 * 512, alignment: 16 });
    args.push(KernelArg::buffer(&vec![0u8; req.size_in_bytes as usize]));
    kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].as_f32(), want.as_slice());
}

#[cfg(feature = "cuda")]
#[test]
fn per_cta_scratch_two_pass_reduction_on_gpu() {
    use cute::cuda::CudaContext;
    let Ok(ctx) = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")) else { return };
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let kernel = scratch_kernel();
    let (x, want) = scratch_inputs();
    let cfg = LaunchConfig::new([4, 1, 1], [128, 1, 1]);
    let mut args = vec![KernelArg::f32(x), KernelArg::f32(vec![0.0; 4])];
    let req = kernel.device_workspace_requirement(cfg, &args).unwrap();
    args.push(KernelArg::buffer(&vec![0u8; req.size_in_bytes as usize]));
    let arch = ctx.best_target(&tc).expect("a target");
    let compiled = kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    let f = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
    ctx.launch(&kernel, &f, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].as_f32(), want.as_slice());
    // A workspace one byte short is refused before the launch.
    let mut short = args.clone();
    short[2] = KernelArg::buffer(&vec![0u8; req.size_in_bytes as usize - 1]);
    let e = ctx.launch(&kernel, &f, cfg, &mut short).unwrap_err().to_string();
    assert!(e.contains("device workspace"), "{e}");
}

/// A 2SM load (`SM100_TMA_LOAD_2SM`) of one tile, for sm_100a's assembler.
fn two_sm_kernel() -> Kernel {
    Kernel::builder("cute_ext_tma_2sm").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((128, 64), (64, 1)), 16);
        let layout = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64));
        let s = experimental::allocate::<F16>(AddressSpace::Smem, layout, 1024);
        let mbar = one_mbarrier();
        let opts = TmaLoadOptions::new().tma_operation_type(OperationType::Sm100TmaLoad2Sm);
        if_(warp_idx().eq(0), || {
            experimental::tma_load_with(&local_tile(&a, (64, 64), (block_idx().x, 0)), &s, &mbar, &opts);
            if_(elect_one(), || mbarrier_arrive(&mbar));
        });
        mbarrier_wait(&mbar, 0);
    })
}

/// Compile `kernel` for `arch` and assemble it when `ptxas` is installed; returns the PTX.
fn assemble(kernel: &Kernel, tc: &Toolchain, arch: Arch, flag: &str) -> String {
    let ptx = kernel.compile_ptx(tc, arch).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let ptxas = std::env::var("CUTE_PTXAS").unwrap_or_else(|_| "ptxas".into());
    let dir = std::env::temp_dir().join(format!("cute_ext_tma_{}_{}", kernel.name(), std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ptx");
    std::fs::write(&file, &ptx).unwrap();
    match std::process::Command::new(&ptxas)
        .arg(format!("-arch={flag}"))
        .arg(&file)
        .arg("-o")
        .arg(dir.join("k.cubin"))
        .output()
    {
        Ok(out) => assert!(out.status.success(), "{}: ptxas: {}", kernel.name(), String::from_utf8_lossy(&out.stderr)),
        Err(_) => eprintln!("skipping the assembly: no ptxas found"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    ptx
}

#[test]
fn kernels_compile_for_sm90a_and_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = assemble(&round_trip_kernel(), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains("cp.async.bulk.tensor.2d.shared::cluster.global"), "{ptx}");
    assert!(ptx.contains("cp.async.bulk.tensor.2d.global.shared::cta"), "{ptx}");
    assert!(ptx.contains("mbarrier.expect_tx"), "{ptx}");
    let ptx = assemble(&multicast_kernel(), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains(".multicast::cluster"), "{ptx}");
    let ptx = assemble(&reduce_kernel(ReductionKind::ADD), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile"), "{ptx}");
    let ptx = assemble(&epilogue_kernel(160), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains("cp.async.bulk.wait_group.read \t1"), "{ptx}");
    assert!(ptx.contains("fence.proxy.async.shared::cta"), "{ptx}");
    let ptx = assemble(&ptr_array_kernel(), &tc, Arch::Sm90a, "sm_90a");
    assert!(ptx.contains("tensormap.replace"), "{ptx}");
    assert!(ptx.contains("fence.proxy.tensormap::generic"), "{ptx}");
    let ptx = assemble(&two_sm_kernel(), &tc, Arch::Sm100a, "sm_100a");
    assert!(ptx.contains("cta_group::2"), "{ptx}");
}

#[test]
fn diagnostics() {
    let tile_kernel = |opts: TmaLoadOptions| {
        Kernel::builder("diag").block_dim([128, 1, 1]).build(move || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
            let s = experimental::allocate::<F16>(AddressSpace::Smem, Layout::new((64, 64), (64, 1)), 128);
            let mbar = one_mbarrier();
            experimental::tma_load_with(&local_tile(&a, (64, 64), (0, 0)), &s, &mbar, &opts);
        })
    };
    // Multicast without its layout or mode.
    let m = build_panic(|| {
        tile_kernel(TmaLoadOptions::new().tma_operation_type(OperationType::Sm90TmaLoadMulticast).multicast_mode(2));
    });
    assert!(m.contains("vmnk_layout") && m.contains("multicast_mode"), "{m}");
    assert!(m.contains("cute_ext_tma.rs"), "the location is the caller's: {m}");
    // Multicast keywords without an operation type.
    let m = build_panic(|| {
        tile_kernel(TmaLoadOptions::new().vmnk_layout(&Layout::col_major((1, 1, 2, 1))).multicast_mode(2));
    });
    assert!(m.contains("tma_operation_type"), "{m}");

    // update_expect_tx double-counting under a pipeline-style arming of the mbarrier.
    let k = Kernel::builder("double_tx").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
        let s = experimental::allocate::<F16>(AddressSpace::Smem, Layout::new((64, 64), (64, 1)), 128);
        let mbar = one_mbarrier();
        if_(warp_idx().eq(0), || {
            // The pipeline's producer_acquire arms the bytes...
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 64 * 64 * 2));
            // ...and the load counts them again.
            experimental::tma_load(&local_tile(&a, (64, 64), (0, 0)), &s, &mbar);
        });
        mbarrier_wait(&mbar, 0);
    });
    let mut args = vec![KernelArg::buffer(&vec![F16::from_f32(1.0); 4096])];
    let e = k.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap_err().to_string();
    assert!(e.contains("deadlock") || e.contains("hang"), "{e}");
    assert!(e.contains("byte"), "the report names the transaction bytes still owed: {e}");

    // A missing or too-small workspace.
    let kernel = scratch_kernel();
    let cfg = LaunchConfig::new([4, 1, 1], [128, 1, 1]);
    let base = vec![KernelArg::f32(vec![0.0; 1024]), KernelArg::f32(vec![0.0; 4])];
    let e = kernel.simulate(cfg, &mut base.clone()).unwrap_err().to_string();
    assert!(e.contains("device workspace") && e.contains("missing"), "{e}");
    let mut short = base.clone();
    short.push(KernelArg::buffer(&vec![0u8; 4 * 512 - 4]));
    let e = kernel.simulate(cfg, &mut short).unwrap_err().to_string();
    assert!(e.contains("2044 bytes") && e.contains("2048"), "{e}");
    // A PerScalar count that is not a scalar parameter.
    let m = build_panic(|| {
        Kernel::build("bad_ws", || {
            experimental::request_device_workspace(WorkspaceSize::PerScalar { count: thread_idx().x, bytes: 4 }, 4);
        });
    });
    assert!(m.contains("scalar parameter"), "{m}");

    // A TMAStorePipeline whose barrier counts 4 warps while 5 take part.
    let k = epilogue_kernel(128);
    let mut args = vec![KernelArg::f32(vec![0.0; EPI_M * EPI_N])];
    let e = k.simulate(LaunchConfig::new([1, 1, 1], [160, 1, 1]), &mut args).unwrap_err().to_string();
    assert!(e.contains("barrier 1"), "{e}");
}

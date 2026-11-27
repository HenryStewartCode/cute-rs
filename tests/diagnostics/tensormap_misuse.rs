//! Tensormap descriptor misuse diagnostics.

use super::*;
use cute::cpasync::{self, TensormapField};

#[derive(Clone, Copy, PartialEq)]
enum Bug {
    NoAcquire,
    NoRelease,
    AcquireFirst,
    NotCopied,
    ReplaceUncopied,
    DivergentCpFence,
    CpFenceNoAcquire,
}

/// One warp copies the TMA atom's descriptor to `desc` (in global memory, or through
/// shared memory with `tensormap.cp_fenceproxy`), points it at `b`, releases,
/// acquires and loads a 16x32 box through it, except for `bug`.
fn kernel(bug: Bug) -> Kernel {
    Kernel::builder("tensormap_misuse").block_dim([32, 1, 1]).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((64, 64), (64, 1)), 16);
        let desc = param_tensor_aligned::<u64>("desc", Layout::col_major(16), 128);
        let smem_layout = Layout::new((16, 32), (32, 1));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma) = cpasync::make_tiled_tma_atom(g2s, &a, smem_layout.clone(), (16, 32), 1).into();
        let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        if_(elect_one(), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let d = desc.ptr();
        if matches!(bug, Bug::DivergentCpFence | Bug::CpFenceNoAcquire) {
            let sd = alloc_smem::<u64>(16, 128);
            if_(elect_one(), || {
                cpasync::copy_tensormap(&atom, &sd);
                cpasync::tensormap_replace(&sd, TensormapField::GlobalAddress, b.ptr().toint_i64());
            });
            sync_warp();
            if bug == Bug::DivergentCpFence {
                if_(elect_one(), || cpasync::cp_fence_tma_desc_release(&d, &sd));
            } else {
                cpasync::cp_fence_tma_desc_release(&d, &sd);
            }
        } else if bug != Bug::NotCopied {
            if_(elect_one(), || {
                if bug != Bug::ReplaceUncopied {
                    cpasync::copy_tensormap(&atom, &d);
                }
                cpasync::tensormap_replace(&d, TensormapField::GlobalAddress, b.ptr().toint_i64());
                if bug == Bug::AcquireFirst {
                    cpasync::fence_tma_desc_acquire(&d);
                }
                if bug != Bug::NoRelease {
                    cpasync::fence_tma_desc_release();
                }
            });
        }
        if !matches!(bug, Bug::NoAcquire | Bug::AcquireFirst | Bug::CpFenceNoAcquire) {
            cpasync::fence_tma_desc_acquire(&d);
        }
        let g = local_tile(&tma, (16, 32), (0, 0));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 16 * 32 * 2));
        copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).tma_desc_ptr(d.toint_i64()));
        mbarrier_wait(&mbar, 0);
    })
}

fn run(bug: Bug) -> SimError {
    let mut args = vec![
        KernelArg::buffer(&vec![F16::from_f32(1.0); 64 * 64]),
        KernelArg::buffer(&vec![F16::from_f32(2.0); 64 * 64]),
        KernelArg::buffer(&[0u64; 16]),
    ];
    simulate_err(&kernel(bug), 32, &mut args)
}

#[test]
fn tma_through_an_updated_descriptor_needs_the_acquire() {
    let e = run(Bug::NoAcquire);
    assert!(
        e.message.contains("cp.async.bulk.tensor")
            && e.message.contains("this thread has not acquired it since warp 0 wrote and released it")
            && e.message.contains("fence_tma_desc_acquire(ptr)"),
        "{e}"
    );
}

#[test]
fn tma_through_an_updated_descriptor_needs_the_release() {
    let e = run(Bug::NoRelease);
    assert!(
        e.message.contains("warp 0 wrote it through the generic proxy without releasing it since")
            && e.message.contains("fence_tma_desc_release()"),
        "{e}"
    );
}

#[test]
fn tma_through_a_descriptor_published_from_shared_memory_needs_the_acquire() {
    let e = run(Bug::CpFenceNoAcquire);
    assert!(
        e.message.contains("cp.async.bulk.tensor")
            && e.message.contains("this thread has not acquired it since warp 0 wrote and released it"),
        "{e}"
    );
}

/// A TMA store through a descriptor in global memory, then a rewrite of that
/// descriptor by the issuing lane, with or without first waiting for the store's
/// bulk group to read its source.
fn rewrite_after_store(wait: bool) -> Kernel {
    Kernel::builder("tensormap_rewrite_after_store").block_dim([32, 1, 1]).build(move || {
        let inp = param_tensor_aligned::<F16>("inp", Layout::new((16, 32), (32, 1)), 16);
        let o = param_tensor_aligned::<F16>("o", Layout::new((64, 64), (64, 1)), 16);
        let o2 = param_tensor_aligned::<F16>("o2", Layout::new((64, 64), (64, 1)), 16);
        let desc = param_tensor_aligned::<u64>("desc", Layout::col_major(16), 128);
        let smem_layout = Layout::new((16, 32), (32, 1));
        let (atom, tma) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &o, smem_layout.clone(), (16, 32), 1).into();
        let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
        let t = thread_idx().x;
        for_range(0, 16, 1, |j| {
            let idx = j * 32 + t;
            let (m, k) = (idx / 32, idx % 32);
            s.store((m, k), inp.load((m, k)));
        });
        fence_view_async_shared();
        sync_threads();
        let d = desc.ptr();
        if_(elect_one(), || {
            cpasync::copy_tensormap(&atom, &d);
            cpasync::fence_tma_desc_release();
        });
        cpasync::fence_tma_desc_acquire(&d);
        let g = local_tile(&tma, (16, 32), (0, 0));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        copy_with(&atom, &ts, &tg, CopyArgs::new().tma_desc_ptr(d.toint_i64()));
        if_(elect_one(), || {
            cp_async_bulk_commit_group();
            if wait {
                cp_async_bulk_wait_group(0, true);
            }
            cpasync::tensormap_replace(&d, TensormapField::GlobalAddress, o2.ptr().toint_i64());
            cpasync::fence_tma_desc_release();
            cp_async_bulk_wait_group(0, false);
        });
    })
}

#[test]
fn rewriting_a_descriptor_a_tma_store_still_reads() {
    let args = || {
        vec![
            KernelArg::buffer(&vec![F16::from_f32(1.0); 16 * 32]),
            KernelArg::buffer(&vec![F16::from_f32(0.0); 64 * 64]),
            KernelArg::buffer(&vec![F16::from_f32(0.0); 64 * 64]),
            KernelArg::buffer(&[0u64; 16]),
        ]
    };
    let e = simulate_err(&rewrite_after_store(false), 32, &mut args());
    assert!(
        e.message.contains("a generic write to the tensor map at byte 0 of `desc`, which a TMA store")
            && e.message.contains("cp_async_bulk_wait_group(0, true)"),
        "{e}"
    );
    let mut ok = args();
    rewrite_after_store(true)
        .simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut ok)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(ok[1].to_vec::<F16>()[..32].iter().all(|x| x.to_f32() == 1.0));
}

#[test]
fn the_acquire_must_follow_the_release() {
    let e = run(Bug::AcquireFirst);
    assert!(e.message.contains("is not ordered after that release"), "{e}");
}

#[test]
fn tma_desc_ptr_at_memory_without_a_descriptor() {
    let e = run(Bug::NotCopied);
    assert!(
        e.message.contains("do not hold a tensor map (uninitialised memory") && e.message.contains("copy_tensormap"),
        "{e}"
    );
    let e = run(Bug::ReplaceUncopied);
    assert!(e.message.starts_with("tensormap.replace") && e.message.contains("do not hold a tensor map"), "{e}");
}

#[test]
fn cp_fenceproxy_is_warp_collective() {
    let e = run(Bug::DivergentCpFence);
    assert!(e.message.contains("tensormap.cp_fenceproxy") && e.message.contains("1 of the 32 lanes"), "{e}");
}

#[test]
fn tensormap_replace_is_refused_for_plain_sm90() {
    let k = kernel(Bug::NoAcquire);
    let e = cute::validate::check_target(k.ir(), Arch::Sm90).unwrap_err();
    assert!(
        e.message.contains("tensormap.replace (update_tma_descriptor, tensormap_replace) needs sm_90a")
            && e.message.contains("being compiled for sm_90"),
        "{e:?}"
    );
    assert!(e.loc.is_some_and(|l| l.file().ends_with("tensormap_misuse.rs")), "{e:?}");
    assert!(cute::validate::check_target(k.ir(), Arch::Sm90a).is_ok());
}

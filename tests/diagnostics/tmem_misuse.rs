//! Tensor-memory allocation misuse diagnostics.

use super::*;

/// Simulate `body` in one CTA of 64 threads (or a cluster of `cluster` CTAs) and return
/// the simulator's report.
fn sim_in(cluster: u32, body: impl FnOnce()) -> SimError {
    let kernel = Kernel::builder("tmem_misuse").block_dim([64, 1, 1]).build(body);
    let cfg = LaunchConfig::new([cluster, 1, 1], [64, 1, 1]).with_cluster([cluster, 1, 1]);
    match kernel.simulate(cfg, &mut Vec::new()) {
        Ok(_) => panic!("expected the simulator to report the misuse"),
        Err(e) => e,
    }
}

fn sim(body: impl FnOnce()) -> String {
    sim_in(1, body).message
}

fn warp0(f: impl FnOnce()) {
    if_(warp_idx().eq(0), f);
}

const A: Arch = Arch::Sm100a;

#[test]
fn leak_at_exit() {
    let e = sim(|| warp0(|| alloc_tmem(64, &alloc_smem::<i32>(1, 4), None, A)));
    assert!(e.contains("exits with tensor memory still allocated") && e.contains("columns 0 to 63"), "{e}");
}

#[test]
fn double_free() {
    let e = sim(|| {
        warp0(|| {
            let slot = alloc_smem::<i32>(1, 4);
            alloc_tmem(32, &slot, None, A);
            let p = retrieve_tmem_ptr::<f32>(4, &slot);
            dealloc_tmem(&p, 32, None, A);
            dealloc_tmem(&p, 32, None, A);
        })
    });
    assert!(e.contains("not the start of a live allocation") && e.contains("freed twice"), "{e}");
}

#[test]
fn mismatched_free() {
    let e = sim(|| {
        warp0(|| {
            let slot = alloc_smem::<i32>(1, 4);
            alloc_tmem(64, &slot, None, A);
            dealloc_tmem(&retrieve_tmem_ptr::<f32>(4, &slot), 32, None, A);
        })
    });
    assert!(e.contains("32 columns, but the allocation has 64"), "{e}");
}

#[test]
fn non_collective_issue() {
    let e = sim(|| {
        let slot = alloc_smem::<i32>(1, 4);
        if_(thread_idx().x.lt(16), || alloc_tmem(32, &slot, None, A));
    });
    assert!(e.contains("tcgen05.alloc executed by 16 of the 32 lanes"), "{e}");
}

#[test]
fn non_uniform_operands() {
    let e = sim(|| {
        warp0(|| {
            let slot = alloc_smem::<i32>(1, 4);
            let n = select(thread_idx().x.lt(4), Int32::from(64), Int32::from(32));
            alloc_tmem(n, &slot, None, A);
        })
    });
    assert!(e.contains("every lane of the warp gives the same operands"), "{e}");
}

#[test]
fn mixed_cta_group() {
    let e = sim_in(2, || {
        warp0(|| {
            relinquish_tmem_alloc_permit(Some(true));
            alloc_tmem(32, &alloc_smem::<i32>(1, 4), None, A);
        })
    })
    .message;
    assert!(e.contains("all tcgen05 instructions of a kernel use the same cta_group"), "{e}");
}

#[test]
fn allocation_beyond_capacity_is_a_deadlock() {
    let e = sim(|| {
        warp0(|| {
            let (a, b) = (alloc_smem::<i32>(1, 4), alloc_smem::<i32>(1, 4));
            alloc_tmem(512, &a, None, A);
            alloc_tmem(32, &b, None, A);
        })
    });
    assert!(e.contains("deadlock") && e.contains("columns 0 to 511") && e.contains("hang"), "{e}");
}

#[test]
fn alloc_after_relinquish() {
    let e = sim(|| {
        warp0(|| {
            relinquish_tmem_alloc_permit(None);
            alloc_tmem(32, &alloc_smem::<i32>(1, 4), None, A);
        })
    });
    assert!(e.contains("relinquished its allocation permit"), "{e}");
}

#[test]
fn retrieve_of_a_freed_allocation() {
    let e = sim(|| {
        warp0(|| {
            let slot = alloc_smem::<i32>(1, 4);
            alloc_tmem(32, &slot, None, A);
            dealloc_tmem(&retrieve_tmem_ptr::<f32>(4, &slot), 32, None, A);
            let _ = retrieve_tmem_ptr::<f32>(4, &slot);
        })
    });
    assert!(e.contains("not in a live TMEM allocation"), "{e}");
}

/// Warp 1 reads the address without a barrier after warp 0's allocation wrote it.
#[test]
fn retrieve_without_a_barrier_races() {
    let e = sim(|| {
        let slot = alloc_smem::<i32>(1, 4);
        warp0(|| alloc_tmem(32, &slot, None, A));
        if_(warp_idx().eq(1), || {
            let _ = retrieve_tmem_ptr::<f32>(4, &slot);
        });
    });
    assert!(e.contains("race"), "{e}");
}

#[test]
fn pair_allocation_needs_an_even_cluster() {
    let e = sim(|| warp0(|| alloc_tmem(32, &alloc_smem::<i32>(1, 4), Some(true), A)));
    assert!(e.contains("even number of CTAs"), "{e}");
}

#[test]
fn exclusive_allocation_freed_by_a_plain_dealloc() {
    let e = sim(|| {
        warp0(|| {
            let slot = alloc_smem::<i32>(1, 4);
            alloc_tmem(576, &slot, None, Arch::Sm107f);
            // A runtime count (64 * 9), so that only the simulator sees it.
            dealloc_tmem(&retrieve_tmem_ptr::<f32>(4, &slot), block_dim().x * 9, None, Arch::Sm100a);
        })
    });
    assert!(e.contains("freed by tcgen05.dealloc.exclusive of the same size"), "{e}");
}

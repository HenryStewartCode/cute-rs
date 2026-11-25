//! Tensor memory (roadmap part m10-tmem-core): `alloc_tmem`, `retrieve_tmem_ptr`,
//! `dealloc_tmem` and `relinquish_tmem_alloc_permit`, with TMEM tensors' address arithmetic.
//! sm_100a: simulated, compiled for sm_100a and assembled by `tests/ptxas.rs`; no GPU here
//! runs them.

use cute::prelude::*;

use super::Case;

const THREADS: usize = 128;
/// Values per thread in `out`: the address it retrieved, then (thread 0) four derived ones.
const PER_BLOCK: usize = THREADS + 4;

/// Warp 0 allocates `ncols` columns (`is_two_cta`: one warp in each CTA of a pair), a
/// `NamedBarrier` over the CTA publishes the address, every thread retrieves the pointer and
/// writes the address to `out`; thread 0 also writes the addresses of a 32-bit TMEM tensor's
/// element (lane 1, column 2), of a 16-bit one's element (lane 1, element 4: column 2), the
/// column past the 16-bit tensor (`find_tmem_tensor_col_offset`) and the pointer's `toint`
/// again. Then warp 0 relinquishes the permit and frees the columns.
fn alloc_kernel(name: &str, ncols: i32, two_cta: bool) -> Kernel {
    let two = Some(two_cta);
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).target(Arch::Sm100a).build(move || {
        let out = param_tensor::<i32>("out", Layout::col_major(2 * PER_BLOCK));
        let slot = alloc_smem::<i32>(1, 4);
        let t = thread_idx().x;
        let base = block_idx().x * PER_BLOCK as i32;
        let barrier = pipeline::NamedBarrier::new(1, THREADS as u32);
        if_(warp_idx().eq(0), || {
            alloc_tmem(ncols, &slot, two, Arch::Sm100a);
            relinquish_tmem_alloc_permit(two);
        });
        barrier.arrive_and_wait();
        let p = retrieve_tmem_ptr::<f32>(16, &slot);
        out.store(base + t, p.toint_i32());
        if_(t.eq(0), || {
            let t32 = make_tensor(p.clone(), Layout::new((128, 32), (1 << 16, 1)));
            out.store(base + THREADS as i32, t32.slice((1, 2)).ptr().toint_i32());
            let p16 = retrieve_tmem_ptr::<F16>(16, &slot);
            let t16 = make_tensor(p16, Layout::new((128, 64), (1 << 17, 1)));
            out.store(base + THREADS as i32 + 1, t16.slice((1, 4)).ptr().toint_i32());
            out.store(base + THREADS as i32 + 2, find_tmem_tensor_col_offset(&t16) as i32);
            out.store(base + THREADS as i32 + 3, slot.load());
        });
        barrier.arrive_and_wait();
        if_(warp_idx().eq(0), || dealloc_tmem(&p, ncols, two, Arch::Sm100a));
    })
}

fn check_block(out: &[i32], ncols: i32, what: &str) -> i32 {
    let addr = out[0];
    assert_eq!(addr >> 16, 0, "{what}: the allocation starts at lane 0");
    assert_eq!((addr & 0xffff) % ncols, 0, "{what}: the allocation is aligned to its size");
    assert!(out[..THREADS].iter().all(|&a| a == addr), "{what}: every thread retrieves the same address");
    assert_eq!(out[THREADS], addr + (1 << 16) + 2, "{what}: f32 element (1, 2)");
    assert_eq!(out[THREADS + 1], addr + (1 << 16) + 2, "{what}: f16 element (1, 4) is column 2");
    assert_eq!(out[THREADS + 2], 32, "{what}: 64 f16 columns span 32 TMEM columns");
    assert_eq!(out[THREADS + 3], addr, "{what}: the shared slot holds the address");
    addr
}

macro_rules! alloc_case {
    ($name:ident, $ncols:expr) => {
        pub fn $name() -> Case {
            fn check(args: &[KernelArg]) {
                check_block(&args[0].to_vec::<i32>()[..PER_BLOCK], $ncols, stringify!($name));
            }
            Case {
                name: stringify!($name),
                kernel: alloc_kernel(stringify!($name), $ncols, false),
                cfg: LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]),
                small_only: false,
                check,
            }
        }
    };
}

alloc_case!(tmem_alloc_32, 32);
alloc_case!(tmem_alloc_64, 64);
alloc_case!(tmem_alloc_512, 512);

/// The pair forms (`cta_group::2`) in a (2,1,1) cluster: both CTAs get the same columns.
pub fn tmem_alloc_pair() -> Case {
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<i32>();
        let a = check_block(&out[..PER_BLOCK], 64, "tmem_alloc_pair CTA 0");
        let b = check_block(&out[PER_BLOCK..], 64, "tmem_alloc_pair CTA 1");
        assert_eq!(a, b, "both CTAs of the pair get the same columns");
    }
    Case {
        name: "tmem_alloc_pair",
        kernel: alloc_kernel("tmem_alloc_pair", 64, true),
        cfg: LaunchConfig::new([2, 1, 1], [THREADS as u32, 1, 1]).with_cluster([2, 1, 1]),
        small_only: false,
        check,
    }
}

/// The TMEM kernels of the sm100 group.
pub fn tmem_cases() -> Vec<Case> {
    vec![tmem_alloc_32(), tmem_alloc_64(), tmem_alloc_512(), tmem_alloc_pair()]
}

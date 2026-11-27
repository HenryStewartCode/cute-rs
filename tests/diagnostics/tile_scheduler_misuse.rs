//! The static persistent tile schedulers: misuse diagnostics.

use super::build_panic;
use cute::prelude::*;
use cute::utils::{
    PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentRuntimeTileScheduler,
};

/// A cluster deeper than one CTA along K, a swizzle size of 0 and an
/// inner mode of 2 are refused.
#[test]
fn schedulers_refuse_bad_arguments() {
    let msg = build_panic(|| {
        PersistentTileSchedulerParams::new((4usize, 4usize, 1usize), [1, 1, 2], Default::default());
    });
    assert!(msg.contains("unsupported cluster_shape_k 2"), "{msg}");
    let msg = build_panic(|| {
        let options = PersistentTileSchedulerOptions { swizzle_size: 0, raster_along_m: true };
        PersistentTileSchedulerParams::new((4usize, 4usize, 1usize), [1, 1, 1], options);
    });
    assert!(msg.contains("swizzle_size must be >= 1, not 0"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("inner_mode_2").block_dim([32, 1, 1]).build(|| {
            let params = PersistentTileSchedulerParams::new((4usize, 4usize, 1usize), [1, 1, 1], Default::default());
            StaticPersistentRuntimeTileScheduler::create(&params, block_idx(), grid_dim(), 2);
        });
    });
    assert!(msg.contains("inner_mode is 0 (M first) or 1 (N first), not 2"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("runtime_grid").block_dim([32, 1, 1]).build(|| {
            let m = IndexExpr::from(param_scalar::<i32>("m"));
            let p = PersistentTileSchedulerParams::new(Shape::from((m, 4usize, 1usize)), [1, 1, 1], Default::default());
            p.get_grid_shape(4);
        });
    });
    assert!(msg.contains("get_grid_shape needs a static problem shape"), "{msg}");
}

/// A cluster of 33 CTAs (or 0) is refused by `HardwareInfo` (needs a
/// GPU; skipped without one).
#[cfg(feature = "cuda")]
#[test]
fn hardware_info_refuses_bad_cluster_sizes() {
    let Ok(info) = cute::cuda::HardwareInfo::new(0) else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    for bad in [33, 0] {
        let e = info.get_max_active_clusters(bad, None).unwrap_err();
        assert!(e.message.contains(&format!("a cluster has 1 to 32 CTAs, not {bad}")), "{e}");
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kernels for launches with a preferred and a fallback cluster shape
//! (CuTe DSL's `is_preferred_cluster`
//! and the fallback params of its tile schedulers).
//!
//! Since CUDA 12.8 (Blackwell), a launch may name two cluster shapes
//! (m6's [`LaunchConfig::with_fallback_cluster`](crate::LaunchConfig::with_fallback_cluster)):
//! the hardware runs each cluster of the grid at the preferred shape when it
//! fits, and otherwise as fallback-shaped clusters, so one launch mixes the
//! two. Everything a kernel derives from the cluster shape (the cluster
//! layout, multicast masks, pipeline arrival counts, scheduler params)
//! therefore exists twice. CuTe DSL clones the kernel body per shape with a
//! compiler pass; cute-rs builds the body twice instead, once per arm of an
//! `if` on [`is_preferred_cluster`], with the arm's cluster shape a static
//! value in each: [`mixed_cluster_arms`].
//!
//! While an arm is built, [`current_cluster_arm`] names it, and the tile
//! schedulers check their params against it (the build-time form of CuTe
//! DSL's run-time `_select_active_params`): the cluster launch control
//! scheduler takes its params' fallback schedule in the fallback arm, and a
//! schedule made for another shape is reported when the scheduler is
//! created. Kernels can make the same check for their own cluster-shape
//! math with [`check_cluster_shape_in_arm`].

use std::cell::RefCell;

use crate::arch::block_in_cluster_dim;
use crate::control::if_else;
use crate::value::Boolean;

/// One arm of a mixed-cluster kernel: the cluster shape the arm runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterArm {
    /// The cluster shape `(M, N, K)` of the clusters that run this arm.
    pub cluster_shape_mnk: [usize; 3],
    /// Whether this is the preferred shape's arm (or the only arm of a
    /// launch without a fallback).
    pub is_preferred: bool,
}

thread_local! {
    static ARMS: RefCell<Vec<ClusterArm>> = const { RefCell::new(Vec::new()) };
}

/// The arm being built, if the kernel is inside [`mixed_cluster_arms`].
pub fn current_cluster_arm() -> Option<ClusterArm> {
    ARMS.with(|a| a.borrow().last().copied())
}

/// Whether this block's cluster has the shape `preferred_shape` (CuTe DSL's
/// `cute.experimental.utils.is_preferred_cluster`): the cluster's size in
/// blocks ([`block_in_cluster_dim`], `%cluster_nctaid`) compared mode by
/// mode.
#[track_caller]
pub fn is_preferred_cluster(preferred_shape: [usize; 3]) -> Boolean {
    let d = block_in_cluster_dim();
    d.x.eq(preferred_shape[0] as i32) & d.y.eq(preferred_shape[1] as i32) & d.z.eq(preferred_shape[2] as i32)
}

/// Check a preferred and a fallback cluster shape against each other and,
/// when `grid` is given, against the launch grid (the constraints of CuTe
/// DSL's `fp16_gemm_4.py` tutorial): the shapes have the same depth (K),
/// every mode of the fallback shape divides the preferred one, and every
/// mode of the preferred shape divides the grid. A launch whose preferred
/// shape cannot be shrunk to whole fallback clusters, or whose grid is not
/// whole preferred clusters, is invalid.
pub fn check_mixed_cluster_shapes(
    preferred: [usize; 3],
    fallback: [usize; 3],
    grid: Option<[u32; 3]>,
) -> Result<(), String> {
    if preferred.contains(&0) || fallback.contains(&0) {
        return Err(format!(
            "cluster shapes need a block in every mode: preferred {preferred:?}, fallback {fallback:?}"
        ));
    }
    if preferred[2] != fallback[2] {
        return Err(format!(
            "the fallback cluster {fallback:?} is {} deep but the preferred cluster {preferred:?} is {}: the depths \
             (K) must match",
            fallback[2], preferred[2]
        ));
    }
    if let Some(i) = (0..3).find(|&i| !preferred[i].is_multiple_of(fallback[i])) {
        return Err(format!(
            "the fallback cluster {fallback:?} does not divide the preferred cluster {preferred:?} (mode {i}: {} is \
             not a multiple of {})",
            preferred[i], fallback[i]
        ));
    }
    if let Some(grid) = grid
        && let Some(i) = (0..3).find(|&i| !(grid[i] as usize).is_multiple_of(preferred[i]))
    {
        return Err(format!(
            "the preferred cluster {preferred:?} does not divide the grid {grid:?} (mode {i}: {} is not a multiple of \
             {})",
            grid[i], preferred[i]
        ));
    }
    Ok(())
}

struct ArmGuard;

impl ArmGuard {
    fn push(arm: ClusterArm) -> ArmGuard {
        ARMS.with(|a| a.borrow_mut().push(arm));
        ArmGuard
    }
}

impl Drop for ArmGuard {
    fn drop(&mut self) {
        ARMS.with(|a| a.borrow_mut().pop());
    }
}

/// Build a kernel body for a launch with clusters of `preferred` blocks and,
/// optionally, fallback clusters of `fallback` blocks (CuTe DSL's
/// mixed-cluster kernels: the "mega-kernel" of two exclusive branches that
/// its `fp16_gemm_4.py` tutorial writes by hand, and that its expansion pass
/// derives from one body).
///
/// Without a fallback (or with one equal to `preferred`), `body` is built
/// once, for `preferred`. Otherwise it is built twice, in the two arms of
/// `if is_preferred_cluster(preferred)`, each time with the arm's static
/// cluster shape: so all cluster-shape math stays static, and each cluster
/// runs the arm of its real shape. [`current_cluster_arm`] names the arm
/// while it is built.
///
/// Shared memory allocated inside `body` is allocated once per arm; allocate
/// what both arms use (tiles, barrier storage) before the call (CuTe DSL's
/// `smem_merge_branch_allocs`).
///
/// Panics when the shapes fail [`check_mixed_cluster_shapes`].
#[track_caller]
pub fn mixed_cluster_arms(preferred: [usize; 3], fallback: Option<[usize; 3]>, body: impl Fn(ClusterArm)) {
    let fallback = fallback.filter(|&f| f != preferred);
    let run = |arm: ClusterArm| {
        let _guard = ArmGuard::push(arm);
        body(arm)
    };
    match fallback {
        None => run(ClusterArm { cluster_shape_mnk: preferred, is_preferred: true }),
        Some(f) => {
            if let Err(e) = check_mixed_cluster_shapes(preferred, f, None) {
                panic!("cute: mixed_cluster_arms: {e}");
            }
            if_else(
                is_preferred_cluster(preferred),
                || run(ClusterArm { cluster_shape_mnk: preferred, is_preferred: true }),
                || run(ClusterArm { cluster_shape_mnk: f, is_preferred: false }),
            );
        }
    }
}

/// Report cluster-shape math that bypasses the arm's static shape: inside a
/// [`mixed_cluster_arms`] arm, panics unless `shape` is the arm's cluster
/// shape (`what` names the value in the message). Outside an arm it does
/// nothing.
#[track_caller]
pub fn check_cluster_shape_in_arm(what: &str, shape: [usize; 3]) {
    if let Some(arm) = current_cluster_arm()
        && arm.cluster_shape_mnk != shape
    {
        panic!(
            "cute: {what} is built for clusters of {shape:?}, but this arm of the mixed-cluster kernel runs clusters \
             of {:?}: derive it from the arm's cluster shape",
            arm.cluster_shape_mnk
        );
    }
}

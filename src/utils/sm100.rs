// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cutlass/epilogue/collective/builders/sm100_builder.inl
//   (sm100_dense_compute_tile_shape_or_override and sm100_tmem_warps),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Host-side preflight checks for Blackwell GEMMs (CuTe DSL's
//! `cutlass.utils.sm100`, from `blackwell_helpers`): TMA's 16-byte alignment
//! of an operand's contiguous dimension ([`get_tma_aligned_contiguous_elements`],
//! [`check_tma_tensor_alignment`], [`check_gemm_tma_alignment`]) and the
//! Int8/Uint8 `tcgen05` MMA that sm_107 lacks ([`check_int8_mma_arch`]).
//! Every check returns a [`CantImplement`] (CuTe DSL's `CantImplementError`)
//! with CuTe DSL's message, so a tuning loop can skip the configuration.
//!
//! Also CuTe DSL's SM100 sizing helpers (`cutlass.utils.sm100`): the TMEM columns of one accumulator stage
//! ([`compute_acc_tmem_cols_per_stage`]) and the epilogue subtile
//! ([`compute_epilogue_tile_size`], whose layout form is
//! [`compute_epilogue_tile_shape`](super::compute_epilogue_tile_shape)).
//! Plain integer code, run when a kernel is configured.

use std::fmt;
use std::str::FromStr;

use crate::backend::Arch;
use crate::ir::DType;
use crate::testing::CantImplement;

/// The alignment of a TMA tensor's contiguous dimension in bytes.
pub const TMA_ALIGNMENT_BYTES: usize = 16;

/// The contiguous dimension of a GEMM operand: `M` or `K` for A, `N` or `K`
/// for B, `M` or `N` for D. It parses from CuTe DSL's strings `"m"`, `"n"`
/// and `"k"` (either case).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GemmMajor {
    M,
    N,
    K,
}

impl GemmMajor {
    /// CuTe DSL's string for it: `"m"`, `"n"` or `"k"`.
    pub fn as_str(self) -> &'static str {
        match self {
            GemmMajor::M => "m",
            GemmMajor::N => "n",
            GemmMajor::K => "k",
        }
    }
}

impl AsRef<str> for GemmMajor {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for GemmMajor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for GemmMajor {
    type Err = CantImplement;

    fn from_str(s: &str) -> Result<GemmMajor, CantImplement> {
        match s.to_ascii_lowercase().as_str() {
            "m" => Ok(GemmMajor::M),
            "n" => Ok(GemmMajor::N),
            "k" => Ok(GemmMajor::K),
            _ => Err(CantImplement::new(format!("[alignment] a major mode must be m, n or k, not {s}"))),
        }
    }
}

/// Reject `tcgen05` integer MMA on sm_107 (CuTe DSL's
/// `check_int8_mma_arch(ab_dtype, target_arch)`): sm_107 has no
/// `tcgen05.mma.kind::i8`, so `Int8` and `Uint8` operands fail here instead
/// of in `ptxas`. Other types pass. The target is `target_arch`, else the
/// `CUTE_DSL_ARCH` environment variable, else the detected GPU (feature
/// `cuda`: device 0's compute capability), else nothing is checked.
pub fn check_int8_mma_arch(ab_dtype: DType, target_arch: Option<Arch>) -> Result<(), CantImplement> {
    if !matches!(ab_dtype, DType::I8 | DType::U8) {
        return Ok(());
    }
    let target = match target_arch {
        Some(arch) => Some(arch.to_string()),
        None => std::env::var("CUTE_DSL_ARCH").ok().filter(|s| !s.is_empty()).or_else(detect_gpu_arch),
    };
    match target {
        Some(target) => check_int8_mma_target(ab_dtype, &target),
        None => Ok(()),
    }
}

/// [`check_int8_mma_arch`] for a target given as a string (`sm_107`,
/// `sm_107a`, `107`, any case), as `CUTE_DSL_ARCH` holds it.
pub fn check_int8_mma_target(ab_dtype: DType, target_arch: &str) -> Result<(), CantImplement> {
    if !matches!(ab_dtype, DType::I8 | DType::U8) {
        return Ok(());
    }
    let normalized = target_arch.trim().to_ascii_lowercase();
    let normalized = normalized.strip_prefix("sm_").unwrap_or(&normalized);
    if normalized.starts_with("107") {
        return Err(CantImplement::new(format!("{target_arch} has no tcgen05 MMA for Int8 or Uint8 operands")));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn detect_gpu_arch() -> Option<String> {
    let ctx = crate::cuda::CudaContext::new(0).ok()?;
    let (major, minor) = ctx.compute_capability();
    Some(format!("sm_{major}{minor}"))
}

#[cfg(not(feature = "cuda"))]
fn detect_gpu_arch() -> Option<String> {
    None
}

/// The fewest contiguous elements of `elem_ty` that make 16 bytes (CuTe
/// DSL's `get_tma_aligned_contiguous_elements`): `128 / bits`. FP6 is
/// refused, as CuTe DSL does until its alignment is validated.
pub fn get_tma_aligned_contiguous_elements(elem_ty: DType) -> Result<usize, CantImplement> {
    if elem_ty.bits() == 6 {
        return Err(CantImplement::new(
            "[alignment] FP6 operands are refused: their TMA alignment rule is not checked yet",
        ));
    }
    Ok(TMA_ALIGNMENT_BYTES * 8 / elem_ty.bits())
}

/// An error unless the contiguous extent of a TMA operand spans a multiple of 16 bytes
/// (CuTe DSL's `check_tma_tensor_alignment`).
pub fn check_tma_tensor_alignment(
    tensor_name: &str,
    elem_ty: DType,
    contiguous_dim_size: usize,
    contiguous_dim_name: &str,
) -> Result<(), CantImplement> {
    let required = get_tma_aligned_contiguous_elements(elem_ty)?;
    if !contiguous_dim_size.is_multiple_of(required) {
        return Err(CantImplement::new(format!(
            "[alignment] {tensor_name}: {contiguous_dim_name}={contiguous_dim_size} {} elements is not a whole \
             number of {TMA_ALIGNMENT_BYTES}-byte TMA units (use a multiple of {required})",
            elem_ty.cute_name()
        )));
    }
    Ok(())
}

/// Validate a GEMM's operands for TMA loads of A and B and, with a `d_dtype`,
/// a TMA store of D (CuTe DSL's `check_gemm_tma_alignment`). `a_major` is
/// `M` or `K`, `b_major` `N` or `K`, `d_major` `M` or `N` (needed with
/// `d_dtype`); anything else is an "Invalid ..._major" error.
/// `output_tensor_name` is `"D"` in CuTe DSL.
#[allow(clippy::too_many_arguments)]
pub fn check_gemm_tma_alignment(
    m: usize,
    n: usize,
    k: usize,
    a_dtype: DType,
    b_dtype: DType,
    d_dtype: Option<DType>,
    a_major: GemmMajor,
    b_major: GemmMajor,
    d_major: Option<GemmMajor>,
    output_tensor_name: &str,
) -> Result<(), CantImplement> {
    check_gemm_tma_alignment_str(
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        d_dtype,
        a_major.as_str(),
        b_major.as_str(),
        d_major.map(GemmMajor::as_str),
        output_tensor_name,
    )
}

/// [`check_gemm_tma_alignment`] with CuTe DSL's strings for the majors
/// (`"m"`, `"n"`, `"k"`, case-sensitive as there), so that a string that is
/// not one gives CuTe DSL's "Invalid a_major: x" message.
#[allow(clippy::too_many_arguments)]
pub fn check_gemm_tma_alignment_str(
    m: usize,
    n: usize,
    k: usize,
    a_dtype: DType,
    b_dtype: DType,
    d_dtype: Option<DType>,
    a_major: &str,
    b_major: &str,
    d_major: Option<&str>,
    output_tensor_name: &str,
) -> Result<(), CantImplement> {
    if !["m", "k"].contains(&a_major) {
        return Err(CantImplement::new(format!("[alignment] a_major must be m or k, not {a_major}")));
    }
    if !["n", "k"].contains(&b_major) {
        return Err(CantImplement::new(format!("[alignment] b_major must be n or k, not {b_major}")));
    }
    if d_dtype.is_some() && !matches!(d_major, Some("m" | "n")) {
        return Err(CantImplement::new(format!(
            "[alignment] d_major must be m or n, not {}",
            d_major.unwrap_or("None")
        )));
    }
    let (a_size, a_name) = if a_major == "m" { (m, "M") } else { (k, "K") };
    check_tma_tensor_alignment("A TMA load", a_dtype, a_size, a_name)?;
    let (b_size, b_name) = if b_major == "n" { (n, "N") } else { (k, "K") };
    check_tma_tensor_alignment("B TMA load", b_dtype, b_size, b_name)?;
    if let Some(d_dtype) = d_dtype {
        let (d_size, d_name) = if d_major == Some("m") { (m, "M") } else { (n, "N") };
        check_tma_tensor_alignment(&format!("{output_tensor_name} TMA store"), d_dtype, d_size, d_name)?;
    }
    Ok(())
}

// ---- SM100 sizing helpers ----
/// The TMEM columns one accumulator stage spans, before the allocation's
/// 32-column minimum and power-of-two rounding (CuTe DSL's
/// `cutlass.utils.sm100.compute_acc_tmem_cols_per_stage`). `cta_tile_m` is
/// the per-CTA M (half the MMA's M for 2-CTA MMAs).
///
/// * 2-CTA: `cta_tile_n / 4` for M up to 32, `/ 2` up to 64, else
///   `cta_tile_n` (the datapaths M leaves free hold more N).
/// * 1-CTA with A from shared memory and M = 64: the accumulator is
///   interleaved, two N tiles of `mma_n` per set of columns:
///   `ceil(n_tiles / 2) * mma_n` with `n_tiles = cta_tile_n / mma_n`.
/// * otherwise `cta_tile_n`.
pub fn compute_acc_tmem_cols_per_stage(
    cta_tile_m: usize,
    cta_tile_n: usize,
    use_2cta: bool,
    mma_n: usize,
    transform_a_source_is_tmem: bool,
) -> usize {
    if use_2cta {
        return match cta_tile_m {
            0..=32 => cta_tile_n / 4,
            33..=64 => cta_tile_n / 2,
            _ => cta_tile_n,
        };
    }
    if !transform_a_source_is_tmem && cta_tile_m == 64 {
        let n_tiles = cta_tile_n / mma_n;
        n_tiles.div_ceil(2) * mma_n
    } else {
        cta_tile_n
    }
}

/// The TMEM warp arrangement `(warp_m, warp_n)` of the epilogue's four
/// warps: `(2, 2)` for 2-CTA MMAs with a per-CTA M of 64, `(4, 1)`
/// otherwise (CuTe C++'s `sm100_tmem_warps`).
pub fn default_tmem_warp_shape_mn(cta_tile_m: usize, use_2cta: bool) -> (usize, usize) {
    if cta_tile_m == 64 && use_2cta { (2, 2) } else { (4, 1) }
}

/// The epilogue subtile `(tile_m, tile_n)` (CuTe DSL's
/// `cutlass.utils.sm100.compute_epilogue_tile_size`, CuTe C++'s
/// `sm100_dense_compute_tile_shape_or_override` for an automatic tile).
/// `d_width` and `c_width` are the element widths of D and of the source C
/// (`None`: no source); `*_is_m_major` say whether they are column-major.
///
/// * `tile_m = min(cta_tile_m, 32 * warp_m)` (32 TMEM datapaths a warp).
/// * The target N: without a source, 4096 elements per subtile (8192 for
///   4-bit D) over `tile_m`; with one, by the wider width: 32 bits: 16 if
///   `cta_tile_m > 64` and `cta_tile_n <= 128`, else 32; 16 bits: 32 if
///   `cta_tile_n <= 128`, else 64; narrower: 64. It is halved until it
///   divides `cta_tile_n` (CuTe DSL's step; CuTe C++ falls back to
///   `cta_tile_n` below instead).
/// * The TMA store minimum per output: `8 * warp_n` for an M-major output
///   (or no source), else `128 / width * warp_n`, and `128 * warp_n` for
///   6-bit types.
/// * `tile_n = min(cta_tile_n, max(target, minimums))`, or `cta_tile_n` if
///   that does not divide it.
///
/// `tmem_warp_shape_mn` overrides [`default_tmem_warp_shape_mn`]. This
/// does not check the minimums against `cta_tile_n`
/// ([`compute_epilogue_tile_shape`](super::compute_epilogue_tile_shape)
/// does).
#[allow(clippy::too_many_arguments)]
pub fn compute_epilogue_tile_size(
    cta_tile_m: usize,
    cta_tile_n: usize,
    use_2cta: bool,
    d_width: usize,
    c_width: Option<usize>,
    d_is_m_major: bool,
    c_is_m_major: bool,
    tmem_warp_shape_mn: Option<(usize, usize)>,
) -> (usize, usize) {
    let (warp_m, warp_n) = tmem_warp_shape_mn.unwrap_or_else(|| default_tmem_warp_shape_mn(cta_tile_m, use_2cta));
    let max_bits = c_width.map_or(d_width, |c| c.max(d_width));
    const DP_FULL: usize = 32;
    let tile_m = cta_tile_m.min(DP_FULL * warp_m);
    let mut n_perf = match c_width {
        None => (if max_bits == 4 { 8192 } else { 4096 }) / tile_m,
        Some(_) => match max_bits {
            32 if cta_tile_m > 64 && cta_tile_n <= 128 => 16,
            32 => 32,
            16 if cta_tile_n <= 128 => 32,
            16 => 64,
            _ => 64,
        },
    };
    while n_perf > 1 && !cta_tile_n.is_multiple_of(n_perf) {
        n_perf /= 2;
    }
    let (n_min_d, n_min_c) = epilogue_n_minimums(warp_n, d_width, c_width, d_is_m_major, c_is_m_major);
    let mut tile_n = cta_tile_n.min(n_perf.max(n_min_c).max(n_min_d));
    if !cta_tile_n.is_multiple_of(tile_n) {
        tile_n = cta_tile_n;
    }
    (tile_m, tile_n)
}

/// The TMA-store minimum N of D and of C for `warp_n` warps along N.
pub(crate) fn epilogue_n_minimums(
    warp_n: usize,
    d_width: usize,
    c_width: Option<usize>,
    d_is_m_major: bool,
    c_is_m_major: bool,
) -> (usize, usize) {
    let n_major_min = |width: usize| if width == 6 { 128 * warp_n } else { 128 / width * warp_n };
    let n_min_d = if d_is_m_major { 8 * warp_n } else { n_major_min(d_width) };
    let n_min_c = match c_width {
        Some(c) if !c_is_m_major => n_major_min(c),
        _ => 8 * warp_n,
    };
    (n_min_d, n_min_c)
}

// The tiled tcgen05 MMA and the shared-memory layout heuristics.
pub use super::sm100_mma::{
    get_smem_layout_atom_ab, make_smem_layout, make_smem_layout_a, make_smem_layout_b, make_smem_layout_c,
    make_trivial_tiled_mma,
};

// The TMA load operation for a cluster shape and MMA.
pub use super::sm100_tma::{
    Sm100TmaLoadOp, cluster_shape_to_tma_atom_a, cluster_shape_to_tma_atom_b, cluster_shape_to_tma_atom_sfb,
};

// The block-scaled tiled MMAs.
pub use super::sm100_mma::{make_blockscaled_trivial_tiled_mma, sm103_make_blockscaled_trivial_tiled_mma};

// The TMEM load and shared-memory store selection and the epilogue layouts.
pub use super::sm100_epilogue::{
    TmemLoadChoice, get_smem_layout_atom_epi, get_smem_store_op, get_tmem_load_op, make_smem_layout_epi,
    select_smem_store_op, select_tmem_load_op,
};
// The sparse GEMM element types and layouts.
pub use super::sm100_sparse::{
    make_sparse_a_dtype, make_sparse_e_dtype, make_sparse_gmem_layout_e, make_sparse_smem_layout_a,
    make_sparse_smem_layout_e, make_sparse_tmem_layout_e, make_tensor_e_atom_layout,
};

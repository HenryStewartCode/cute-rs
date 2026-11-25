// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's block-scaled MMA arithmetic (roadmap part
//! m10-sm120-blockscaled), modelled from the PTX ISA (9.7.14.5 `mma` with
//! `.block_scale`, its fragment tables for `.m16n8k64 .e2m1` and `.m16n8k32`
//! 8-bit operands, and the scale-factor tables for byte-id and thread-id 0)
//! independently of cute-rs's layout algebra.
//!
//! * **Shared arithmetic.** [`decode_operand`], [`decode_scale`] and
//!   [`block_scaled_dot`] are the block-scale arithmetic: decode A and B,
//!   multiply each K-block's products by its two scale factors and add them
//!   to the accumulator in Float32 in order of K. The tcgen05 block-scaled
//!   MMAs (m10-umma-blockscaled) are meant to reuse them.
//! * **Registers.** A warp-level block-scaled `mma.sync` takes A and B as
//!   32-bit registers: FP4 packed eight per register (`kind::mxf4`,
//!   `kind::mxf4nvf4`), or four 8-bit containers per register
//!   (`kind::mxf8f6f4`), where an E4M3 or E5M2 value fills its byte and an
//!   E2M1 value sits in bits 5:2 (the data-format rules of `kind::f8f6f4`).
//! * **Scale factors.** A and B each take a 32-bit register in every lane,
//!   the scale of K-block `j` (of 16 or 32 elements) in byte `j` (byte-id
//!   0). With thread-id 0, row `r` of A takes its scales from lane
//!   `4 (r % 8) + r / 8` and column `n` of B from lane `4 n`. UE8M0 is
//!   `2^(e - 127)` (`0xFF` NaN); UE4M3 is E4M3 without the sign bit.
//! * **Sparse** (m11-prims-mma-forms, `mma.sp::ordered_metadata` with
//!   `.block_scale`, `m16n8k64` and `m16n8k128`). A's four registers hold
//!   the compressed half of K in the dense layout of half the K; B's four
//!   registers extend the dense B layout to the full K. Each row of A has
//!   sixteen 4-bit metadata fields (2:4 on 8-bit containers, 4:8 by pairs
//!   on FP4), field `c` of row `r` in nibble `4 (r / 8) + c % 4` of lane
//!   `4 (r % 8) + c / 4` (sparsity selector 0). Scale factor `j` is byte
//!   `bid + j`; row `r` of A takes its register from lane
//!   `4 (r % 8) + 2 tid_a + r / 8`, column `n` of B from lane `4 n + tid_b`.
//!   These placements are cute-rs's extension of the dense and 16-bit
//!   sparse tables; the ISA's figures for these shapes could not be
//!   checked, and no hardware runs them.

use super::{Machine, R};
use crate::ir::{BlockScaledMmaKind, DType, SpBlockScaledMmaKind, ValueId};

/// The value of an A or B element of type `ty` (E2M1, E4M3 or E5M2) from
/// its bits.
pub(super) fn decode_operand(ty: DType, bits: u32) -> f64 {
    crate::numeric::decode(ty.float_format().expect("a float operand type"), bits)
}

/// The value of a scale factor (`F8E8M0FNU` as UE8M0, `F8E4M3FN` as UE4M3)
/// from its byte.
pub(super) fn decode_scale(sf: DType, byte: u32) -> f64 {
    let bits = if sf == DType::F8E4M3FN { byte & 0x7f } else { byte & 0xff };
    crate::numeric::decode(sf.float_format().expect("a float scale type"), bits)
}

/// `acc + sum_k a(k) sfa(k / vec) b(k) sfb(k / vec)` over `k` in `0..k`,
/// each product (exact in f64 for the narrow formats) rounded to Float32 and
/// added in order of K in Float32.
pub(super) fn block_scaled_dot(
    acc: f32,
    k: usize,
    vec: usize,
    a: impl Fn(usize) -> f64,
    sfa: impl Fn(usize) -> f64,
    b: impl Fn(usize) -> f64,
    sfb: impl Fn(usize) -> f64,
) -> f32 {
    (0..k).fold(acc, |acc, kk| acc + (a(kk) * sfa(kk / vec) * b(kk) * sfb(kk / vec)) as f32)
}

/// `(row, k)` of nibble or byte `j` of register `r` of lane `lane`'s A
/// fragment.
fn a_coord(kind: BlockScaledMmaKind, lane: usize, r: usize, j: usize) -> (usize, usize) {
    let (g, t) = (lane >> 2, lane % 4);
    if kind.is_fp4_packed() {
        // .e2m1 m16n8k64: a0..a7 row g, a8..a15 row g + 8, then 32 columns on.
        (g + 8 * (r & 1), 8 * t + j + 32 * (r >> 1))
    } else {
        // 8-bit m16n8k32: a0..a3 row g, a4..a7 row g + 8, then 16 columns on.
        (g + 8 * (r & 1), 4 * t + j + 16 * (r >> 1))
    }
}

/// `(k, n)` of nibble or byte `j` of register `r` of lane `lane`'s B
/// fragment.
fn b_coord(kind: BlockScaledMmaKind, lane: usize, r: usize, j: usize) -> (usize, usize) {
    let (g, t) = (lane >> 2, lane % 4);
    if kind.is_fp4_packed() { (8 * t + j + 32 * r, g) } else { (4 * t + j + 16 * r, g) }
}

/// The element of type `ty` at position `j` of a register: a nibble when
/// packed, else a byte (an E2M1 value in its bits 5:2).
fn element(ty: DType, packed: bool, reg: u32, j: usize) -> f64 {
    if packed {
        decode_operand(ty, (reg >> (4 * j)) & 0xf)
    } else {
        let byte = (reg >> (8 * j)) & 0xff;
        let bits = if ty == DType::F4E2M1FN { (byte >> 2) & 0xf } else { byte };
        decode_operand(ty, bits)
    }
}

impl<'a> Machine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_block_scaled(
        &mut self,
        kind: BlockScaledMmaKind,
        d: &[ValueId],
        a: &[ValueId],
        b: &[ValueId],
        c: &[ValueId],
        sfa: ValueId,
        sfb: ValueId,
        active: &[usize],
    ) -> R {
        let bases = self.full_warps(active, "a block-scaled mma.sync")?;
        let (m, n, k) = kind.shape_mnk();
        let (vec, sf) = (kind.sf_vec_size as usize, kind.sf_dtype());
        let (per, packed) = (kind.values_per_register(), kind.is_fp4_packed());
        let nsf = kind.scales_per_mma();
        for base in bases {
            let mut am = vec![vec![0f64; k]; m];
            let mut bm = vec![vec![0f64; n]; k];
            let mut cm = vec![vec![0f32; n]; m];
            for lane in 0..32 {
                let t = base + lane;
                for (r, v) in a.iter().enumerate() {
                    let reg = self.get(*v, t) as u32;
                    for j in 0..per {
                        let (row, col) = a_coord(kind, lane, r, j);
                        am[row][col] = element(kind.a, packed, reg, j);
                    }
                }
                for (r, v) in b.iter().enumerate() {
                    let reg = self.get(*v, t) as u32;
                    for j in 0..per {
                        let (kk, col) = b_coord(kind, lane, r, j);
                        bm[kk][col] = element(kind.b, packed, reg, j);
                    }
                }
                for (i, v) in c.iter().enumerate() {
                    let (g, tt) = (lane >> 2, lane % 4);
                    cm[g + 8 * (i >> 1)][2 * tt + (i & 1)] = f32::from_bits(self.get(*v, t) as u32);
                }
            }
            // Scale factors: row r of A from lane 4 (r % 8) + r / 8, column
            // n of B from lane 4 n, K-block j in byte j.
            let sfa_m: Vec<Vec<f64>> = (0..m)
                .map(|r| {
                    let reg = self.get(sfa, base + 4 * (r % 8) + r / 8) as u32;
                    (0..nsf).map(|j| decode_scale(sf, (reg >> (8 * j)) & 0xff)).collect()
                })
                .collect();
            let sfb_n: Vec<Vec<f64>> = (0..n)
                .map(|col| {
                    let reg = self.get(sfb, base + 4 * col) as u32;
                    (0..nsf).map(|j| decode_scale(sf, (reg >> (8 * j)) & 0xff)).collect()
                })
                .collect();
            for lane in 0..32 {
                let (g, tt) = (lane >> 2, lane % 4);
                for (i, v) in d.iter().enumerate() {
                    let (r, col) = (g + 8 * (i >> 1), 2 * tt + (i & 1));
                    let acc = block_scaled_dot(
                        cm[r][col],
                        k,
                        vec,
                        |kk| am[r][kk],
                        |j| sfa_m[r][j],
                        |kk| bm[kk][col],
                        |j| sfb_n[col][j],
                    );
                    self.set(*v, base + lane, acc.to_bits() as u64);
                }
            }
            self.report.mma_instructions += 1;
        }
        Ok(())
    }
}

impl<'a> Machine<'a> {
    /// m11-prims-mma-forms: the warp-level sparse block-scaled `mma.sp`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_sp_block_scaled(
        &mut self,
        kind: SpBlockScaledMmaKind,
        d: &[ValueId],
        [a, b, c]: [&Vec<ValueId>; 3],
        e: ValueId,
        sfa: ValueId,
        sfb: ValueId,
        active: &[usize],
    ) -> R {
        let bases = self.full_warps(active, "a sparse block-scaled mma.sp")?;
        let (m, n, k) = kind.shape_mnk();
        let kh = k / 2;
        let (per, packed) = (kind.values_per_register(), kind.is_fp4_packed());
        // The dense kind of half the K: its A layout is the compressed A's.
        let half = BlockScaledMmaKind { a: kind.a, b: kind.b, sf_vec_size: 32 };
        let (vec, nsf) = (kind.vec(), kind.scale_vec as usize);
        let span = if packed { 8 } else { 4 };
        for base in bases {
            let mut am = vec![vec![0f64; kh]; m];
            let mut bm = vec![vec![0f64; n]; k];
            let mut cm = vec![vec![0f32; n]; m];
            for lane in 0..32 {
                let t = base + lane;
                for (r, v) in a.iter().enumerate() {
                    let reg = self.get(*v, t) as u32;
                    for j in 0..per {
                        let (row, col) = a_coord(half, lane, r, j);
                        am[row][col] = element(kind.a, packed, reg, j);
                    }
                }
                for (r, v) in b.iter().enumerate() {
                    let reg = self.get(*v, t) as u32;
                    for j in 0..per {
                        let (kk, col) = b_coord(half, lane, r, j);
                        bm[kk][col] = element(kind.b, packed, reg, j);
                    }
                }
                for (i, v) in c.iter().enumerate() {
                    let (g, tt) = (lane >> 2, lane % 4);
                    cm[g + 8 * (i >> 1)][2 * tt + (i & 1)] = f32::from_bits(self.get(*v, t) as u32);
                }
            }
            // Expand A with the metadata.
            let mut dense = vec![vec![0f64; k]; m];
            for (r, row) in dense.iter_mut().enumerate() {
                for chunk in 0..k / span {
                    let (lane, nibble) = (4 * (r % 8) + chunk / 4, 4 * (r / 8) + chunk % 4);
                    let bits = (self.get(e, base + lane) >> (4 * nibble) & 0xf) as u32;
                    let Some((i0, i1)) = super::sparse_indices(bits) else {
                        return Err(self.err(
                            format!(
                                "{}: metadata nibble {nibble} of lane {lane} (row {r}, K elements {} to {} of A) is \
                                 {bits:#06b}, which does not name two positions in increasing order",
                                kind.ptx(),
                                span * chunk,
                                span * chunk + span - 1
                            ),
                            Some(base + lane),
                        ));
                    };
                    if packed {
                        for x in 0..2 {
                            row[8 * chunk + 2 * i0 + x] = am[r][4 * chunk + x];
                            row[8 * chunk + 2 * i1 + x] = am[r][4 * chunk + 2 + x];
                        }
                    } else {
                        row[4 * chunk + i0] = am[r][2 * chunk];
                        row[4 * chunk + i1] = am[r][2 * chunk + 1];
                    }
                }
            }
            let byte = |reg: u32, bid: u8, j: usize| (reg >> (8 * (bid as usize + j))) & 0xff;
            let sfa_m: Vec<Vec<f64>> = (0..m)
                .map(|r| {
                    let reg = self.get(sfa, base + 4 * (r % 8) + 2 * kind.tid_a as usize + r / 8) as u32;
                    (0..nsf).map(|j| decode_scale(kind.sf, byte(reg, kind.bid_a, j))).collect()
                })
                .collect();
            let sfb_n: Vec<Vec<f64>> = (0..n)
                .map(|col| {
                    let reg = self.get(sfb, base + 4 * col + kind.tid_b as usize) as u32;
                    (0..nsf).map(|j| decode_scale(kind.sf, byte(reg, kind.bid_b, j))).collect()
                })
                .collect();
            for lane in 0..32 {
                let (g, tt) = (lane >> 2, lane % 4);
                for (i, v) in d.iter().enumerate() {
                    let (r, col) = (g + 8 * (i >> 1), 2 * tt + (i & 1));
                    let acc = block_scaled_dot(
                        cm[r][col],
                        k,
                        vec,
                        |kk| dense[r][kk],
                        |j| sfa_m[r][j],
                        |kk| bm[kk][col],
                        |j| sfb_n[col][j],
                    );
                    self.set(*v, base + lane, acc.to_bits() as u64);
                }
            }
            self.report.mma_instructions += 1;
        }
        Ok(())
    }
}

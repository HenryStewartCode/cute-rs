// SPDX-License-Identifier: MIT OR Apache-2.0

//! The elements one TMA instruction moves, in each of the PTX ISA's tensor access modes: the tiled box, the im2col
//! column of pixels, and the four rows of `.tile::gather4` and `.tile::scatter4`. This is
//! the simulator's model of the hardware, written from the PTX ISA's description of the
//! modes alone (no layout algebra), so that a simulated copy built from an atom checks the
//! atom's construction.

use super::{MAX_RANK, TmaDesc};

/// How a TMA instruction addresses its tensor: the coordinates (and offsets) it gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmaAccess<'a> {
    /// `.tile`: the box whose first element is at `start` (one coordinate per dimension).
    Tile { start: &'a [i64] },
    /// `.im2col` (`offsets`: one per spatial dimension, W first) or `.im2col_no_offs`
    /// (`offsets` empty): the column of pixels starting at the filter base `start`
    /// (`c, w, (h, (d,)) n`).
    Im2col { start: &'a [i64], offsets: &'a [i64] },
    /// `.tile::gather4` and `.tile::scatter4`: four rows of `box_dims[0]` elements of a
    /// 2-D tensor, from column `x` of rows `rows`.
    Rows4 { x: i64, rows: [i64; 4] },
}

/// One element of a TMA access ([`TmaDesc::access_elements`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TmaElement {
    /// Its coordinates in the tensor (may be negative or past the end).
    pub coords: [i64; MAX_RANK],
    /// Its bit offset in shared memory from the start of the box, before any swizzle
    /// (the padded sub-byte encodings leave the end of each 16-byte slot unwritten).
    pub smem_bit: u64,
    /// Its global address in bits (byte address times 8, plus the bit within the byte),
    /// or `None` when it is out of bounds.
    pub global_bit: Option<u64>,
}

impl TmaDesc {
    /// Every element `access` moves, in shared-memory order, with its coordinates, its
    /// bit offset in the box's shared-memory image and its global bit address (`None`
    /// out of bounds, which a load fills and a store skips). Errors are what the PTX ISA
    /// leaves undefined or a descriptor of the wrong kind for the access:
    ///
    /// - `.tile`: the box (as [`TmaDesc::box_elements`], and the sub-byte encodings by
    ///   16-value groups);
    /// - `.im2col`: [`Im2colParams::pixels_per_column`](super::Im2colParams) pixels from
    ///   the filter base, each [`Im2colParams::channels_per_pixel`](super::Im2colParams)
    ///   channels from `c`. The pixels walk the bounding box (spatial coordinates
    ///   `lower_corner` to `extent - 1 + upper_corner`), W fastest, by the traversal
    ///   stride of each spatial dimension, back to the lower corner and on to the next
    ///   row, plane and image at its end; each pixel's element is read at its position
    ///   plus the instruction's im2col offsets, out of bounds (padding, or images past N)
    ///   when that falls outside the tensor. The filter base must be inside the bounding
    ///   box;
    /// - `.tile::gather4`/`.tile::scatter4`: `box_dims[0]` elements from `x` in each of the
    ///   four rows, one after another (`box_dims[1]` is 1).
    pub fn access_elements(&self, access: &TmaAccess<'_>) -> Result<Vec<TmaElement>, String> {
        let r = (self.rank as usize).min(MAX_RANK);
        let bits = u64::from(self.data_type.bit_width());
        let base = u128::from(self.global_address) * 8;
        // The global bit of the element at `coords`, if in bounds.
        let place = |coords: &[i64; MAX_RANK]| -> Option<u64> {
            let mut a = base as i128;
            for (i, &c) in coords.iter().enumerate().take(r) {
                if c < 0 || c as u64 >= self.global_dims[i] {
                    return None;
                }
                a += if i == 0 { c as i128 * bits as i128 } else { c as i128 * self.global_strides[i - 1] as i128 * 8 };
            }
            u64::try_from(a).ok()
        };
        // The bit offset in shared memory of the `n`th element in box order.
        let slot = self.data_type.group_slot_bytes();
        let smem_bit = |n: u64| -> u64 {
            match slot {
                Some(s) => n / 16 * s * 8 + n % 16 * bits,
                None => n * bits,
            }
        };
        let mut out = Vec::new();
        match *access {
            TmaAccess::Tile { start } => {
                if self.im2col.is_some() {
                    return Err("a tile-mode access with an im2col tensor map (it takes im2col coordinates and \
                                offsets: an im2col atom's copy)"
                        .to_string());
                }
                if start.len() != r {
                    return Err(format!("{} coordinates for a tensor map of rank {r}", start.len()));
                }
                // The PTX ISA's restrictions on tensor copies: with .b4x16_p64 and .b6x16_p32
                // the first coordinate is a multiple of 128.
                if self.data_type.is_padded() && start[0].rem_euclid(128) != 0 {
                    return Err(format!(
                        "the first coordinate {} is not a multiple of 128, as the padded {} encoding requires",
                        start[0],
                        self.data_type.name()
                    ));
                }
                let extents: Vec<u64> = (0..r).map(|i| self.box_extent(i)).collect();
                let total: u64 = extents.iter().product();
                for n in 0..total {
                    let mut rest = n;
                    let mut coords = [0i64; MAX_RANK];
                    for i in 0..r {
                        let j = rest % extents[i];
                        rest /= extents[i];
                        let step = if i == 0 && self.interleave == super::TensorMapInterleave::none {
                            1
                        } else {
                            i64::from(self.element_strides[i].max(1))
                        };
                        coords[i] = start[i] + j as i64 * step;
                    }
                    out.push(TmaElement { coords, smem_bit: smem_bit(n), global_bit: place(&coords) });
                }
            }
            TmaAccess::Im2col { start, offsets } => {
                let Some(p) = &self.im2col else {
                    return Err("an im2col access with a tiled tensor map (make the atom with make_im2col_tma_atom, \
                                whose descriptor is an im2col one)"
                        .to_string());
                };
                if start.len() != r {
                    return Err(format!("{} coordinates for an im2col tensor map of rank {r}", start.len()));
                }
                let ns = r - 2;
                if !offsets.is_empty() && offsets.len() != ns {
                    return Err(format!("{} im2col offsets for a tensor map of rank {r}: give {ns}", offsets.len()));
                }
                let max = super::im2col_offset_max(r);
                if let Some(o) = offsets.iter().find(|&&o| o < 0 || o > i64::from(max)) {
                    return Err(format!("the im2col offset {o} is outside 0 to {max} (the range for rank {r})"));
                }
                let lower: Vec<i64> = (0..ns).map(|k| i64::from(p.lower_corner[k])).collect();
                let upper: Vec<i64> =
                    (0..ns).map(|k| self.global_dims[k + 1] as i64 - 1 + i64::from(p.upper_corner[k])).collect();
                let mut pixel: Vec<i64> = start[1..r - 1].to_vec();
                for k in 0..ns {
                    if pixel[k] < lower[k] || pixel[k] > upper[k] {
                        return Err(format!(
                            "the filter base {:?} (spatial coordinates, W first) is outside the bounding box, which \
                             spans {lower:?} to {upper:?}: the PTX ISA requires it inside",
                            &start[1..r - 1]
                        ));
                    }
                }
                let mut image = start[r - 1];
                let cpp = u64::from(p.channels_per_pixel);
                for q in 0..u64::from(p.pixels_per_column) {
                    for j in 0..cpp {
                        let mut coords = [0i64; MAX_RANK];
                        coords[0] = start[0] + j as i64;
                        for k in 0..ns {
                            coords[k + 1] = pixel[k] + offsets.get(k).copied().unwrap_or(0);
                        }
                        coords[r - 1] = image;
                        out.push(TmaElement { coords, smem_bit: smem_bit(q * cpp + j), global_bit: place(&coords) });
                    }
                    // The next pixel: W steps by its traversal stride, wrapping to the lower
                    // corner (and the next row, plane, image) past the upper one.
                    let mut k = 0;
                    loop {
                        if k == ns {
                            image += 1;
                            break;
                        }
                        pixel[k] += i64::from(self.element_strides[k + 1].max(1));
                        if pixel[k] <= upper[k] {
                            break;
                        }
                        pixel[k] = lower[k];
                        k += 1;
                    }
                }
            }
            TmaAccess::Rows4 { x, rows } => {
                if self.im2col.is_some() || r != 2 {
                    return Err(format!(
                        "the gather4 and scatter4 modes take a 2-D tiled tensor map, not a {}rank-{r} one",
                        if self.im2col.is_some() { "im2col " } else { "" }
                    ));
                }
                let len = u64::from(self.box_dims[0]);
                for (k, &y) in rows.iter().enumerate() {
                    for j in 0..len {
                        let mut coords = [0i64; MAX_RANK];
                        coords[0] = x + j as i64;
                        coords[1] = y;
                        let n = k as u64 * len + j;
                        out.push(TmaElement { coords, smem_bit: smem_bit(n), global_bit: place(&coords) });
                    }
                }
            }
        }
        Ok(out)
    }

    /// Elements one instruction of `access`'s kind moves: four rows for
    /// [`TmaAccess::Rows4`], the box otherwise.
    pub fn access_count(&self, rows4: bool) -> u64 {
        if rows4 { 4 * u64::from(self.box_dims[0]) } else { self.box_elements_count() }
    }

    /// Bytes of shared memory one instruction of `access`'s kind covers (with the padded
    /// encodings' padding): the transaction count is [`TmaDesc::access_tx_bytes`].
    pub fn access_smem_bytes(&self, rows4: bool) -> u64 {
        let n = self.access_count(rows4);
        match self.data_type.group_slot_bytes() {
            Some(s) => n.div_ceil(16) * s,
            None => n * u64::from(self.data_type.bit_width()) / 8,
        }
    }

    /// Bytes one instruction of `access`'s kind counts down an mbarrier by: its elements
    /// at the encoding's width (packed), whether in bounds or filled.
    pub fn access_tx_bytes(&self, rows4: bool) -> u64 {
        self.access_count(rows4) * u64::from(self.data_type.bit_width()) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Im2colParams, TensorMapDataType};
    use super::*;

    /// A 4-D NHWC f16 tensor: C 8, W 5, H 4, N 2, densely packed.
    fn nhwc() -> TmaDesc {
        let mut d = TmaDesc::new(TensorMapDataType::float16, 4);
        d.global_address = 0x1000;
        d.global_dims[..4].copy_from_slice(&[8, 5, 4, 2]);
        d.global_strides[..3].copy_from_slice(&[16, 80, 320]);
        d.im2col = Some(Im2colParams {
            lower_corner: [-1, -1, 0],
            upper_corner: [-1, -1, 0],
            channels_per_pixel: 8,
            pixels_per_column: 6,
        });
        d
    }

    #[test]
    fn im2col_walks_the_bounding_box_with_offsets() {
        let d = nhwc();
        // A 3x3 filter with padding 1: the box spans w -1..=3, h -1..=2. Start at the
        // filter base (w, h) = (2, -1) of image 0, offsets (1, 0): pixels (2,-1), (3,-1),
        // then wrap to (-1, 0), (0, 0), (1, 0), (2, 0); each read at w + 1.
        let e = d.access_elements(&TmaAccess::Im2col { start: &[0, 2, -1, 0], offsets: &[1, 0] }).unwrap();
        assert_eq!(e.len(), 48);
        let pixels: Vec<[i64; 3]> = e.iter().step_by(8).map(|x| [x.coords[1], x.coords[2], x.coords[3]]).collect();
        assert_eq!(pixels, vec![[3, -1, 0], [4, -1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]);
        // h = -1 is padding; (0, 0, 0) is in bounds at byte 0.
        assert_eq!(e[0].global_bit, None);
        assert_eq!(e[16].global_bit, Some(0x1000 * 8));
        assert_eq!(e[17].global_bit, Some(0x1002 * 8));
        assert_eq!(e[47].smem_bit, 47 * 16);
        // Past the last pixel of image 0 the walk goes on into image 1.
        let e = d.access_elements(&TmaAccess::Im2col { start: &[0, 3, 2, 0], offsets: &[0, 0] }).unwrap();
        let pixels: Vec<[i64; 3]> = e.iter().step_by(8).map(|x| [x.coords[1], x.coords[2], x.coords[3]]).collect();
        assert_eq!(pixels[..3], [[3, 2, 0], [-1, -1, 1], [0, -1, 1]]);
        // Images past N are out of bounds.
        let e = d.access_elements(&TmaAccess::Im2col { start: &[0, 3, 2, 1], offsets: &[1, 1] }).unwrap();
        assert!(e[8..].iter().all(|x| x.global_bit.is_none()));
        // A base outside the box, and offsets out of range.
        let err = d.access_elements(&TmaAccess::Im2col { start: &[0, 4, 0, 0], offsets: &[0, 0] }).unwrap_err();
        assert!(err.contains("outside the bounding box"), "{err}");
        let err = d.access_elements(&TmaAccess::Im2col { start: &[0, 0, 0, 0], offsets: &[256, 0] }).unwrap_err();
        assert!(err.contains("outside 0 to 255"), "{err}");
        let err = d.access_elements(&TmaAccess::Tile { start: &[0, 0, 0, 0] }).unwrap_err();
        assert!(err.contains("im2col tensor map"), "{err}");
    }

    #[test]
    fn im2col_traversal_stride_skips_pixels() {
        let mut d = nhwc();
        d.element_strides[1] = 2;
        // W walks -1, 1, 3 by 2, then wraps to -1 on the next row.
        let e = d.access_elements(&TmaAccess::Im2col { start: &[0, -1, 0, 0], offsets: &[] }).unwrap();
        let pixels: Vec<[i64; 2]> = e.iter().step_by(8).map(|x| [x.coords[1], x.coords[2]]).collect();
        assert_eq!(pixels, vec![[-1, 0], [1, 0], [3, 0], [-1, 1], [1, 1], [3, 1]]);
    }

    #[test]
    fn gather4_reads_four_rows() {
        let mut d = TmaDesc::new(TensorMapDataType::float32, 2);
        d.global_address = 0x2000;
        d.global_dims[..2].copy_from_slice(&[16, 10]);
        d.global_strides[0] = 64;
        d.box_dims[..2].copy_from_slice(&[8, 1]);
        let e = d.access_elements(&TmaAccess::Rows4 { x: 12, rows: [3, 0, 12, 9] }).unwrap();
        assert_eq!(e.len(), 32);
        assert_eq!(e[0].global_bit, Some((0x2000 + 3 * 64 + 12 * 4) * 8));
        assert_eq!(e[4].global_bit, None);
        assert_eq!(e[8].global_bit, Some((0x2000 + 12 * 4) * 8));
        assert!(e[16..24].iter().all(|x| x.global_bit.is_none()));
        assert_eq!(e[31].smem_bit, 31 * 32);
        assert_eq!(d.access_smem_bytes(true), 128);
    }

    #[test]
    fn padded_fp4_groups_take_16_byte_slots() {
        let mut d = TmaDesc::new(TensorMapDataType::f416u4_align16b, 2);
        d.global_address = 0x4000;
        d.global_dims[..2].copy_from_slice(&[256, 4]);
        d.global_strides[0] = 128;
        d.box_dims[..2].copy_from_slice(&[128, 2]);
        d.validate().unwrap();
        let e = d.access_elements(&TmaAccess::Tile { start: &[128, 1] }).unwrap();
        assert_eq!(e[0].global_bit, Some((0x4000 + 128 + 64) * 8));
        assert_eq!(e[1].global_bit, Some((0x4000 + 128 + 64) * 8 + 4));
        assert_eq!(e[15].smem_bit, 60);
        assert_eq!(e[16].smem_bit, 128);
        assert_eq!(d.smem_box_bytes(), 256);
        assert_eq!(d.box_bytes(), 128);
        let err = d.access_elements(&TmaAccess::Tile { start: &[64, 0] }).unwrap_err();
        assert!(err.contains("not a multiple of 128"), "{err}");
    }
}

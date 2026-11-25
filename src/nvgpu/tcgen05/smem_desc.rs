// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm100_desc.hpp (UMMA::SmemDescriptor's bit fields,
//   UMMA::LayoutType, and the canonical-layout checks and LBO/SBO choice of
//   UMMA::make_umma_desc),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! UMMA shared-memory descriptors: the
//! 64-bit operand descriptors of `tcgen05.mma` (CuTe DSL's
//! `tcgen05.make_umma_smem_desc`), split into the fields fixed when a kernel
//! is built ([`umma_desc_fields`]) and the start address added at run time,
//! and the decoder ([`UmmaOperandMap`]) that maps a descriptor back to the
//! shared-memory address of each operand element.
//!
//! The descriptor (PTX ISA, "tcgen05 shared memory descriptor"): the start
//! address in 16-byte units in bits 0-13, the leading byte offset (LBO) in
//! 16-byte units in bits 16-29, the stride byte offset (SBO) in 16-byte units
//! in bits 32-45, the version (1) in bits 46-47, the base offset in bits
//! 49-51, the LBO mode in bit 52 and the layout (swizzle) type in bits 61-63.
//! On sm_107 the start address and LBO take 15 bits (0-14 and 16-30).

use crate::Arch;
use crate::layout::{Layout, Swizzle};
use crate::nvgpu::OperandMajorMode;
use crate::tensor::{AddressSpace, Ptr};
use crate::value::{Numeric, Value};

/// The swizzle a UMMA descriptor names (CuTe C++'s `UMMA::LayoutType`,
/// bits 61-63): none, the 128-byte swizzle with 32-byte atomicity, and the
/// 128-, 64- and 32-byte swizzles with 16-byte atomicity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UmmaLayoutType {
    SwizzleNone = 0,
    Swizzle128BBase32B = 1,
    Swizzle128B = 2,
    Swizzle64B = 4,
    Swizzle32B = 6,
}

impl UmmaLayoutType {
    pub const ALL: [UmmaLayoutType; 5] = [
        UmmaLayoutType::SwizzleNone,
        UmmaLayoutType::Swizzle128BBase32B,
        UmmaLayoutType::Swizzle128B,
        UmmaLayoutType::Swizzle64B,
        UmmaLayoutType::Swizzle32B,
    ];

    /// The type of a 3-bit field value, if it names one.
    pub fn from_bits(v: u64) -> Option<UmmaLayoutType> {
        UmmaLayoutType::ALL.into_iter().find(|t| *t as u64 == v)
    }

    /// The type whose swizzle on byte addresses is `sw`: `Swizzle<0,_,_>`
    /// (none), `Swizzle<1,4,3>`, `Swizzle<2,4,3>`, `Swizzle<3,4,3>` or
    /// `Swizzle<2,5,2>`.
    pub fn from_byte_swizzle(sw: Swizzle) -> Option<UmmaLayoutType> {
        match (sw.bits, sw.base, sw.shift) {
            (0, _, _) => Some(UmmaLayoutType::SwizzleNone),
            (1, 4, 3) => Some(UmmaLayoutType::Swizzle32B),
            (2, 4, 3) => Some(UmmaLayoutType::Swizzle64B),
            (3, 4, 3) => Some(UmmaLayoutType::Swizzle128B),
            (2, 5, 2) => Some(UmmaLayoutType::Swizzle128BBase32B),
            _ => None,
        }
    }

    /// The swizzle on byte addresses (`Swizzle<0,4,3>` for none).
    pub fn byte_swizzle(self) -> Swizzle {
        match self {
            UmmaLayoutType::SwizzleNone => Swizzle { bits: 0, base: 4, shift: 3 },
            UmmaLayoutType::Swizzle32B => Swizzle { bits: 1, base: 4, shift: 3 },
            UmmaLayoutType::Swizzle64B => Swizzle { bits: 2, base: 4, shift: 3 },
            UmmaLayoutType::Swizzle128B => Swizzle { bits: 3, base: 4, shift: 3 },
            UmmaLayoutType::Swizzle128BBase32B => Swizzle { bits: 2, base: 5, shift: 2 },
        }
    }

    /// 16-byte units in a swizzle row along MN (MN-major) or in a row of
    /// the 8-row block (K-major): 1, 2, 4 or 8.
    pub fn row_units(self) -> usize {
        match self {
            UmmaLayoutType::SwizzleNone => 1,
            UmmaLayoutType::Swizzle32B => 2,
            UmmaLayoutType::Swizzle64B => 4,
            UmmaLayoutType::Swizzle128B | UmmaLayoutType::Swizzle128BBase32B => 8,
        }
    }

    /// Rows of the swizzle atom along K of an MN-major operand: 4 for the
    /// 32-byte-atomicity swizzle, 8 otherwise.
    pub fn k_rows(self) -> usize {
        if self == UmmaLayoutType::Swizzle128BBase32B { 4 } else { 8 }
    }

    /// Bytes after which the swizzle pattern repeats (and to which a
    /// swizzled operand's buffer must be aligned): 8 rows of 32, 64 or 128
    /// bytes, or 4 rows of 128 bytes for `Swizzle128BBase32B`; 16 for none.
    pub fn repeat_bytes(self) -> usize {
        if self == UmmaLayoutType::SwizzleNone { 16 } else { self.byte_swizzle().period() }
    }
}

/// The width of the start-address and LBO fields: 14 bits, or 15 on
/// sm_107 (CuTe C++'s `CUTLASS_ARCH_MMA_SM107` descriptor, which CuTe DSL
/// builds with `sm107_make_umma_smem_desc`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum UmmaDescEncoding {
    #[default]
    Sm100,
    Sm107,
}

impl UmmaDescEncoding {
    /// The encoding for a target: `Sm107` for sm_107, sm_107a and sm_107f.
    pub fn for_arch(arch: Arch) -> UmmaDescEncoding {
        if arch.version() == (10, 7) { UmmaDescEncoding::Sm107 } else { UmmaDescEncoding::Sm100 }
    }

    /// Bits of the start-address and LBO fields.
    pub fn address_bits(self) -> u32 {
        match self {
            UmmaDescEncoding::Sm100 => 14,
            UmmaDescEncoding::Sm107 => 15,
        }
    }

    fn mask(self) -> u64 {
        (1u64 << self.address_bits()) - 1
    }
}

/// A UMMA shared-memory descriptor's fields (CuTe C++'s
/// `UMMA::SmemDescriptor`). Addresses and offsets are in 16-byte units.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SmemDescriptor {
    /// The operand's shared-memory address `>> 4`.
    pub start_address: u32,
    /// The leading byte offset `>> 4`.
    pub leading_byte_offset: u32,
    /// The stride byte offset `>> 4`.
    pub stride_byte_offset: u32,
    /// 1 on Blackwell.
    pub version: u8,
    /// The swizzle pattern's start row: 0 when the operand's buffer is
    /// aligned to the pattern's repeat, which cute-rs requires.
    pub base_offset: u8,
    /// The LBO mode bit (0: CuTe's default, relative offsets).
    pub lbo_mode: bool,
    pub layout_type: UmmaLayoutType,
}

impl SmemDescriptor {
    /// The 64-bit descriptor. Panics if a field does not fit.
    #[track_caller]
    pub fn encode(&self, enc: UmmaDescEncoding) -> u64 {
        let wide = enc.mask();
        let fit = |name: &str, v: u64, mask: u64| {
            assert!(v <= mask, "UMMA descriptor: the {name} {v} does not fit its {}-bit field", mask.count_ones());
            v
        };
        // The bits are placed by the public
        // encoder (`prims::Tcgen05SmemDesc`).
        crate::prims::Tcgen05SmemDesc::pack(
            fit("start address", self.start_address as u64, wide),
            fit("leading byte offset", self.leading_byte_offset as u64, wide),
            fit("stride byte offset", self.stride_byte_offset as u64, 0x3fff),
            fit("version", self.version as u64, 3),
            fit("base offset", self.base_offset as u64, 7),
            self.lbo_mode,
            false,
            self.layout_type,
        )
    }

    /// The fields of a 64-bit descriptor, or why it is not one (a reserved
    /// layout type).
    pub fn decode(desc: u64, enc: UmmaDescEncoding) -> Result<SmemDescriptor, String> {
        let layout_type = UmmaLayoutType::from_bits(desc >> 61)
            .ok_or_else(|| format!("UMMA descriptor {desc:#018x} has the reserved layout type {}", desc >> 61))?;
        Ok(SmemDescriptor {
            start_address: (desc & enc.mask()) as u32,
            leading_byte_offset: ((desc >> 16) & enc.mask()) as u32,
            stride_byte_offset: ((desc >> 32) & 0x3fff) as u32,
            version: ((desc >> 46) & 3) as u8,
            base_offset: ((desc >> 49) & 7) as u8,
            lbo_mode: (desc >> 52) & 1 == 1,
            layout_type,
        })
    }

    /// This descriptor with the operand at byte address `addr` of shared
    /// memory (which must be a multiple of 16).
    #[track_caller]
    pub fn with_start_address(self, addr: u32) -> SmemDescriptor {
        assert!(addr.is_multiple_of(16), "UMMA descriptor: the start address {addr:#x} is not 16-byte aligned");
        SmemDescriptor { start_address: addr >> 4, ..self }
    }
}

/// The fields of the UMMA descriptor of an operand tile fixed when the
/// kernel is built: everything but the start address (CuTe C++'s
/// `UMMA::make_umma_desc<Major>`). `tile` maps `(mn, k)` (a rank-2 layout,
/// or a rank-1 layout whose mode is that) to element offsets, without the
/// swizzle, which `layout_type` names; `bits` is the element width.
///
/// The tile must be one of the canonical forms, with `T = 128 / bits`
/// elements to a 16-byte unit, `W` the swizzle's units per row (1, 2, 4 or
/// 8) and `R` its rows along K (8, or 4 for `Swizzle128BBase32B`):
///
/// * K-major, 32 bytes of K (`2T` elements): `((8,m),(T,2)):((W T, SBO),(1,
///   L))` with `L = LBO` unswizzled and `T` swizzled (`Swizzle128BBase32B` is
///   MN-major only). The swizzled forms have no LBO; CuTe writes 1. With the
///   sm_107 encoding also 64 bytes of K, `(T,4)`.
/// * MN-major: `((T,W,m),(R,k)):((1,T,A),(W T,B))`, where unswizzled `A` is
///   the SBO and `B` the LBO, and swizzled the other way round.
///
/// A stride of a mode of extent 1 (`m` or `k` of 1) is unused and written
/// as 0. Returns the descriptor with start address 0, or why the tile is
/// not canonical.
pub fn umma_desc_fields(
    tile: &Layout,
    major: OperandMajorMode,
    layout_type: UmmaLayoutType,
    bits: usize,
    enc: UmmaDescEncoding,
) -> Result<SmemDescriptor, String> {
    let tile = if tile.rank() == 1 && tile.mode(0).rank() == 2 { tile.mode(0) } else { tile.clone() };
    if tile.rank() != 2 {
        return Err(format!("the tile {tile} is not a rank-2 (MN, K) layout"));
    }
    if !tile.is_static() {
        return Err(format!("the tile {tile} has runtime extents or strides"));
    }
    if !(bits.is_power_of_two() && (4..=128).contains(&bits)) {
        return Err(format!("{bits}-bit elements, where UMMA operands take 4, 8, 16, 32, 64 or 128 bits"));
    }
    let (mn, k) = (tile.mode(0).size(), tile.mode(1).size());
    let f = |m: usize, kk: usize| tile.call(m + mn * kk);
    let t = 128 / bits;
    let w = layout_type.row_units();
    let swizzled = layout_type != UmmaLayoutType::SwizzleNone;
    type Canonical = Box<dyn Fn(usize, usize) -> usize>;
    // Strides in elements, read off the tile, and the canonical form they define.
    let (lbo, sbo, canonical, form): (usize, usize, Canonical, String) = match major {
        OperandMajorMode::K => {
            if layout_type == UmmaLayoutType::Swizzle128BBase32B {
                return Err("the 128-byte swizzle with 32-byte atomicity is only for MN-major operands".into());
            }
            // The sm_103a K = 96 FP4 MMAs read 48 bytes of K (3 units).
            // The Rubin K = 64 (8-bit) and K = 128 (packed FP4) MMAs read 64
            // bytes (4 units), with the sm_107 encoding.
            let k_ok = k == 2 * t || (bits == 4 && k == 3 * t) || (enc == UmmaDescEncoding::Sm107 && k == 4 * t);
            if !k_ok || !mn.is_multiple_of(8) {
                return Err(format!(
                    "a K-major tile is (8 m) x {} elements of {bits} bits (32 bytes of K), but this one is {mn} x {k}",
                    2 * t
                ));
            }
            let sbo = if mn > 8 { f(8, 0) } else { 0 };
            let l = if swizzled { t } else { f(0, t) };
            let canon = move |m: usize, kk: usize| (m % 8) * w * t + (m / 8) * sbo + (kk / t) * l + kk % t;
            let form = if swizzled {
                format!("((8,{}),({t},{})):(({},{sbo}),(1,{t}))", mn / 8, k / t, w * t)
            } else {
                format!("((8,{}),({t},{})):(({t},{sbo}),(1,{l}))", mn / 8, k / t)
            };
            // Swizzled K-major has no LBO: CuTe writes 1 unit.
            (if swizzled { 128 / bits } else { l }, sbo, Box::new(canon), form)
        }
        OperandMajorMode::MN => {
            // Measured on a B200 (sm_100a): an MN-major operand of 32-bit elements is read
            // correctly only with the 32-byte-atomicity swizzle, and that swizzle only with
            // 32-bit elements; the other pairings give zeros. CuTe's own selector
            // (`sm100_smem_selector`) pairs them the same way.
            let base32 = layout_type == UmmaLayoutType::Swizzle128BBase32B;
            if base32 != (bits == 32) {
                return Err(if base32 {
                    format!(
                        "the 128-byte swizzle with 32-byte atomicity (MN_SW128_32B) is for MN-major operands of \
                         32-bit elements, and these are {bits}-bit: the hardware reads other widths wrongly"
                    )
                } else {
                    format!(
                        "an MN-major operand of 32-bit elements needs the 128-byte swizzle with 32-byte atomicity \
                         (MN_SW128_32B), not {layout_type:?}: the hardware reads the other layouts wrongly"
                    )
                });
            }
            let r = layout_type.k_rows();
            let block = w * t;
            if !mn.is_multiple_of(block) || !k.is_multiple_of(r) {
                return Err(format!(
                    "an MN-major tile with this swizzle is a multiple of {block} (MN) x {r} (K) elements of {bits} \
                     bits, but this one is {mn} x {k}"
                ));
            }
            let a = if mn > block { f(block, 0) } else { 0 };
            let b = if k > r { f(0, r) } else { 0 };
            let canon = move |m: usize, kk: usize| m % block + (m / block) * a + (kk % r) * block + (kk / r) * b;
            let form = format!("(({t},{w},{}),({r},{})):((1,{t},{a}),({block},{b}))", mn / block, k / r);
            let (lbo, sbo) = if swizzled { (a, b) } else { (b, a) };
            (lbo, sbo, Box::new(canon), form)
        }
    };
    for kk in 0..k {
        for m in 0..mn {
            let (got, want) = (f(m, kk), canonical(m, kk));
            if got != want {
                return Err(format!(
                    "element ({m}, {kk}) of the tile {tile} is at offset {got}, but the canonical {major:?}-major form \
                     with {layout_type:?} puts it at {want}: expected {form} (in elements)"
                ));
            }
        }
    }
    let unit = |name: &str, elems: usize, field_bits: u32| -> Result<u32, String> {
        let bytes = elems * bits / 8;
        if !bytes.is_multiple_of(16) || bytes >> 4 >= 1 << field_bits {
            return Err(format!("a {name} of {bytes} bytes, not a multiple of 16 below {} KiB", 1 << (field_bits - 6)));
        }
        Ok((bytes >> 4) as u32)
    };
    Ok(SmemDescriptor {
        start_address: 0,
        leading_byte_offset: unit("leading byte offset", lbo, enc.address_bits())?,
        stride_byte_offset: unit("stride byte offset", sbo, 14)?,
        version: 1,
        base_offset: 0,
        lbo_mode: false,
        layout_type,
    })
}

/// The UMMA shared-memory descriptor of an MMA operand tile (CuTe DSL's
/// `tcgen05.make_umma_smem_desc(src, layout, major, next_src=None)`): `src`
/// points at the tile's first element in shared memory and carries the
/// tile's swizzle (a pointer from [`recast_ptr`](crate::recast_ptr) with the
/// layout atom's byte swizzle), and `layout` is the tile's `(MN, K)` layout
/// without the swizzle. Everything but the start address is fixed when the
/// kernel is built ([`umma_desc_fields`]); the start address, `src`'s
/// address `>> 4`, is ORed in at run time. The encoding follows the target
/// (15-bit fields on sm_107).
///
/// `next_src` (CuTe DSL's circular double-buffering form, a closed MLIR op)
/// is not supported: its effect on the descriptor is not documented in the
/// PTX ISA or CuTe C++. Passing one panics.
///
/// Panics unless `src` is a shared-memory pointer with one of the five UMMA
/// swizzles (or none) whose buffer is aligned to the swizzle's repeat, and
/// `layout` is a canonical form for `major` (naming the expected form).
#[track_caller]
pub fn make_umma_smem_desc<T: Numeric>(
    src: &Ptr<T>,
    layout: &Layout,
    major: OperandMajorMode,
    next_src: Option<&Ptr<T>>,
) -> Value<u64> {
    assert!(
        next_src.is_none(),
        "make_umma_smem_desc: next_src (circular double buffering) is not supported: its effect on the descriptor is \
         not documented outside CuTe DSL's closed MLIR op"
    );
    assert!(
        src.memspace() == AddressSpace::Smem,
        "make_umma_smem_desc: the operand pointer is in {:?}, not shared memory",
        src.memspace()
    );
    let sw = src.swizzle().unwrap_or(Swizzle { bits: 0, base: 4, shift: 3 });
    let layout_type = UmmaLayoutType::from_byte_swizzle(sw).unwrap_or_else(|| {
        panic!(
            "make_umma_smem_desc: the pointer's swizzle {sw} (on bytes) is none of UMMA's: Swizzle<0,4,3>, \
             Swizzle<1,4,3>, Swizzle<2,4,3>, Swizzle<3,4,3> or Swizzle<2,5,2>"
        )
    });
    let enc = crate::builder::target_arch().map(UmmaDescEncoding::for_arch).unwrap_or_default();
    let fields = umma_desc_fields(layout, major, layout_type, T::DTYPE.bits(), enc)
        .unwrap_or_else(|why| panic!("make_umma_smem_desc: the operand tile is not a canonical UMMA layout: {why}"));
    let align = crate::builder::with(|b| b.buffer(src.buffer).align);
    let repeat = layout_type.repeat_bytes();
    assert!(
        align >= repeat,
        "make_umma_smem_desc: the operand's buffer is aligned to {align} bytes, but {layout_type:?} needs {repeat} \
         (the descriptor's base offset is 0)"
    );
    let constant = fields.encode(enc);
    let addr: Value<u32> = src.toint_i32().to::<u32>();
    crate::prims::Tcgen05SmemDesc::from_bits(constant).with_start_address_bits(addr, enc.address_bits()).value()
}

/// CuTe DSL's `tcgen05.helpers.int_to_smem_descriptor`: an Int64 as a
/// descriptor (cute-rs keeps descriptors as `Uint64`s; the bits are kept).
#[track_caller]
pub fn int_to_smem_descriptor(v: Value<i64>) -> Value<u64> {
    v.to::<u64>()
}

/// CuTe DSL's `tcgen05.helpers.smem_descriptor_to_int`: a descriptor's
/// bits as an Int64.
#[track_caller]
pub fn smem_descriptor_to_int(desc: Value<u64>) -> Value<i64> {
    desc.to::<i64>()
}

/// Where each element of a UMMA operand lives in shared memory, decoded
/// from its descriptor as the PTX ISA defines the canonical layouts and
/// swizzles (independently of cute-rs's layout algebra, so that the
/// simulator fetching operands through it checks the encoder): the
/// operand is `mn x k` elements of `bits` bits, `major`-major.
#[derive(Clone, Debug)]
pub struct UmmaOperandMap {
    pub desc: SmemDescriptor,
    pub major: OperandMajorMode,
    pub bits: usize,
    pub mn: usize,
    pub k: usize,
}

impl UmmaOperandMap {
    /// Decode `desc`, or say why it cannot be read (the LBO mode, a base
    /// offset other than 0, version other than 1, or a K-major
    /// `Swizzle128BBase32B`).
    pub fn new(
        desc: u64,
        enc: UmmaDescEncoding,
        major: OperandMajorMode,
        bits: usize,
        mn: usize,
        k: usize,
    ) -> Result<UmmaOperandMap, String> {
        let d = SmemDescriptor::decode(desc, enc)?;
        if d.version != 1 {
            return Err(format!("UMMA descriptor {desc:#018x}: version {}, not 1", d.version));
        }
        if d.base_offset != 0 || d.lbo_mode {
            return Err(format!("UMMA descriptor {desc:#018x}: a base offset or LBO mode, which are not modelled"));
        }
        if major == OperandMajorMode::K && d.layout_type == UmmaLayoutType::Swizzle128BBase32B {
            return Err("the 128-byte swizzle with 32-byte atomicity is only for MN-major operands".into());
        }
        if !(bits.is_power_of_two() && (4..=128).contains(&bits)) {
            return Err(format!("{bits}-bit operand elements"));
        }
        Ok(UmmaOperandMap { desc: d, major, bits, mn, k })
    }

    /// The bit address in shared memory of element `(m, kk)`.
    pub fn bit_address(&self, m: usize, kk: usize) -> u64 {
        assert!(m < self.mn && kk < self.k, "UmmaOperandMap: ({m}, {kk}) is outside {} x {}", self.mn, self.k);
        let d = &self.desc;
        let ty = d.layout_type;
        // Elements per 16-byte unit; byte offsets as elements.
        let t = 128 / self.bits;
        let units = |v: u32| v as usize * 128 / self.bits;
        let (lbo, sbo) = (units(d.leading_byte_offset), units(d.stride_byte_offset));
        let swizzled = ty != UmmaLayoutType::SwizzleNone;
        let w = match ty {
            UmmaLayoutType::SwizzleNone => 1,
            UmmaLayoutType::Swizzle32B => 2,
            UmmaLayoutType::Swizzle64B => 4,
            UmmaLayoutType::Swizzle128B | UmmaLayoutType::Swizzle128BBase32B => 8,
        };
        // The PTX ISA's canonical layouts, in elements (T a 16-byte unit):
        //   K-major,  none:     ((8,m),(T,2k)):((T,SBO),(1,LBO))
        //   K-major,  swizzled: ((8,m),(T,2k)):((W T,SBO),(1,T))
        //   MN-major, none:     ((T,1,m),(8,k)):((1,T,SBO),(T,LBO))
        //   MN-major, swizzled: ((T,W,m),(R,k)):((1,T,LBO),(W T,SBO)), R = 4 for 128B_BASE32B, else 8
        let elem = match self.major {
            OperandMajorMode::K => {
                let row = if swizzled { w * t } else { t };
                let unit_step = if swizzled { t } else { lbo };
                (m % 8) * row + (m / 8) * sbo + (kk % t) + (kk / t) * unit_step
            }
            OperandMajorMode::MN => {
                let r = if ty == UmmaLayoutType::Swizzle128BBase32B { 4 } else { 8 };
                let (mn_step, k_step) = if swizzled { (lbo, sbo) } else { (sbo, lbo) };
                (m % t) + ((m / t) % w) * t + (m / (t * w)) * mn_step + (kk % r) * w * t + (kk / r) * k_step
            }
        };
        let bits = (d.start_address as u64) * 128 + (elem * self.bits) as u64;
        let byte = swizzle_address(ty, bits / 8);
        byte * 8 + bits % 8
    }

    /// The byte address of element `(m, kk)` (elements of 8 bits or more).
    pub fn byte_address(&self, m: usize, kk: usize) -> u64 {
        assert!(
            self.bits >= 8,
            "UmmaOperandMap::byte_address: {}-bit elements share bytes; use bit_address",
            self.bits
        );
        self.bit_address(m, kk) / 8
    }
}

/// The PTX ISA's swizzle of a shared-memory byte address: the 16-byte
/// chunk index (bits 4-6) XORed with the row (bits 7-9), over 1, 2 or 3
/// bits for the 32-, 64- and 128-byte swizzles; for the 128-byte swizzle
/// with 32-byte atomicity the 32-byte chunk (bits 5-6) with bits 7-8.
fn swizzle_address(ty: UmmaLayoutType, addr: u64) -> u64 {
    let (chunk_lo, n) = match ty {
        UmmaLayoutType::SwizzleNone => return addr,
        UmmaLayoutType::Swizzle32B => (4, 1),
        UmmaLayoutType::Swizzle64B => (4, 2),
        UmmaLayoutType::Swizzle128B => (4, 3),
        UmmaLayoutType::Swizzle128BBase32B => (5, 2),
    };
    let mask = (1u64 << n) - 1;
    let row = (addr >> 7) & mask;
    addr ^ (row << chunk_lo)
}

//! TMA descriptors (tensor maps) as plain Rust: the concrete descriptor ([`TmaDesc`]) with
//! the checks `cuTensorMapEncodeTiled` applies, the simulator's own 128-byte encoding of it,
//! the traversal of one box and where its bytes land in shared memory (with the
//! address-based 32B/64B/128B swizzles), and the build-time description of a descriptor
//! whose extents and strides may be launch values ([`TmaDescSpec`]).
//!
//! A tensor map describes a tensor of up to five dimensions in global memory in TMA's
//! order: dimension 0 is the innermost, contiguous one, and dimension `i >= 1` has a byte
//! stride. A box of `box_dims` elements (every `element_strides[i]`-th along dimension
//! `i`) is copied between the tensor and shared memory, where it is dense, dimension 0
//! fastest; out-of-bounds elements are filled (zeros or a NaN) on loads and skipped on
//! stores.
//!
//! The kernel receives a descriptor as a 128-byte, 64-byte-aligned grid-constant parameter.
//! On the GPU it holds the driver's opaque `CUtensorMap`; in the simulator it holds
//! [`TmaDesc::to_sim_bytes`], which TMA instructions decode. Descriptors reach kernels two
//! ways: a hidden parameter tied to a tensor parameter ([`tma_descriptor`]), encoded at
//! each launch from the tensor's address and the launch's scalar arguments, or a parameter
//! the host binds explicitly ([`param_tensormap`] with a host-built
//! [`TensorMap`](crate::tensor_map::TensorMap)).

use std::fmt;
use std::panic::Location;

use crate::builder::{self, user_panic};
use crate::ir::{BufferId, DType, Param, ValueId};
use crate::layout::{IndexExpr, StrideLeaf};
use crate::tensor::{Engine, Tensor};
use crate::value::{Int64, Numeric, Value};

// The access modes.
mod access;
pub use access::{TmaAccess, TmaElement};

/// The size of a descriptor (`CUtensorMap`): 128 bytes.
pub const TENSOR_MAP_BYTES: usize = 128;
/// The alignment of a descriptor: 64 bytes.
pub const TENSOR_MAP_ALIGN: usize = 64;
/// The most dimensions a tiled tensor map has.
pub const MAX_RANK: usize = 5;
/// The largest box extent.
pub const MAX_BOX_DIM: u32 = 256;
/// The largest traversal (element) stride.
pub const MAX_ELEMENT_STRIDE: u32 = 8;

macro_rules! tensor_map_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $v:ident = $n:expr, $s:expr;)* }) => {
        $(#[$m])*
        // The variants are named after the CUDA enumerators.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
        #[repr(u32)]
        pub enum $name {
            #[default]
            $($(#[$vm])* $v = $n,)*
        }

        impl $name {
            /// Every value, in the driver's order.
            pub const ALL: &'static [$name] = &[$($name::$v),*];

            /// The driver's enum value.
            pub fn code(self) -> u32 {
                self as u32
            }

            /// The value with the driver's code `code`.
            pub fn from_code(code: u32) -> Option<$name> {
                Self::ALL.iter().copied().find(|v| v.code() == code)
            }

            /// The value's name (`s128b`, `float16`, ...).
            pub fn name(self) -> &'static str {
                match self { $($name::$v => $s,)* }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.name())
            }
        }
    };
}

tensor_map_enum! {
    /// The element encoding of a tensor map (`CUtensorMapDataType`; CuTe DSL's
    /// `TensorMapDataType`). The three sub-byte encodings
    /// move FP4 and FP6 values in groups of 16: `f416u4_align8b` packed as in global
    /// memory (PTX's `.b4x16`), `f416u4_align16b` and `f416u6_align16b` each group padded to
    /// a 16-byte slot of shared memory (`.b4x16_p64`, `.b6x16_p32`).
    TensorMapDataType {
        uint8 = 0, "uint8";
        uint16 = 1, "uint16";
        uint32 = 2, "uint32";
        int32 = 3, "int32";
        uint64 = 4, "uint64";
        int64 = 5, "int64";
        float16 = 6, "float16";
        float32 = 7, "float32";
        float64 = 8, "float64";
        bfloat16 = 9, "bfloat16";
        float32_ftz = 10, "float32_ftz";
        tfloat32 = 11, "tfloat32";
        tfloat32_ftz = 12, "tfloat32_ftz";
        f416u4_align8b = 13, "f416u4_align8b";
        f416u4_align16b = 14, "f416u4_align16b";
        f416u6_align16b = 15, "f416u6_align16b";
    }
}

tensor_map_enum! {
    /// The interleaved layout a tensor map addresses (`CUtensorMapInterleave`; CuTe DSL's
    /// `TensorMapInterleave`): none, or channel groups of 16 or 32 bytes (`NC/8HWC8`).
    TensorMapInterleave {
        none = 0, "none";
        i16b = 1, "i16b";
        i32b = 2, "i32b";
    }
}

tensor_map_enum! {
    /// The shared-memory swizzle of a tensor map (`CUtensorMapSwizzle`; CuTe DSL's
    /// `TensorMapSwizzle`). Of the SM100 `s128b_atom_*` modes, `s128b_atom_32b` (the
    /// 128-byte swizzle of 32-byte atoms, `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`) is supported on sm_100 and newer; the other two are
    /// refused.
    TensorMapSwizzle {
        none = 0, "none";
        s32b = 1, "s32b";
        s64b = 2, "s64b";
        s128b = 3, "s128b";
        s128b_atom_32b = 4, "s128b_atom_32b";
        s128b_atom_32b_flip_8b = 5, "s128b_atom_32b_flip_8b";
        s128b_atom_64b = 6, "s128b_atom_64b";
    }
}

tensor_map_enum! {
    /// The L2 fetch granularity of a tensor map (`CUtensorMapL2promotion`; CuTe DSL's
    /// `TensorMapL2Promotion`): a performance hint the simulator ignores.
    TensorMapL2Promotion {
        none = 0, "none";
        l2_64b = 1, "l2_64b";
        l2_128b = 2, "l2_128b";
        l2_256b = 3, "l2_256b";
    }
}

tensor_map_enum! {
    /// What fills out-of-bounds elements of a load (`CUtensorMapFloatOOBfill`; CuTe DSL's
    /// `TensorMapFloatOOBFill`): zeros, or (floating-point encodings only) the NaN that
    /// asks the tensor cores to treat the product as zero.
    TensorMapFloatOOBFill {
        none = 0, "none";
        nan_request_zero_fma = 1, "nan_request_zero_fma";
    }
}

impl TensorMapDataType {
    /// Bits of one element: 4 or 6 for the packed encodings.
    pub fn bit_width(self) -> u32 {
        use TensorMapDataType::*;
        match self {
            uint8 => 8,
            uint16 | float16 | bfloat16 => 16,
            uint32 | int32 | float32 | float32_ftz | tfloat32 | tfloat32_ftz => 32,
            uint64 | int64 | float64 => 64,
            f416u4_align8b | f416u4_align16b => 4,
            f416u6_align16b => 6,
        }
    }

    /// For the sub-byte encodings, the shared-memory bytes of one group of 16 values: 8
    /// for the packed `f416u4_align8b`, 16 for the padded `f416u4_align16b` (8 bytes of
    /// values, 8 of padding) and `f416u6_align16b` (12 and 4); `None` for whole-byte
    /// encodings.
    pub fn group_slot_bytes(self) -> Option<u64> {
        use TensorMapDataType::*;
        match self {
            f416u4_align8b => Some(8),
            f416u4_align16b | f416u6_align16b => Some(16),
            _ => None,
        }
    }

    /// One of the padded sub-byte encodings (`f416u4_align16b`, `f416u6_align16b`).
    pub fn is_padded(self) -> bool {
        self.group_slot_bytes() == Some(16)
    }

    /// Bytes of one element (whole-byte encodings).
    pub fn bytes(self) -> u64 {
        u64::from(self.bit_width() / 8)
    }

    /// One of the packed sub-byte encodings.
    pub fn is_packed(self) -> bool {
        self.bit_width() < 8
    }

    /// A floating-point encoding (which may take the NaN out-of-bounds fill).
    pub fn is_float(self) -> bool {
        use TensorMapDataType::*;
        matches!(self, float16 | float32 | float64 | bfloat16 | float32_ftz | tfloat32 | tfloat32_ftz)
    }

    /// The encoding a tensor of `dtype` uses by default: the type itself where TMA has
    /// it, `uint8` for the other 8-bit types (FP8, `i8`), `uint16` for `i16`, the packed
    /// `f416u4_align8b` for FP4 and the padded `f416u6_align16b` for FP6 (C++ CuTe's
    /// `TMA::to_CUtensorMapDataType`), or `None` for types TMA does not move.
    pub fn for_dtype(dtype: DType) -> Option<TensorMapDataType> {
        use TensorMapDataType::*;
        Some(match dtype {
            DType::F16 => float16,
            DType::BF16 => bfloat16,
            DType::F32 => float32,
            DType::F64 => float64,
            DType::TF32 => tfloat32,
            DType::I32 => int32,
            DType::U32 => uint32,
            DType::I64 => int64,
            DType::U64 => uint64,
            DType::I16 | DType::U16 => uint16,
            // Packed Int4 moves as the packed 4-bit encoding too.
            DType::F4E2M1FN | DType::I4 => f416u4_align8b,
            DType::F6E2M3FN | DType::F6E3M2FN => f416u6_align16b,
            t if t.bits() == 8 && !t.is_sub_byte() => uint8,
            _ => return None,
        })
    }
}

impl TensorMapSwizzle {
    /// The span the 16-byte chunks are permuted within (32, 64 or 128 bytes), or `None`
    /// without a swizzle.
    pub fn span_bytes(self) -> Option<u64> {
        match self {
            TensorMapSwizzle::none => None,
            TensorMapSwizzle::s32b => Some(32),
            TensorMapSwizzle::s64b => Some(64),
            _ => Some(128),
        }
    }

    /// The SM100 atom modes (sm_100 and newer).
    pub fn is_atom_mode(self) -> bool {
        matches!(
            self,
            TensorMapSwizzle::s128b_atom_32b
                | TensorMapSwizzle::s128b_atom_32b_flip_8b
                | TensorMapSwizzle::s128b_atom_64b
        )
    }

    /// The SM100 atom modes cute-rs does not support: `s128b_atom_32b_flip_8b` and
    /// `s128b_atom_64b`.
    pub fn is_unsupported_atom_mode(self) -> bool {
        self.is_atom_mode() && self != TensorMapSwizzle::s128b_atom_32b
    }

    /// The shared-memory byte address `addr` of a box byte after this swizzle (PTX ISA,
    /// tensor-map swizzle modes): the swizzle is address-based, so the 16-byte chunk index
    /// (address bits 4 and up, one, two or three of them for 32B, 64B and 128B) is XORed
    /// with the same number of address bits from bit 7 (the 128-byte row within a
    /// 256-, 512- or 1024-byte pattern). CuTe's `Swizzle<1,4,3>`, `<2,4,3>`, `<3,4,3>` on
    /// bytes. `s128b_atom_32b` permutes the four 32-byte atoms of each 128-byte row
    /// (address bits 5-6) with bits 7-8, CuTe's `Swizzle<2,5,2>` on bytes.
    pub fn apply(self, addr: u64) -> u64 {
        if self == TensorMapSwizzle::s128b_atom_32b {
            return addr ^ (((addr >> 7) & 3) << 5);
        }
        let bits = match self {
            TensorMapSwizzle::none => return addr,
            TensorMapSwizzle::s32b => 1,
            TensorMapSwizzle::s64b => 2,
            _ => 3,
        };
        let mask = (1u64 << bits) - 1;
        addr ^ (((addr >> 7) & mask) << 4)
    }

    /// The bytes after which the swizzle pattern repeats (256, 512 or 1024; 512 for
    /// `s128b_atom_32b`), or 16 without a swizzle. A box placed at a multiple of it starts
    /// the pattern at its first row.
    pub fn pattern_bytes(self) -> u64 {
        match self {
            TensorMapSwizzle::none => 16,
            TensorMapSwizzle::s32b => 256,
            TensorMapSwizzle::s64b | TensorMapSwizzle::s128b_atom_32b => 512,
            _ => 1024,
        }
    }

    /// CuTe's swizzle on byte offsets for this mode (`Swizzle<B,4,3>`; `Swizzle<2,5,2>`
    /// for `s128b_atom_32b`), for building a shared-memory layout that matches.
    pub fn to_byte_swizzle(self) -> crate::layout::Swizzle {
        match self {
            TensorMapSwizzle::none => crate::layout::Swizzle::new(0, 4, 3),
            TensorMapSwizzle::s128b_atom_32b => crate::layout::Swizzle::new(2, 5, 2),
            TensorMapSwizzle::s32b => crate::layout::Swizzle::new(1, 4, 3),
            TensorMapSwizzle::s64b => crate::layout::Swizzle::new(2, 4, 3),
            _ => crate::layout::Swizzle::new(3, 4, 3),
        }
    }
}

/// The im2col fields of a tensor map (`cuTensorMapEncodeIm2col`): the bounding box of the pixels in the spatial (W, H, D) dimensions,
/// given by its corners relative to the tensor's extents, and the pixels and channels one
/// instruction moves. The box covers spatial coordinates `lower_corner[i]` to
/// `extent[i] - 1 + upper_corner[i]` of dimension `i + 1` (W is dimension 1); entries past
/// `rank - 2` are unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Im2colParams {
    /// `pixelBoxLowerCorner`, W first.
    pub lower_corner: [i32; 3],
    /// `pixelBoxUpperCorner`, W first.
    pub upper_corner: [i32; 3],
    /// `channelsPerPixel`: elements of dimension 0 (C) per pixel, 1 to 256.
    pub channels_per_pixel: u32,
    /// `pixelsPerColumn`: pixels per instruction, 1 to 1024.
    pub pixels_per_column: u32,
}

/// The widest an im2col corner may be for a tensor of rank `rank` (3 to 5): corners are
/// signed 16-, 8- and 5-bit values (the PTX ISA's ranges, `[-2^15, 2^15 - 1]`,
/// `[-2^7, 2^7 - 1]`, `[-2^4, 2^4 - 1]`).
pub fn im2col_corner_range(rank: usize) -> (i32, i32) {
    let bits = match rank {
        3 => 15,
        4 => 7,
        _ => 4,
    };
    (-(1 << bits), (1 << bits) - 1)
}

/// The largest im2col offset an instruction may give for a tensor of rank `rank`: the
/// PTX ISA's `[0, 2^16 - 1]`, `[0, 2^8 - 1]`, `[0, 2^5 - 1]` for ranks 3, 4 and 5.
pub fn im2col_offset_max(rank: usize) -> u32 {
    match rank {
        3 => (1 << 16) - 1,
        4 => (1 << 8) - 1,
        _ => (1 << 5) - 1,
    }
}

/// A concrete TMA descriptor: everything `cuTensorMapEncodeTiled` (or, with
/// [`im2col`](TmaDesc::im2col), `cuTensorMapEncodeIm2col`) takes, with byte strides. Arrays
/// hold `rank` entries (`global_strides` `rank - 1`); the rest are unused. An im2col
/// descriptor has no box extents (`box_dims` is unused): its box is
/// [`Im2colParams::pixels_per_column`] pixels of [`Im2colParams::channels_per_pixel`]
/// channels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TmaDesc {
    pub data_type: TensorMapDataType,
    pub rank: u32,
    /// The global address of element (0, ..., 0).
    pub global_address: u64,
    /// Elements in each dimension, dimension 0 innermost.
    pub global_dims: [u64; MAX_RANK],
    /// Bytes between consecutive indices of dimensions 1 to `rank - 1`.
    pub global_strides: [u64; MAX_RANK - 1],
    /// Elements of the box in each dimension (counting skipped ones).
    pub box_dims: [u32; MAX_RANK],
    /// The traversal step in each dimension (dimension 0's is ignored without interleave).
    pub element_strides: [u32; MAX_RANK],
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub l2_promotion: TensorMapL2Promotion,
    pub oob_fill: TensorMapFloatOOBFill,
    /// The im2col fields of an im2col descriptor; `None` for a tiled one.
    pub im2col: Option<Im2colParams>,
}

/// The simulator's descriptor magic, `CuTM`.
const SIM_MAGIC: [u8; 4] = *b"CuTM";
/// The version of the simulator's encoding.
const SIM_VERSION: u8 = 1;

/// One element of a box traversal ([`TmaDesc::box_elements`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoxElement {
    /// Its coordinates in the tensor (may be negative or past the end).
    pub coords: [i64; MAX_RANK],
    /// Its byte offset in the dense box in shared memory, before any swizzle.
    pub box_byte: u64,
    /// Its global address, or `None` when it is out of bounds.
    pub global_address: Option<u64>,
}

impl TmaDesc {
    /// A descriptor of `rank` dimensions with no swizzle, interleave, promotion or fill,
    /// unit element strides, and zeros elsewhere: set the fields after.
    pub fn new(data_type: TensorMapDataType, rank: u32) -> TmaDesc {
        TmaDesc {
            data_type,
            rank,
            global_address: 0,
            global_dims: [0; MAX_RANK],
            global_strides: [0; MAX_RANK - 1],
            box_dims: [0; MAX_RANK],
            element_strides: [1; MAX_RANK],
            interleave: TensorMapInterleave::none,
            swizzle: TensorMapSwizzle::none,
            l2_promotion: TensorMapL2Promotion::none,
            oob_fill: TensorMapFloatOOBFill::none,
            im2col: None,
        }
    }

    fn r(&self) -> usize {
        (self.rank as usize).min(MAX_RANK)
    }

    /// Check what `cuTensorMapEncodeTiled` requires (see [`check_tensor_map`]), or for an
    /// im2col descriptor `cuTensorMapEncodeIm2col` (see [`check_im2col`]).
    pub fn validate(&self) -> Result<(), String> {
        let r = self.r();
        let dims: Vec<Option<u64>> = self.global_dims[..r].iter().map(|&d| Some(d)).collect();
        let strides: Vec<Option<u64>> = self.global_strides[..r.saturating_sub(1)].iter().map(|&s| Some(s)).collect();
        if let Some(i) = &self.im2col {
            return check_im2col(
                &TensorMapCheck {
                    data_type: self.data_type,
                    rank: self.rank as usize,
                    global_address: Some(self.global_address),
                    global_dims: &dims,
                    global_strides: &strides,
                    box_dims: &self.box_dims[..r],
                    element_strides: &self.element_strides[..r],
                    interleave: self.interleave,
                    swizzle: self.swizzle,
                    oob_fill: self.oob_fill,
                },
                i,
            );
        }
        check_tensor_map(&TensorMapCheck {
            data_type: self.data_type,
            rank: self.rank as usize,
            global_address: Some(self.global_address),
            global_dims: &dims,
            global_strides: &strides,
            box_dims: &self.box_dims[..r],
            element_strides: &self.element_strides[..r],
            interleave: self.interleave,
            swizzle: self.swizzle,
            oob_fill: self.oob_fill,
        })
    }

    /// Elements the box holds along dimension `i`: `ceil(box_dims[i] / element_strides[i])`
    /// (dimension 0's stride counts only with interleave).
    pub fn box_extent(&self, i: usize) -> u64 {
        let s = if i == 0 && self.interleave == TensorMapInterleave::none { 1 } else { self.element_strides[i].max(1) };
        u64::from(self.box_dims[i]).div_ceil(u64::from(s))
    }

    /// Elements one box holds (an im2col box: its pixels times its channels).
    pub fn box_elements_count(&self) -> u64 {
        if let Some(i) = &self.im2col {
            return u64::from(i.channels_per_pixel) * u64::from(i.pixels_per_column);
        }
        (0..self.r()).map(|i| self.box_extent(i)).product()
    }

    /// Bytes of one box in global memory, which is the transaction count of a load
    /// (out-of-bounds elements count too, as the filled bytes they become): the elements
    /// at the encoding's width, packed for the sub-byte encodings (CuTe DSL's
    /// `global_tx_bytes`).
    pub fn box_bytes(&self) -> u64 {
        self.box_elements_count() * u64::from(self.data_type.bit_width()) / 8
    }

    /// Bytes of one box in shared memory: [`box_bytes`](TmaDesc::box_bytes), except for the
    /// padded sub-byte encodings, whose groups of 16 values take 16 bytes each (CuTe DSL's
    /// `shared_storage_bytes`).
    pub fn smem_box_bytes(&self) -> u64 {
        match self.data_type.group_slot_bytes() {
            Some(slot) => self.box_elements_count().div_ceil(16) * slot,
            None => self.box_bytes(),
        }
    }

    /// Bytes of the global tensor's dimension `i` stride (dimension 0's is the element).
    pub fn stride_bytes(&self, i: usize) -> u64 {
        if i == 0 { self.data_type.bytes() } else { self.global_strides[i - 1] }
    }

    /// Every element of the box whose first element is at coordinates `start` (one per
    /// dimension), in shared-memory order (dimension 0 fastest): coordinates, byte in the
    /// dense box, and global address when in bounds. Whole-byte encodings only.
    pub fn box_elements(&self, start: &[i64]) -> Vec<BoxElement> {
        let r = self.r();
        assert_eq!(start.len(), r, "a box of a rank-{r} tensor map starts at {r} coordinates");
        let elem = self.data_type.bytes();
        let extents: Vec<u64> = (0..r).map(|i| self.box_extent(i)).collect();
        let total: u64 = extents.iter().product();
        let mut out = Vec::with_capacity(total as usize);
        for n in 0..total {
            let mut rest = n;
            let mut coords = [0i64; MAX_RANK];
            let mut in_bounds = true;
            let mut addr = self.global_address as i128;
            for i in 0..r {
                let j = rest % extents[i];
                rest /= extents[i];
                let step = if i == 0 && self.interleave == TensorMapInterleave::none {
                    1
                } else {
                    i64::from(self.element_strides[i].max(1))
                };
                let c = start[i] + j as i64 * step;
                coords[i] = c;
                in_bounds &= c >= 0 && (c as u64) < self.global_dims[i];
                addr += c as i128 * self.stride_bytes(i) as i128;
            }
            out.push(BoxElement { coords, box_byte: n * elem, global_address: in_bounds.then_some(addr as u64) });
        }
        out
    }

    /// The shared-memory address of byte `box_byte` of a box placed at `smem_base`, after
    /// the descriptor's swizzle ([`TensorMapSwizzle::apply`] on the absolute address).
    pub fn smem_address(&self, smem_base: u64, box_byte: u64) -> u64 {
        self.swizzle.apply(smem_base + box_byte)
    }

    /// The simulator's 128-byte encoding: magic `CuTM`, version, the enum codes and rank,
    /// then the address, dimensions, strides, box and element strides, little-endian.
    /// The driver's own `CUtensorMap` format is undocumented; this one round-trips through
    /// [`TmaDesc::from_sim_bytes`] and is what a simulated kernel's descriptor parameter
    /// holds.
    pub fn to_sim_bytes(&self) -> [u8; TENSOR_MAP_BYTES] {
        let mut b = [0u8; TENSOR_MAP_BYTES];
        b[0..4].copy_from_slice(&SIM_MAGIC);
        b[4] = SIM_VERSION;
        b[5] = self.data_type.code() as u8;
        b[6] = self.rank as u8;
        b[7] = self.interleave.code() as u8;
        b[8] = self.swizzle.code() as u8;
        b[9] = self.l2_promotion.code() as u8;
        b[10] = self.oob_fill.code() as u8;
        // The mode at byte 11 (0 tiled, 1 im2col) and the im2col fields in
        // the last 16 bytes, which tiled descriptors leave zero.
        if let Some(i) = &self.im2col {
            b[11] = 1;
            for k in 0..3 {
                let clamp = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
                b[112 + 2 * k..114 + 2 * k].copy_from_slice(&clamp(i.lower_corner[k]).to_le_bytes());
                b[118 + 2 * k..120 + 2 * k].copy_from_slice(&clamp(i.upper_corner[k]).to_le_bytes());
            }
            b[124..126].copy_from_slice(&(i.channels_per_pixel.min(u32::from(u16::MAX)) as u16).to_le_bytes());
            b[126..128].copy_from_slice(&(i.pixels_per_column.min(u32::from(u16::MAX)) as u16).to_le_bytes());
        }
        b[16..24].copy_from_slice(&self.global_address.to_le_bytes());
        for i in 0..MAX_RANK {
            b[24 + 8 * i..32 + 8 * i].copy_from_slice(&self.global_dims[i].to_le_bytes());
        }
        for i in 0..MAX_RANK - 1 {
            b[64 + 8 * i..72 + 8 * i].copy_from_slice(&self.global_strides[i].to_le_bytes());
        }
        for i in 0..MAX_RANK {
            b[96 + 2 * i..98 + 2 * i]
                .copy_from_slice(&(self.box_dims[i].min(u32::from(u16::MAX)) as u16).to_le_bytes());
            b[106 + i] = self.element_strides[i].min(255) as u8;
        }
        b
    }

    /// Whether `bytes` hold the simulator's encoding of a descriptor (its magic).
    pub fn is_sim_encoding(bytes: &[u8]) -> bool {
        bytes.len() == TENSOR_MAP_BYTES && bytes[0..4] == SIM_MAGIC
    }

    /// Decode [`TmaDesc::to_sim_bytes`].
    pub fn from_sim_bytes(bytes: &[u8]) -> Result<TmaDesc, String> {
        if bytes.len() != TENSOR_MAP_BYTES {
            return Err(format!("a tensor map is {TENSOR_MAP_BYTES} bytes, not {}", bytes.len()));
        }
        if bytes[0..4] != SIM_MAGIC {
            return Err("these bytes are not a tensor map the simulator encoded (no CuTM magic): build it with \
                        create_tensor_map_tiled or bind a tensor with tma_descriptor"
                .to_string());
        }
        if bytes[4] != SIM_VERSION {
            return Err(format!("tensor map encoding version {} is not {SIM_VERSION}", bytes[4]));
        }
        let code = |i: usize| u32::from(bytes[i]);
        let bad = |what: &str, i: usize| format!("tensor map: {} is not a {what} code", bytes[i]);
        let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let mut d = TmaDesc::new(
            TensorMapDataType::from_code(code(5)).ok_or_else(|| bad("data type", 5))?,
            u32::from(bytes[6]),
        );
        d.interleave = TensorMapInterleave::from_code(code(7)).ok_or_else(|| bad("interleave", 7))?;
        d.swizzle = TensorMapSwizzle::from_code(code(8)).ok_or_else(|| bad("swizzle", 8))?;
        d.l2_promotion = TensorMapL2Promotion::from_code(code(9)).ok_or_else(|| bad("L2 promotion", 9))?;
        d.oob_fill = TensorMapFloatOOBFill::from_code(code(10)).ok_or_else(|| bad("OOB fill", 10))?;
        d.global_address = u64_at(16);
        for i in 0..MAX_RANK {
            d.global_dims[i] = u64_at(24 + 8 * i);
            d.box_dims[i] = u32::from(u16::from_le_bytes([bytes[96 + 2 * i], bytes[97 + 2 * i]]));
            d.element_strides[i] = u32::from(bytes[106 + i]);
        }
        for i in 0..MAX_RANK - 1 {
            d.global_strides[i] = u64_at(64 + 8 * i);
        }
        match bytes[11] {
            0 => {}
            1 => {
                let i16_at = |i: usize| i32::from(i16::from_le_bytes([bytes[i], bytes[i + 1]]));
                let u16_at = |i: usize| u32::from(u16::from_le_bytes([bytes[i], bytes[i + 1]]));
                let mut p = Im2colParams {
                    channels_per_pixel: u16_at(124),
                    pixels_per_column: u16_at(126),
                    ..Im2colParams::default()
                };
                for k in 0..3 {
                    p.lower_corner[k] = i16_at(112 + 2 * k);
                    p.upper_corner[k] = i16_at(118 + 2 * k);
                }
                d.im2col = Some(p);
            }
            m => return Err(format!("tensor map: {m} is not a tensor-map mode code (0 tiled, 1 im2col)")),
        }
        Ok(d)
    }
}

impl fmt::Display for TmaDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.r();
        write!(
            f,
            "tensor map {} rank {r} at {:#x}: dims {:?}, strides {:?} bytes, box {:?}, element strides {:?}, \
             interleave {}, swizzle {}, l2 {}, oob {}",
            self.data_type,
            self.global_address,
            &self.global_dims[..r],
            &self.global_strides[..r.saturating_sub(1)],
            &self.box_dims[..r],
            &self.element_strides[..r],
            self.interleave,
            self.swizzle,
            self.l2_promotion,
            self.oob_fill
        )?;
        if let Some(i) = &self.im2col {
            let n = r.saturating_sub(2);
            write!(
                f,
                ", im2col corners {:?} to {:?}, {} channels x {} pixels",
                &i.lower_corner[..n.min(3)],
                &i.upper_corner[..n.min(3)],
                i.channels_per_pixel,
                i.pixels_per_column
            )?;
        }
        Ok(())
    }
}

/// The fields of a descriptor as far as they are known: `None` for an address, extent or
/// stride only known at launch. [`check_tensor_map`] checks what is known, so the same
/// rules run when a kernel is built and when it is launched.
#[derive(Clone, Copy, Debug)]
pub struct TensorMapCheck<'a> {
    pub data_type: TensorMapDataType,
    pub rank: usize,
    pub global_address: Option<u64>,
    pub global_dims: &'a [Option<u64>],
    /// Byte strides of dimensions 1 and up.
    pub global_strides: &'a [Option<u64>],
    pub box_dims: &'a [u32],
    pub element_strides: &'a [u32],
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub oob_fill: TensorMapFloatOOBFill,
}

/// Check a descriptor against what `cuTensorMapEncodeTiled` accepts (CUDA 12.8's
/// documented requirements), for the fields that are known:
///
/// - the swizzle is not one of the SM100 atom modes cute-rs does not support
///   (`s128b_atom_32b_flip_8b` and `s128b_atom_64b`);
/// - the sub-byte encodings have no interleave or NaN
///   fill and a box whose inner extent is a whole number of 16-value groups; the padded
///   ones (`f416u4_align16b`, `f416u6_align16b`) also an inner box of exactly 128 values
///   (PTX's 64 bytes of `.b4x16_p64`, 96 of `.b6x16_p32`), an inner extent that is a
///   multiple of 128, a 32-byte-aligned address and strides, and no swizzle or the
///   128-byte one (the PTX ISA's restrictions on tensor copies, which CuTe DSL's
///   `cutlass.experimental.cuda` tensor maps check when they are made);
/// - the rank is 1 to 5, and at least 3 with an interleave;
/// - the address is a multiple of 16 bytes (32 with the 32-byte interleave);
/// - every extent is 1 to 2^32;
/// - every stride is a multiple of 16 bytes (32 with the 32-byte interleave) below 2^40;
/// - every box extent is 1 to 256, and without interleave the box's inner extent is a
///   multiple of 16 bytes and, with a swizzle, no wider than the swizzle's span;
/// - every element stride is 1 to 8;
/// - the 32-byte interleave comes with the 32-byte swizzle;
/// - the NaN fill comes with a floating-point encoding.
pub fn check_tensor_map(c: &TensorMapCheck<'_>) -> Result<(), String> {
    let t = c.data_type;
    if c.swizzle.is_unsupported_atom_mode() {
        return Err(format!(
            "tensor map: the {} swizzle is not supported (of the SM100 atom modes, cute-rs takes \
             s128b_atom_32b)",
            c.swizzle
        ));
    }
    if !(1..=MAX_RANK).contains(&c.rank) {
        return Err(format!("tensor map: rank {} is outside 1 to {MAX_RANK}", c.rank));
    }
    if c.interleave != TensorMapInterleave::none && c.rank < 3 {
        return Err(format!("tensor map: the {} interleave needs rank 3 or more, got rank {}", c.interleave, c.rank));
    }
    let lens = [
        ("extents", c.global_dims.len(), c.rank),
        ("strides", c.global_strides.len(), c.rank - 1),
        ("box extents", c.box_dims.len(), c.rank),
        ("element strides", c.element_strides.len(), c.rank),
    ];
    for (what, got, want) in lens {
        if got != want {
            return Err(format!("tensor map of rank {}: {got} {what} given, expected {want}", c.rank));
        }
    }
    let wide = c.interleave == TensorMapInterleave::i32b;
    let base_align = if wide { 32 } else { 16 };
    if let Some(a) = c.global_address
        && a % base_align != 0
    {
        return Err(format!(
            "tensor map: the global address {a:#x} is not {base_align}-byte aligned{}",
            if wide { " (the 32-byte interleave needs 32)" } else { "" }
        ));
    }
    for (i, d) in c.global_dims.iter().enumerate() {
        if let Some(d) = *d
            && !(1..=1u64 << 32).contains(&d)
        {
            return Err(format!("tensor map: dimension {i} has {d} elements, outside 1 to 2^32"));
        }
    }
    let stride_align = if wide { 32 } else { 16 };
    for (k, s) in c.global_strides.iter().enumerate() {
        if let Some(s) = *s {
            if s % stride_align != 0 {
                return Err(format!(
                    "tensor map: the stride of dimension {} is {s} bytes, not a multiple of {stride_align}",
                    k + 1
                ));
            }
            if s >= 1u64 << 40 {
                return Err(format!("tensor map: the stride of dimension {} is {s} bytes, not below 2^40", k + 1));
            }
        }
    }
    for (i, &b) in c.box_dims.iter().enumerate() {
        if !(1..=MAX_BOX_DIM).contains(&b) {
            return Err(format!("tensor map: box dimension {i} is {b} elements, outside 1 to {MAX_BOX_DIM}"));
        }
    }
    if t.is_packed() {
        check_packed(c)?;
    }
    let inner = u64::from(c.box_dims[0]) * u64::from(t.bit_width()) / 8;
    if c.interleave == TensorMapInterleave::none {
        if !inner.is_multiple_of(16) {
            return Err(format!(
                "tensor map: the box's inner dimension is {inner} bytes ({} x {t}), not a multiple of 16",
                c.box_dims[0]
            ));
        }
        if let Some(span) = c.swizzle.span_bytes()
            && inner > span
        {
            return Err(format!(
                "tensor map: the {} swizzle needs the box's inner dimension to be at most {span} bytes, got {inner}",
                c.swizzle
            ));
        }
    }
    for (i, &s) in c.element_strides.iter().enumerate() {
        if !(1..=MAX_ELEMENT_STRIDE).contains(&s) {
            return Err(format!("tensor map: element stride {i} is {s}, outside 1 to {MAX_ELEMENT_STRIDE}"));
        }
    }
    if wide && c.swizzle != TensorMapSwizzle::s32b {
        return Err(format!("tensor map: the i32b interleave needs the s32b swizzle, got {}", c.swizzle));
    }
    if c.oob_fill == TensorMapFloatOOBFill::nan_request_zero_fma && !t.is_float() {
        return Err(format!("tensor map: the NaN out-of-bounds fill needs a floating-point encoding, not {t}"));
    }
    Ok(())
}

/// The rules of the sub-byte encodings (see [`check_tensor_map`]).
fn check_packed(c: &TensorMapCheck<'_>) -> Result<(), String> {
    let t = c.data_type;
    if c.interleave != TensorMapInterleave::none {
        return Err(format!("tensor map: the sub-byte {t} encoding takes no interleave, not {}", c.interleave));
    }
    if c.oob_fill == TensorMapFloatOOBFill::nan_request_zero_fma {
        return Err(format!("tensor map: the sub-byte {t} encoding does not take the NaN out-of-bounds fill"));
    }
    let box0 = c.box_dims.first().copied().unwrap_or(0);
    if !box0.is_multiple_of(16) {
        return Err(format!(
            "tensor map: the box's inner dimension is {box0} values of the sub-byte {t} encoding, which moves them in \
             groups of 16: make it a multiple of 16"
        ));
    }
    if !t.is_padded() {
        // The driver: an odd number of packed 4-bit values is refused.
        if let Some(Some(d)) = c.global_dims.first()
            && !d.is_multiple_of(2)
        {
            return Err(format!("tensor map: the packed {t} encoding needs an even inner extent, got {d}"));
        }
        return Ok(());
    }
    let (ptx, global_bytes) =
        if t == TensorMapDataType::f416u4_align16b { (".b4x16_p64", 64) } else { (".b6x16_p32", 96) };
    if box0 != 128 {
        return Err(format!(
            "tensor map: the padded {t} encoding ({ptx}) needs a box whose inner dimension is exactly {global_bytes} \
             bytes of global memory (128 values), got {box0} values"
        ));
    }
    if let Some(Some(d)) = c.global_dims.first()
        && !d.is_multiple_of(128)
    {
        return Err(format!(
            "tensor map: the padded {t} encoding ({ptx}) needs the inner extent to be a multiple of 128 values \
             ({global_bytes} bytes), got {d}"
        ));
    }
    if let Some(a) = c.global_address
        && !a.is_multiple_of(32)
    {
        return Err(format!("tensor map: the padded {t} encoding ({ptx}) needs a 32-byte-aligned address, got {a:#x}"));
    }
    for (k, s) in c.global_strides.iter().enumerate() {
        if let Some(s) = *s
            && !s.is_multiple_of(32)
        {
            return Err(format!(
                "tensor map: the padded {t} encoding ({ptx}) needs strides that are multiples of 32 bytes; dimension \
                 {}'s is {s}",
                k + 1
            ));
        }
    }
    if !matches!(c.swizzle, TensorMapSwizzle::none | TensorMapSwizzle::s128b) {
        return Err(format!(
            "tensor map: the padded {t} encoding ({ptx}) takes no swizzle or the 128-byte one, not {}",
            c.swizzle
        ));
    }
    Ok(())
}

/// Check an im2col descriptor against what `cuTensorMapEncodeIm2col` accepts (CUDA 12.8's
/// documented requirements and the PTX ISA's im2col mode),
/// for the fields that are known: the rank is 3 to 5 (NWC, NHWC, NDHWC), the encoding a
/// whole-byte one, no interleave (cute-rs's choice: the driver takes the interleaves,
/// but CuTe's im2col atoms never use them), each corner within the signed range of its
/// rank ([`im2col_corner_range`]), 1 to 256 channels per pixel whose bytes are a multiple
/// of 16 and within the swizzle's span, 1 to 1024 pixels per column, and the address,
/// extents, strides, element strides and fill as for a tiled descriptor.
pub fn check_im2col(c: &TensorMapCheck<'_>, p: &Im2colParams) -> Result<(), String> {
    let t = c.data_type;
    if !(3..=MAX_RANK).contains(&c.rank) {
        return Err(format!(
            "im2col tensor map: rank {} is outside 3 to {MAX_RANK} (im2col takes NWC, NHWC and NDHWC tensors)",
            c.rank
        ));
    }
    if t.is_packed() {
        return Err(format!("im2col tensor map: the sub-byte {t} encoding is not supported in im2col mode"));
    }
    if c.interleave != TensorMapInterleave::none {
        return Err(format!(
            "im2col tensor map: cute-rs's im2col descriptors take no interleave, not {}",
            c.interleave
        ));
    }
    let (lo, hi) = im2col_corner_range(c.rank);
    for k in 0..c.rank - 2 {
        let dim = ["W", "H", "D"][k];
        for (what, v) in [("lower", p.lower_corner[k]), ("upper", p.upper_corner[k])] {
            if !(lo..=hi).contains(&v) {
                return Err(format!(
                    "im2col tensor map: the {what} corner of the bounding box in {dim} is {v}, outside {lo} to {hi} \
                     (the range of a rank-{} tensor's corners)",
                    c.rank
                ));
            }
        }
    }
    if !(1..=256).contains(&p.channels_per_pixel) {
        return Err(format!("im2col tensor map: {} channels per pixel is outside 1 to 256", p.channels_per_pixel));
    }
    if !(1..=1024).contains(&p.pixels_per_column) {
        return Err(format!("im2col tensor map: {} pixels per column is outside 1 to 1024", p.pixels_per_column));
    }
    let inner = u64::from(p.channels_per_pixel) * t.bytes();
    if !inner.is_multiple_of(16) {
        return Err(format!(
            "im2col tensor map: a pixel's {} channels are {inner} bytes ({} x {t}), not a multiple of 16",
            p.channels_per_pixel, p.channels_per_pixel
        ));
    }
    if let Some(span) = c.swizzle.span_bytes()
        && inner > span
    {
        return Err(format!(
            "im2col tensor map: the {} swizzle needs a pixel's channels to be at most {span} bytes, got {inner}",
            c.swizzle
        ));
    }
    if c.swizzle.is_atom_mode() {
        return Err(format!("im2col tensor map: the {} swizzle is not supported", c.swizzle));
    }
    // The driver: the bounding box has a non-zero area in every spatial dimension.
    for k in 0..c.rank - 2 {
        if let Some(Some(d)) = c.global_dims.get(k + 1)
            && i64::from(p.lower_corner[k]) > *d as i64 - 1 + i64::from(p.upper_corner[k])
        {
            return Err(format!(
                "im2col tensor map: the bounding box in {} spans {} to {}, which is empty",
                ["W", "H", "D"][k],
                p.lower_corner[k],
                *d as i64 - 1 + i64::from(p.upper_corner[k])
            ));
        }
    }
    // The rest as for a tiled descriptor: checked with a box of one pixel's channels.
    let mut boxes = vec![1u32; c.rank];
    boxes[0] = p.channels_per_pixel;
    check_tensor_map(&TensorMapCheck { box_dims: &boxes, ..*c })
        .map_err(|e| e.replacen("tensor map:", "im2col tensor map:", 1))
}

/// A descriptor as the kernel's build knows it ([`tma_descriptor`]): the fields fixed when
/// it is built, and the extents, byte strides and start of a tensor parameter as
/// expressions over constants and the kernel's scalar parameters. [`TmaDescSpec::evaluate`]
/// turns it into a [`TmaDesc`] at each launch, from the tensor's address and the launch's
/// scalar arguments; the simulator and the CUDA launcher both call it.
#[derive(Clone, Debug, PartialEq)]
pub struct TmaDescSpec {
    pub data_type: TensorMapDataType,
    /// Elements in each dimension, TMA order (dimension 0 innermost).
    pub global_dims: Vec<IndexExpr>,
    /// Byte strides of dimensions 1 and up.
    pub global_strides: Vec<IndexExpr>,
    pub box_dims: Vec<u32>,
    pub element_strides: Vec<u32>,
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub l2_promotion: TensorMapL2Promotion,
    pub oob_fill: TensorMapFloatOOBFill,
    /// Bytes from the buffer's start to element (0, ..., 0).
    pub base_offset: IndexExpr,
    /// The im2col fields of an im2col descriptor; `None`
    /// for a tiled one.
    pub im2col: Option<Im2colParams>,
}

impl TmaDescSpec {
    /// The scalar-parameter values the expressions read.
    pub fn dependencies(&self) -> Vec<ValueId> {
        let mut out = Vec::new();
        for e in self.global_dims.iter().chain(&self.global_strides).chain(std::iter::once(&self.base_offset)) {
            e.dependencies(&mut out);
        }
        out.sort_by_key(|v| v.0);
        out.dedup();
        out
    }

    /// Check what is known when the kernel is built (with the buffer's alignment
    /// promise standing in for its address).
    pub fn check_static(&self, buffer_align: usize) -> Result<(), String> {
        let dims: Vec<Option<u64>> = self.global_dims.iter().map(|e| e.as_static().map(|v| v as u64)).collect();
        let strides: Vec<Option<u64>> = self.global_strides.iter().map(|e| e.as_static().map(|v| v as u64)).collect();
        let need = if self.interleave == TensorMapInterleave::i32b { 32 } else { 16 };
        if buffer_align < need {
            return Err(format!(
                "the tensor's start is only promised {buffer_align}-byte aligned, and a tensor map's global address \
                 must be {need}-byte aligned (declare it with that alignment, e.g. TensorSpec::with_assumed_align)"
            ));
        }
        // The address is the buffer's (aligned to its promise) plus the static offset; the
        // check sees an address with exactly that alignment when the offset is static.
        let address = self.base_offset.as_static().map(|o| buffer_align as u64 + o as u64);
        let check = TensorMapCheck {
            data_type: self.data_type,
            rank: self.global_dims.len(),
            global_address: address,
            global_dims: &dims,
            global_strides: &strides,
            box_dims: &self.box_dims,
            element_strides: &self.element_strides,
            interleave: self.interleave,
            swizzle: self.swizzle,
            oob_fill: self.oob_fill,
        };
        match &self.im2col {
            Some(p) => check_im2col(&check, p),
            None => check_tensor_map(&check),
        }
    }

    /// The concrete descriptor for a launch: `buffer_address` is the tensor parameter's
    /// address, `env` gives each scalar parameter's value (as a signed integer), or `None`
    /// for a value the launch does not have. The result is checked
    /// ([`TmaDesc::validate`]).
    pub fn evaluate(&self, buffer_address: u64, env: &dyn Fn(ValueId) -> Option<i64>) -> Result<TmaDesc, String> {
        for v in self.dependencies() {
            if env(v).is_none() {
                return Err(format!("tensor map: value %{} is not a scalar argument of the launch", v.0));
            }
        }
        let get = |v: ValueId| env(v).unwrap_or(0);
        let eval = |e: &IndexExpr, what: &str| -> Result<u64, String> {
            let x = e.eval_signed(&get);
            u64::try_from(x).map_err(|_| format!("tensor map: {what} is negative ({x})"))
        };
        let rank = self.global_dims.len();
        let mut d = TmaDesc::new(self.data_type, rank as u32);
        d.global_address = buffer_address.wrapping_add(eval(&self.base_offset, "the start offset")?);
        for (i, e) in self.global_dims.iter().enumerate().take(MAX_RANK) {
            d.global_dims[i] = eval(e, &format!("dimension {i}'s extent"))?;
        }
        for (k, e) in self.global_strides.iter().enumerate().take(MAX_RANK - 1) {
            d.global_strides[k] = eval(e, &format!("dimension {}'s stride", k + 1))?;
        }
        for (i, &b) in self.box_dims.iter().enumerate().take(MAX_RANK) {
            d.box_dims[i] = b;
        }
        for (i, &s) in self.element_strides.iter().enumerate().take(MAX_RANK) {
            d.element_strides[i] = s;
        }
        d.interleave = self.interleave;
        d.swizzle = self.swizzle;
        d.l2_promotion = self.l2_promotion;
        d.oob_fill = self.oob_fill;
        d.im2col = self.im2col;
        d.validate()?;
        Ok(d)
    }
}

/// A kernel parameter that holds a tensor map ([`KernelIr::tensor_maps`](crate::ir::KernelIr::tensor_maps)).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorMapParam {
    /// Its index among the kernel's parameters.
    pub param: usize,
    /// The by-value buffer that holds its 128 bytes.
    pub buffer: BufferId,
    /// For a hidden parameter, the tensor parameter it describes and how; `None` for one
    /// the host binds ([`param_tensormap`]).
    pub source: Option<TensorMapSource>,
}

/// What a hidden tensor-map parameter describes.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorMapSource {
    /// The tensor parameter (a global buffer).
    pub tensor: BufferId,
    pub spec: TmaDescSpec,
}

/// The options of a descriptor beyond its box (CuTe DSL's keyword arguments of
/// `create_tensor_map_tiled`): traversal strides, interleave, swizzle, L2 promotion,
/// out-of-bounds fill and the data format.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorMapOptions {
    /// One per dimension, TMA order; all 1 when `None`.
    pub traversal_strides: Option<Vec<u32>>,
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub l2_promotion: TensorMapL2Promotion,
    pub oob_fill: TensorMapFloatOOBFill,
    /// The transfer format; the element type's own encoding when `None`.
    pub tma_format: Option<crate::tensor_map::TensorMapDataFormat>,
}

impl TensorMapOptions {
    pub fn new() -> TensorMapOptions {
        TensorMapOptions::default()
    }

    pub fn traversal_strides(mut self, strides: &[u32]) -> Self {
        self.traversal_strides = Some(strides.to_vec());
        self
    }

    pub fn interleave(mut self, interleave: TensorMapInterleave) -> Self {
        self.interleave = interleave;
        self
    }

    pub fn swizzle(mut self, swizzle: TensorMapSwizzle) -> Self {
        self.swizzle = swizzle;
        self
    }

    pub fn l2_promotion(mut self, l2: TensorMapL2Promotion) -> Self {
        self.l2_promotion = l2;
        self
    }

    pub fn oob_fill(mut self, fill: TensorMapFloatOOBFill) -> Self {
        self.oob_fill = fill;
        self
    }

    pub fn tma_format(mut self, format: crate::tensor_map::TensorMapDataFormat) -> Self {
        self.tma_format = Some(format);
        self
    }
}

/// A tensor-map parameter inside a kernel: a hidden one ([`tma_descriptor`]) or one the host
/// binds ([`param_tensormap`]). Its bytes live in the kernel's parameter space as a grid
/// constant; [`get_ptr`](TmaDescriptor::get_ptr) gives their generic address, which TMA
/// instructions take.
#[derive(Clone, Debug)]
pub struct TmaDescriptor {
    buffer: BufferId,
    name: String,
    spec: Option<TmaDescSpec>,
}

impl TmaDescriptor {
    /// The parameter's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// How the descriptor is computed, for a hidden one.
    pub fn spec(&self) -> Option<&TmaDescSpec> {
        self.spec.as_ref()
    }

    /// The by-value buffer holding the descriptor.
    pub fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// The generic address of the descriptor in the parameter space (CuTe DSL's
    /// `TensorMap.get_ptr`, `cvta.param`): what `cp.async.bulk.tensor` and
    /// `prefetch.tensormap` take.
    #[track_caller]
    pub fn get_ptr(&self) -> Int64 {
        param_address(self.buffer)
    }

    /// The descriptor's bytes as 16 `u64` loads, for checking it.
    #[track_caller]
    pub fn load_word(&self, k: usize) -> Value<u64> {
        let loc = Location::caller();
        if k >= TENSOR_MAP_BYTES / 8 {
            user_panic(loc, format!("cute: `{}`: a tensor map has 16 words, not {}", self.name, k + 1));
        }
        load_param::<u64>(self.buffer, k, loc)
    }
}

/// The generic address of by-value buffer `buffer` (a `.param` aggregate).
#[track_caller]
pub(crate) fn param_address(buffer: BufferId) -> Int64 {
    let loc = Location::caller();
    let result = builder::with(|b| {
        let index = b.constant(crate::ir::Literal::I32(0), loc);
        let result = b.new_value(DType::I64);
        b.push(crate::ir::Op::PtrToInt { result, buffer, index, elem: DType::U8 }, loc);
        result
    });
    Value::from_id(result)
}

fn load_param<T: Numeric>(buffer: BufferId, index: usize, loc: &'static Location<'static>) -> Value<T> {
    let result = builder::with(|b| {
        let index = b.constant(crate::ir::Literal::I32(index as i32), loc);
        let result = b.new_value(T::DTYPE);
        b.push(
            crate::ir::Op::Load {
                results: vec![result],
                buffer,
                index,
                attrs: crate::ir::MemAttrs::default(),
                pred: None,
                pass_thru: vec![],
            },
            loc,
        );
        result
    });
    Value::from_id(result)
}

/// Declare a tensor-map parameter the host binds (CuTe DSL's `GridConstant[TensorMap]`
/// kernel argument): 128 bytes, 64-byte aligned, a grid constant. Launches pass
/// [`KernelArg::tensor_map`](crate::KernelArg::tensor_map) of a host-built
/// [`TensorMap`](crate::tensor_map::TensorMap); the CUDA launcher encodes it with
/// `cuTensorMapEncodeTiled`, and the simulator keeps its own encoding.
#[track_caller]
pub fn param_tensormap(name: &str) -> TmaDescriptor {
    let p = crate::abi::param_bytes(name, TENSOR_MAP_BYTES, TENSOR_MAP_ALIGN, true);
    let buffer = p.buffer_id();
    builder::with(|b| b.register_tensor_map(buffer, None));
    TmaDescriptor { buffer, name: name.to_string(), spec: None }
}

/// The TMA dimensions of a tensor's flat modes: per TMA dimension (innermost first) the
/// mode it comes from, its extent and its element stride. Modes are sorted by stride
/// (static strides first by value, runtime strides after, in mode order) unless
/// `stride_order` gives the order, innermost first; the innermost must have stride 1.
pub(crate) fn tma_modes(
    shape: &[IndexExpr],
    stride: &[StrideLeaf],
    stride_order: Option<&[usize]>,
) -> Result<Vec<(usize, IndexExpr, IndexExpr)>, String> {
    let mut strides = Vec::new();
    for (i, s) in stride.iter().enumerate() {
        match s {
            StrideLeaf::Int(e) => strides.push(e.clone()),
            StrideLeaf::Basis(_) => return Err(format!("mode {i} has a basis stride, not an integer one")),
        }
    }
    let order: Vec<usize> = match stride_order {
        Some(o) => {
            let mut seen = o.to_vec();
            seen.sort_unstable();
            if seen != (0..shape.len()).collect::<Vec<_>>() {
                return Err(format!("stride_order {o:?} is not a permutation of the {} modes", shape.len()));
            }
            o.to_vec()
        }
        None => {
            let mut o: Vec<usize> = (0..shape.len()).collect();
            o.sort_by_key(|&i| (strides[i].as_static().is_none(), strides[i].as_static().unwrap_or(0), i));
            o
        }
    };
    // Size-1 modes with a non-unit stride are kept: TMA needs a stride for every dimension
    // but dimension 0, and a size-1 dimension never moves.
    let first = order[0];
    if strides[first].as_static() != Some(1) && !(order.len() == 1 && shape[first].as_static() == Some(1)) {
        return Err(format!(
            "the innermost mode (mode {first}) has stride {:?}, not 1: TMA's dimension 0 is contiguous",
            strides[first]
        ));
    }
    Ok(order.into_iter().map(|i| (i, shape[i].clone(), strides[i].clone())).collect())
}

/// Register a hidden tensor-map parameter describing the global tensor parameter `tensor`
/// (cute-rs's form of the descriptor a CuTe DSL TMA atom owns): the CUDA launcher encodes
/// it at each launch from the tensor's device address and the launch's scalar arguments
/// with `cuTensorMapEncodeTiled`, and the simulator from its own address map. It is
/// appended after the kernel's own parameters, so launches pass the same arguments as
/// before.
///
/// `tensor` must be a tensor parameter ([`param_tensor`](crate::param_tensor)), unswizzled,
/// with a flat layout whose extents and strides are constants or the kernel's scalar
/// parameters (runtime extents of a [`TensorSpec`](crate::TensorSpec), [`param_extent`](crate::param_extent)).
/// The TMA dimensions are the tensor's modes in increasing stride order (or
/// `stride_order`, innermost first), and `box_dims` gives the box in the tensor's mode order.
/// What the build can check (box extents, element strides, swizzle span, static strides
/// and extents, the start's alignment) fails here; what depends on launch values fails the
/// launch, naming the parameter.
#[track_caller]
pub fn tma_descriptor<T: Numeric>(
    tensor: &Tensor<T>,
    box_dims: &[u32],
    stride_order: Option<&[usize]>,
    options: &TensorMapOptions,
) -> TmaDescriptor {
    let loc = Location::caller();
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: tma_descriptor: {m}")) };
    let Engine::Memory { buffer, .. } = tensor.engine else { fail("the tensor is not in memory".to_string()) };
    let (is_param, decl) = builder::with(|b| {
        (b.params().iter().any(|p| matches!(p, Param::Buffer(x) if *x == buffer)), b.buffer(buffer).clone())
    });
    if !is_param {
        fail(format!("`{}` is not a tensor parameter: TMA describes a kernel argument in global memory", decl.name));
    }
    if !matches!(tensor.inner, crate::tensor::TensorInner::Identity) {
        fail(format!("`{}` is swizzled or composed: TMA describes a plain strided tensor", decl.name));
    }
    let dtype = T::DTYPE;
    let data_type = match crate::tensor_map::resolve_format(dtype, options.tma_format) {
        Ok(t) => t,
        Err(e) => fail(e),
    };
    let flat = tensor.layout.flatten();
    let shape = flat.shape().flat_leaves();
    let stride = flat.stride().flat_leaves();
    if box_dims.len() != shape.len() {
        fail(format!("{} box extents for a tensor of {} modes", box_dims.len(), shape.len()));
    }
    let modes = tma_modes(&shape, &stride, stride_order).unwrap_or_else(|e| fail(format!("`{}`: {e}", decl.name)));
    let elem = dtype.bytes();
    let spec = TmaDescSpec {
        data_type,
        global_dims: modes.iter().map(|(_, e, _)| e.clone()).collect(),
        global_strides: modes
            .iter()
            .skip(1)
            .map(|(_, _, s)| IndexExpr::product(s.clone(), IndexExpr::Static(elem)))
            .collect(),
        box_dims: modes.iter().map(|(i, _, _)| box_dims[*i]).collect(),
        element_strides: options.traversal_strides.clone().unwrap_or_else(|| vec![1; modes.len()]),
        interleave: options.interleave,
        swizzle: options.swizzle,
        l2_promotion: options.l2_promotion,
        oob_fill: options.oob_fill,
        base_offset: IndexExpr::product(tensor.offset.clone(), IndexExpr::Static(elem)),
        im2col: None,
    };
    match declare_hidden_descriptor(buffer, &decl, spec) {
        Ok(d) => d,
        Err(e) => fail(e),
    }
}

/// Register a hidden tensor-map parameter describing the global tensor parameter `buffer`
/// (declared as `decl`) with `spec`, after checking that every runtime leaf of the spec is
/// one of the kernel's scalar parameters and what the build can check of it
/// ([`TmaDescSpec::check_static`]). [`tma_descriptor`] and the TMA atoms
/// ([`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom)) share it.
#[track_caller]
pub(crate) fn declare_hidden_descriptor(
    buffer: BufferId,
    decl: &crate::ir::BufferDecl,
    spec: TmaDescSpec,
) -> Result<TmaDescriptor, String> {
    // Every runtime leaf must be one of the kernel's scalar parameters: the launch knows
    // nothing else.
    let scalars: Vec<ValueId> = builder::with(|b| {
        b.params()
            .iter()
            .filter_map(|p| match p {
                Param::Scalar { value, .. } => Some(*value),
                _ => None,
            })
            .collect()
    });
    if let Some(v) = spec.dependencies().into_iter().find(|v| !scalars.contains(v)) {
        return Err(format!(
            "`{}`'s extents, strides or offset use %{}, a value computed in the kernel: a descriptor is encoded \
             on the host, so they must be constants or scalar parameters",
            decl.name, v.0
        ));
    }
    spec.check_static(decl.align).map_err(|e| format!("`{}`: {e}", decl.name))?;
    let _at = builder::LocationScope::new();
    let (buffer_id, name) = builder::with(|b| {
        let name = format!("{}_tma_desc_{}", decl.name, b.tensor_map_count());
        let id = b.declare_hidden_tensor_map(&name, TENSOR_MAP_BYTES, TENSOR_MAP_ALIGN);
        b.register_tensor_map(id, Some(TensorMapSource { tensor: buffer, spec: spec.clone() }));
        (id, name)
    });
    Ok(TmaDescriptor { buffer: buffer_id, name, spec: Some(spec) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc2d() -> TmaDesc {
        let mut d = TmaDesc::new(TensorMapDataType::float16, 2);
        d.global_address = 0x1000;
        d.global_dims[..2].copy_from_slice(&[128, 64]);
        d.global_strides[0] = 256;
        d.box_dims[..2].copy_from_slice(&[64, 8]);
        d
    }

    #[test]
    fn sim_encoding_round_trips() {
        let mut d = desc2d();
        d.swizzle = TensorMapSwizzle::s128b;
        d.l2_promotion = TensorMapL2Promotion::l2_128b;
        d.oob_fill = TensorMapFloatOOBFill::nan_request_zero_fma;
        d.element_strides[1] = 2;
        let b = d.to_sim_bytes();
        assert!(TmaDesc::is_sim_encoding(&b));
        assert_eq!(TmaDesc::from_sim_bytes(&b).unwrap(), d);
        assert!(TmaDesc::from_sim_bytes(&[0u8; 128]).unwrap_err().contains("CuTM"));
        assert!(TmaDesc::from_sim_bytes(&[0u8; 64]).unwrap_err().contains("128 bytes"));
    }

    #[test]
    fn enums_match_the_driver() {
        assert_eq!(TensorMapDataType::bfloat16.code(), 9);
        assert_eq!(TensorMapDataType::f416u4_align8b.code(), 13);
        assert_eq!(TensorMapDataType::f416u6_align16b.code(), 15);
        assert_eq!(TensorMapSwizzle::s128b_atom_64b.code(), 6);
        assert_eq!(TensorMapL2Promotion::l2_256b.code(), 3);
        assert_eq!(TensorMapInterleave::i32b.code(), 2);
        assert_eq!(TensorMapFloatOOBFill::nan_request_zero_fma.code(), 1);
        assert_eq!(TensorMapDataType::for_dtype(DType::F8E4M3FN), Some(TensorMapDataType::uint8));
        assert_eq!(TensorMapDataType::for_dtype(DType::F4E2M1FN), Some(TensorMapDataType::f416u4_align8b));
        assert_eq!(TensorMapDataType::for_dtype(DType::F6E3M2FN), Some(TensorMapDataType::f416u6_align16b));
        assert_eq!(TensorMapDataType::for_dtype(DType::Bool), None);
    }

    #[test]
    fn swizzle_128b_xors_chunks_with_the_row() {
        // With a 1024-byte-aligned box, 16-byte chunk c of 128-byte row r lands at chunk
        // c ^ (r % 8) of the same row.
        let base = 0x4000u64;
        for r in 0..16u64 {
            for c in 0..8u64 {
                for byte in [0u64, 5, 15] {
                    let a = TensorMapSwizzle::s128b.apply(base + r * 128 + c * 16 + byte);
                    assert_eq!(a, base + r * 128 + ((c ^ (r % 8)) * 16) + byte, "r {r} c {c}");
                }
            }
        }
        // 64B: chunk c (of 4 in a 64-byte half row) XORs with bits 7..8 of the address.
        assert_eq!(TensorMapSwizzle::s64b.apply(128 + 16), 128);
        assert_eq!(TensorMapSwizzle::s64b.apply(3 * 128 + 16), 3 * 128 + 32);
        // 32B: chunk bit 4 XOR address bit 7.
        assert_eq!(TensorMapSwizzle::s32b.apply(128), 128 + 16);
        assert_eq!(TensorMapSwizzle::s32b.apply(256 + 16), 256 + 16);
        assert_eq!(TensorMapSwizzle::none.apply(1234), 1234);
        // Address-based: a box at a non-pattern-aligned base starts mid-pattern.
        assert_eq!(TensorMapSwizzle::s128b.apply(0x4080), 0x4080 ^ (1 << 4));
        // Matches CuTe's Swizzle<3,4,3> on byte offsets.
        let sw = TensorMapSwizzle::s128b.to_byte_swizzle();
        for a in (0..2048u64).step_by(16) {
            assert_eq!(sw.apply(a as usize) as u64, TensorMapSwizzle::s128b.apply(a));
        }
    }

    #[test]
    fn box_traversal_counts_strides_and_bounds() {
        let mut d = desc2d();
        d.box_dims[..2].copy_from_slice(&[16, 4]);
        d.element_strides[1] = 2;
        // Box starting at (120, 62): x 120..136 (8 past the end), y 62 and 64 (64 is out).
        let e = d.box_elements(&[120, 62]);
        assert_eq!(e.len(), 16 * 2);
        assert_eq!(e[0].coords[..2], [120, 62]);
        assert_eq!(e[0].global_address, Some(0x1000 + 120 * 2 + 62 * 256));
        assert_eq!(e[7].global_address, Some(0x1000 + 127 * 2 + 62 * 256));
        assert_eq!(e[8].global_address, None);
        assert_eq!(e[16].coords[..2], [120, 64]);
        assert_eq!(e[16].global_address, None);
        assert_eq!(e[16].box_byte, 32);
        assert_eq!(d.box_bytes(), 64);
        // Negative starts are out of bounds too.
        let e = d.box_elements(&[-8, 0]);
        assert_eq!(e[0].global_address, None);
        assert_eq!(e[8].global_address, Some(0x1000));
        // Without interleave dimension 0's element stride is ignored.
        d.element_strides[0] = 4;
        assert_eq!(d.box_extent(0), 16);
    }

    fn rejects(d: &TmaDesc, needle: &str) {
        let e = d.validate().unwrap_err();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    }

    #[test]
    fn validation_follows_the_driver_rules() {
        let ok = desc2d();
        ok.validate().unwrap();
        let mut d = ok.clone();
        d.global_address = 0x1008;
        rejects(&d, "not 16-byte aligned");
        let mut d = ok.clone();
        d.global_strides[0] = 40;
        rejects(&d, "not a multiple of 16");
        let mut d = ok.clone();
        d.global_strides[0] = 1 << 40;
        rejects(&d, "below 2^40");
        let mut d = ok.clone();
        d.global_dims[0] = (1 << 32) + 1;
        rejects(&d, "outside 1 to 2^32");
        let mut d = ok.clone();
        d.global_dims[1] = 0;
        rejects(&d, "outside 1 to 2^32");
        let mut d = ok.clone();
        d.box_dims[1] = 257;
        rejects(&d, "box dimension 1 is 257");
        let mut d = ok.clone();
        d.box_dims[0] = 4;
        rejects(&d, "8 bytes (4 x float16), not a multiple of 16");
        let mut d = ok.clone();
        d.swizzle = TensorMapSwizzle::s64b;
        rejects(&d, "at most 64 bytes, got 128");
        let mut d = ok.clone();
        d.swizzle = TensorMapSwizzle::s128b;
        d.validate().unwrap();
        let mut d = ok.clone();
        d.element_strides[1] = 9;
        rejects(&d, "element stride 1 is 9");
        let mut d = ok.clone();
        d.rank = 6;
        rejects(&d, "rank 6");
        let mut d = ok.clone();
        d.rank = 0;
        rejects(&d, "rank 0");
        let mut d = ok.clone();
        d.interleave = TensorMapInterleave::i16b;
        rejects(&d, "needs rank 3");
        let mut d = TmaDesc::new(TensorMapDataType::float16, 3);
        d.global_address = 0x1010;
        d.global_dims[..3].copy_from_slice(&[8, 16, 16]);
        d.global_strides[..2].copy_from_slice(&[32, 512]);
        d.box_dims[..3].copy_from_slice(&[8, 8, 8]);
        d.interleave = TensorMapInterleave::i32b;
        d.swizzle = TensorMapSwizzle::s32b;
        rejects(&d, "not 32-byte aligned");
        d.global_address = 0x1000;
        d.validate().unwrap();
        d.swizzle = TensorMapSwizzle::s64b;
        rejects(&d, "needs the s32b swizzle");
        let mut d = ok.clone();
        d.data_type = TensorMapDataType::int32;
        d.box_dims[0] = 32;
        d.oob_fill = TensorMapFloatOOBFill::nan_request_zero_fma;
        rejects(&d, "floating-point");
        let mut d = ok.clone();
        d.data_type = TensorMapDataType::f416u4_align16b;
        rejects(&d, "exactly 64 bytes");
        let mut d = ok.clone();
        d.swizzle = TensorMapSwizzle::s128b_atom_64b;
        rejects(&d, "not supported");
    }

    #[test]
    fn spec_evaluation_reads_scalar_arguments() {
        let (m, ld) = (ValueId(7), ValueId(9));
        let spec = TmaDescSpec {
            data_type: TensorMapDataType::float32,
            global_dims: vec![IndexExpr::Static(64), IndexExpr::Dynamic(m)],
            global_strides: vec![IndexExpr::product(IndexExpr::Dynamic(ld), IndexExpr::Static(4))],
            box_dims: vec![32, 16],
            element_strides: vec![1, 1],
            interleave: TensorMapInterleave::none,
            swizzle: TensorMapSwizzle::s128b,
            l2_promotion: TensorMapL2Promotion::none,
            oob_fill: TensorMapFloatOOBFill::none,
            base_offset: IndexExpr::Static(256),
            im2col: None,
        };
        assert_eq!(spec.dependencies(), vec![m, ld]);
        spec.check_static(16).unwrap();
        let env = |v: ValueId| match v.0 {
            7 => Some(100),
            9 => Some(68),
            _ => None,
        };
        let d = spec.evaluate(0x7f00_0000_0000, &env).unwrap();
        assert_eq!(d.global_address, 0x7f00_0000_0100);
        assert_eq!(&d.global_dims[..2], &[64, 100]);
        assert_eq!(d.global_strides[0], 272);
        // A stride that is not a multiple of 16 is a launch-time error.
        let env = |v: ValueId| match v.0 {
            7 => Some(100),
            9 => Some(66),
            _ => None,
        };
        assert!(spec.evaluate(0x7f00_0000_0000, &env).unwrap_err().contains("264 bytes, not a multiple of 16"));
        let none = |_: ValueId| None;
        assert!(spec.evaluate(0, &none).unwrap_err().contains("not a scalar argument"));
        // A static stride that breaks a rule fails the static check.
        let mut bad = spec.clone();
        bad.global_strides = vec![IndexExpr::Static(260)];
        assert!(bad.check_static(16).unwrap_err().contains("not a multiple of 16"));
        let mut bad = spec.clone();
        bad.base_offset = IndexExpr::Static(4);
        assert!(bad.check_static(16).unwrap_err().contains("not 16-byte aligned"));
    }
}

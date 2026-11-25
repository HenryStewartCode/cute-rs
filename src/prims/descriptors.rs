// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm90_desc.hpp (GmmaDescriptor's bit fields and
//   LayoutType values),
//   include/cute/arch/mma_sm100_desc.hpp (UMMA::SmemDescriptor's,
//   UMMA::InstrDescriptor's and UMMA::InstrDescriptorBlockScaled's bit
//   fields, the format codes and the shape rules of the SM100 MMA structs in
//   include/cute/arch/mma_sm100_umma.hpp),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Typed MMA descriptors (roadmap part m11-prims-descriptors-swizzle; CuTe
//! DSL's `cutlass.experimental.primitives` descriptor types).
//!
//! Each type is a bit-level encoder with a `build` that checks every field
//! (and, through `for_mma`, the shapes and types an instruction kind and CTA
//! group take), getters for the encoded fields, and for the shared-memory
//! descriptors `advance_start_address`. Field positions follow the PTX ISA
//! tables (wgmma "Matrix Descriptor Format"; tcgen05 "Shared memory
//! descriptor" and "Instruction descriptor"), cross-checked against CuTe
//! C++'s descriptor unions and CuTe DSL's types; disagreements are noted
//! where the field is encoded.
//!
//! A descriptor is a build-time constant ([`Bits::Static`]) when every field
//! is, and a kernel value ([`Bits::Runtime`]) otherwise, usually because the
//! shared-memory address is only known when the kernel runs. The WGMMA and
//! tcgen05 MMA atoms build their descriptors with these types.

use crate::DType;
use crate::ir::UmmaDataKind;
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::CtaGroup;
use crate::tensor::{AddressSpace, Ptr};
use crate::value::{Numeric, Value};

/// A descriptor's bits, or one of its fields: a constant when everything it
/// is built from is known when the kernel is built, a kernel value
/// otherwise.
#[derive(Clone, Copy, Debug)]
pub enum Bits<T: Numeric> {
    Static(T),
    Runtime(Value<T>),
}

macro_rules! bits_impl {
    ($t:ty) => {
        impl From<$t> for Bits<$t> {
            fn from(v: $t) -> Self {
                Bits::Static(v)
            }
        }

        impl From<Value<$t>> for Bits<$t> {
            fn from(v: Value<$t>) -> Self {
                Bits::Runtime(v)
            }
        }

        impl Bits<$t> {
            /// The constant, if the bits are known when the kernel is built.
            pub fn as_static(&self) -> Option<$t> {
                match self {
                    Bits::Static(v) => Some(*v),
                    Bits::Runtime(_) => None,
                }
            }

            /// The bits as a kernel value (a constant is materialised).
            #[track_caller]
            pub fn value(self) -> Value<$t> {
                match self {
                    Bits::Static(v) => Value::constant(v),
                    Bits::Runtime(v) => v,
                }
            }

            /// The `width`-bit field at bit `lo`.
            #[track_caller]
            fn field(self, lo: u32, width: u32) -> Bits<$t> {
                let mask = ((1u64 << width) - 1) as $t;
                match self {
                    Bits::Static(v) => Bits::Static((v >> lo) & mask),
                    Bits::Runtime(v) => Bits::Runtime((v >> (lo as $t)) & mask),
                }
            }

            /// These bits with the `width`-bit field at `lo` replaced by `f`
            /// (which is masked to the field).
            #[track_caller]
            #[allow(dead_code)]
            fn set_field(self, lo: u32, width: u32, f: Bits<$t>) -> Bits<$t> {
                let mask = (((1u64 << width) - 1) as $t) << lo;
                match (self, f) {
                    (Bits::Static(v), Bits::Static(f)) => Bits::Static((v & !mask) | ((f << lo) & mask)),
                    (v, f) => Bits::Runtime((v.value() & !mask) | ((f.value() << (lo as $t)) & mask)),
                }
            }
        }
    };
}

bits_impl!(u64);
bits_impl!(u32);
bits_impl!(i32);

impl Bits<u32> {
    /// The shared-memory byte address of a pointer (`ptr.toint_i32()`, as an
    /// unsigned value). Panics unless the pointer is into shared memory.
    #[track_caller]
    pub fn smem_address<T: Numeric>(ptr: &Ptr<T>) -> Bits<u32> {
        assert!(
            ptr.memspace() == AddressSpace::Smem,
            "cute::prims: a descriptor's start address is a shared-memory address, but the pointer is in {:?}",
            ptr.memspace()
        );
        Bits::Runtime(ptr.toint_i32().to::<u32>())
    }
}

impl<T: Numeric> From<&Ptr<T>> for Bits<u32> {
    #[track_caller]
    fn from(ptr: &Ptr<T>) -> Self {
        Bits::<u32>::smem_address(ptr)
    }
}

/// A byte offset or address as a descriptor field: a multiple of 16 whose
/// 16-byte count fits `field_bits` bits.
fn units(name: &str, bytes: u64, field_bits: u32) -> Result<u64, String> {
    if !bytes.is_multiple_of(16) {
        return Err(format!("the {name} {bytes:#x} is not a multiple of 16 bytes (the field drops the low 4 bits)"));
    }
    if bytes >> 4 >= 1 << field_bits {
        return Err(format!(
            "the {name} {bytes:#x} does not fit its {field_bits}-bit field of 16-byte units (it must be below {:#x})",
            16u64 << field_bits
        ));
    }
    Ok(bytes >> 4)
}

fn check(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

fn major_bit(m: OperandMajorMode) -> u32 {
    (m == OperandMajorMode::MN) as u32
}

// ---------------------------------------------------------------------------------------
// WGMMA (SM90) shared-memory descriptor.
// ---------------------------------------------------------------------------------------

/// The swizzle a WGMMA shared-memory descriptor names (CuTe DSL's
/// `WgmmaSwizzle`, bits 62-63): none (interleaved), and the 128-, 64- and
/// 32-byte swizzles. The codes differ from tcgen05's ([`Tcgen05SmemSwizzle`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WgmmaSwizzle {
    Interleave = 0,
    Swizzle128B = 1,
    Swizzle64B = 2,
    Swizzle32B = 3,
}

impl WgmmaSwizzle {
    pub const ALL: [WgmmaSwizzle; 4] =
        [WgmmaSwizzle::Interleave, WgmmaSwizzle::Swizzle128B, WgmmaSwizzle::Swizzle64B, WgmmaSwizzle::Swizzle32B];

    /// The swizzle of a 2-bit field value.
    pub fn from_bits(v: u64) -> Option<WgmmaSwizzle> {
        WgmmaSwizzle::ALL.into_iter().find(|s| *s as u64 == v)
    }

    /// The swizzle whose pattern on byte addresses is `Swizzle<bits,4,3>`
    /// (`bits` 0 to 3).
    pub fn from_swizzle_bits(bits: u32) -> Option<WgmmaSwizzle> {
        Some(match bits {
            0 => WgmmaSwizzle::Interleave,
            1 => WgmmaSwizzle::Swizzle32B,
            2 => WgmmaSwizzle::Swizzle64B,
            3 => WgmmaSwizzle::Swizzle128B,
            _ => return None,
        })
    }
}

impl From<crate::nvgpu::warpgroup::GmmaLayoutType> for WgmmaSwizzle {
    fn from(t: crate::nvgpu::warpgroup::GmmaLayoutType) -> Self {
        WgmmaSwizzle::from_bits(t as u64).expect("the GMMA layout types are the WGMMA swizzle codes")
    }
}

/// The shared-memory matrix descriptor of `wgmma.mma_async` (CuTe DSL's
/// `WgmmaSmemDesc`; CuTe C++'s `GmmaDescriptor`), 64 bits:
///
/// | bits    | field                                                    |
/// |---------|----------------------------------------------------------|
/// | 0-13    | start address, `(addr & 0x3FFFF) >> 4`                   |
/// | 16-29   | leading-dimension byte offset `>> 4`                     |
/// | 32-45   | stride-dimension byte offset `>> 4`                      |
/// | 49-51   | base offset (swizzled layouts only)                      |
/// | 62-63   | swizzle ([`WgmmaSwizzle`])                               |
#[derive(Clone, Copy, Debug)]
pub struct WgmmaSmemDesc {
    bits: Bits<u64>,
}

impl WgmmaSmemDesc {
    /// `(lowest bit, width)` of each field.
    pub const START_ADDRESS: (u32, u32) = (0, 14);
    pub const LEADING_BYTE_OFFSET: (u32, u32) = (16, 14);
    pub const STRIDE_BYTE_OFFSET: (u32, u32) = (32, 14);
    pub const BASE_OFFSET: (u32, u32) = (49, 3);
    pub const LAYOUT_TYPE: (u32, u32) = (62, 2);

    /// A descriptor from its 64 bits (not checked).
    pub fn from_bits(bits: impl Into<Bits<u64>>) -> WgmmaSmemDesc {
        WgmmaSmemDesc { bits: bits.into() }
    }

    /// The descriptor of an operand at shared-memory byte address
    /// `start_address` (a constant, a `Value<u32>`, or a shared-memory
    /// pointer, `(&ptr).into()`), with the leading and stride byte offsets
    /// in bytes (multiples of 16 below 256 KiB), the base offset (0 to 7;
    /// only with a swizzle) and the swizzle. Fails, saying which field is
    /// out of range, otherwise.
    #[track_caller]
    pub fn build(
        start_address: impl Into<Bits<u32>>,
        leading_byte_offset: u32,
        stride_byte_offset: u32,
        base_offset: u8,
        swizzle: WgmmaSwizzle,
    ) -> Result<WgmmaSmemDesc, String> {
        let lbo = units("leading byte offset", leading_byte_offset.into(), 14)?;
        let sbo = units("stride byte offset", stride_byte_offset.into(), 14)?;
        check(base_offset < 8, || format!("the base offset {base_offset} does not fit its 3-bit field"))?;
        check(base_offset == 0 || swizzle != WgmmaSwizzle::Interleave, || {
            format!("a base offset ({base_offset}) only applies to a swizzled layout, not to an interleaved one")
        })?;
        let rest = lbo << 16 | sbo << 32 | u64::from(base_offset) << 49 | (swizzle as u64) << 62;
        let desc = WgmmaSmemDesc { bits: Bits::Static(rest) };
        match start_address.into() {
            Bits::Static(a) => {
                let start = units("start address", (a & 0x3FFFF).into(), 14)?;
                check(a < 1 << 18, || format!("the start address {a:#x} is past the 256 KiB the field addresses"))?;
                Ok(WgmmaSmemDesc { bits: Bits::Static(rest | start) })
            }
            Bits::Runtime(a) => Ok(desc.with_start_address(a)),
        }
    }

    /// This descriptor, whose start address is 0, with the start address
    /// `addr` ORed in at run time (`(addr >> 4) & 0x3FFF`; the low 4 bits
    /// are dropped unchecked).
    #[track_caller]
    pub fn with_start_address(self, addr: Value<u32>) -> WgmmaSmemDesc {
        let start = (addr.to::<u64>() >> 4u64) & 0x3fffu64;
        let bits = match self.bits {
            Bits::Static(c) => start | c,
            Bits::Runtime(v) => start | v,
        };
        WgmmaSmemDesc { bits: Bits::Runtime(bits) }
    }

    /// The 64 bits.
    pub fn bits(&self) -> Bits<u64> {
        self.bits
    }

    /// The 64 bits as a kernel value.
    #[track_caller]
    pub fn value(&self) -> Value<u64> {
        self.bits.value()
    }

    /// The start address field (the address `>> 4`).
    pub fn start_address(&self) -> Bits<u64> {
        self.bits.field(Self::START_ADDRESS.0, Self::START_ADDRESS.1)
    }

    /// The leading byte offset field (bytes `>> 4`).
    pub fn leading_byte_offset(&self) -> Bits<u64> {
        self.bits.field(Self::LEADING_BYTE_OFFSET.0, Self::LEADING_BYTE_OFFSET.1)
    }

    /// The stride byte offset field (bytes `>> 4`).
    pub fn stride_byte_offset(&self) -> Bits<u64> {
        self.bits.field(Self::STRIDE_BYTE_OFFSET.0, Self::STRIDE_BYTE_OFFSET.1)
    }

    /// The base offset field.
    pub fn base_offset(&self) -> Bits<u64> {
        self.bits.field(Self::BASE_OFFSET.0, Self::BASE_OFFSET.1)
    }

    /// The swizzle field (a [`WgmmaSwizzle`] code).
    pub fn layout_type(&self) -> Bits<u64> {
        self.bits.field(Self::LAYOUT_TYPE.0, Self::LAYOUT_TYPE.1)
    }

    /// The swizzle, for a descriptor known when the kernel is built.
    pub fn swizzle(&self) -> Option<WgmmaSwizzle> {
        self.layout_type().as_static().and_then(WgmmaSwizzle::from_bits)
    }

    /// The descriptor with its start address advanced by `bytes` (a
    /// multiple of 16; a constant offset with any of its low 4 bits set is
    /// refused). The addition is confined to the 14-bit field (a carry does
    /// not reach the other fields).
    #[track_caller]
    pub fn advance_start_address(&self, bytes: impl Into<Bits<u64>>) -> Result<WgmmaSmemDesc, String> {
        let inc = match bytes.into() {
            Bits::Static(b) => {
                check(b.is_multiple_of(16), || {
                    format!("the byte increment {b} has low bits the 16-byte start-address field cannot hold")
                })?;
                Bits::Static(b >> 4)
            }
            Bits::Runtime(b) => Bits::Runtime(b >> 4u64),
        };
        let (lo, w) = Self::START_ADDRESS;
        let sum = match (self.start_address(), inc) {
            (Bits::Static(s), Bits::Static(i)) => Bits::Static(s.wrapping_add(i)),
            (s, i) => Bits::Runtime(s.value() + i.value()),
        };
        Ok(WgmmaSmemDesc { bits: self.bits.set_field(lo, w, sum) })
    }
}

// ---------------------------------------------------------------------------------------
// tcgen05 (SM100) shared-memory descriptor.
// ---------------------------------------------------------------------------------------

/// The swizzle a tcgen05 shared-memory descriptor names (CuTe DSL's
/// `Tcgen05SmemSwizzle`, bits 61-63): `SwizzleNone` (0),
/// `Swizzle128BBase32B` (the 128-byte swizzle with 32-byte atoms, 1),
/// `Swizzle128B` (2), `Swizzle64B` (4) and `Swizzle32B` (6); 3, 5 and 7 are
/// reserved. The same type as the UMMA atoms' [`UmmaLayoutType`](crate::nvgpu::tcgen05::UmmaLayoutType).
pub type Tcgen05SmemSwizzle = crate::nvgpu::tcgen05::UmmaLayoutType;

/// The fields of a [`Tcgen05SmemDesc`] other than the start address.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tcgen05SmemDescFields {
    /// The leading-dimension byte offset in bytes (relative), or the
    /// address it names (absolute mode); a multiple of 16 below 512 KiB.
    pub leading_byte_offset: u32,
    /// The stride-dimension byte offset in bytes; a multiple of 16 below
    /// 256 KiB.
    pub stride_byte_offset: u32,
    /// The swizzle pattern's start row (0 to 7).
    pub base_offset: u8,
    /// The leading-dimension stride mode (bit 52): relative offsets
    /// (`false`, CuTe's default) or absolute addresses.
    pub absolute_lbo: bool,
    /// The `lut::b` K-segment start offset (bit 53).
    pub k_segment_offset: bool,
    pub swizzle: Tcgen05SmemSwizzle,
}

impl Default for Tcgen05SmemDescFields {
    fn default() -> Self {
        Tcgen05SmemDescFields {
            leading_byte_offset: 0,
            stride_byte_offset: 0,
            base_offset: 0,
            absolute_lbo: false,
            k_segment_offset: false,
            swizzle: Tcgen05SmemSwizzle::SwizzleNone,
        }
    }
}

/// The shared-memory descriptor of `tcgen05.mma` (CuTe DSL's
/// `Tcgen05SmemDesc`; CuTe C++'s `UMMA::SmemDescriptor`), 64 bits:
///
/// | bits    | field                                                       |
/// |---------|-------------------------------------------------------------|
/// | 0-14    | start address `>> 4`                                        |
/// | 16-30   | leading-dimension byte offset (or absolute address) `>> 4`  |
/// | 32-45   | stride-dimension byte offset `>> 4`                         |
/// | 46-48   | version, fixed `0b001`                                      |
/// | 49-51   | base offset                                                 |
/// | 52      | leading-dimension stride mode (0 relative, 1 absolute)      |
/// | 53      | `lut::b` K-segment start offset                             |
/// | 61-63   | swizzle ([`Tcgen05SmemSwizzle`])                            |
///
/// The current PTX ISA widens the start address and LBO to 15 bits; CuTe
/// C++ gives them 14 bits on sm_100 (bit 14 and 30 reserved) and 15 on
/// sm_107, and its version field 2 bits (46-47). The two agree on every
/// address below 256 KiB, which is all of an SM100 CTA's shared memory.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05SmemDesc {
    bits: Bits<u64>,
}

impl Tcgen05SmemDesc {
    pub const START_ADDRESS: (u32, u32) = (0, 15);
    pub const LEADING_BYTE_OFFSET: (u32, u32) = (16, 15);
    pub const STRIDE_BYTE_OFFSET: (u32, u32) = (32, 14);
    pub const VERSION: (u32, u32) = (46, 3);
    pub const BASE_OFFSET: (u32, u32) = (49, 3);
    pub const LBO_MODE: (u32, u32) = (52, 1);
    pub const K_SEGMENT_OFFSET: (u32, u32) = (53, 1);
    pub const LAYOUT_TYPE: (u32, u32) = (61, 3);

    /// A descriptor from its 64 bits (not checked).
    pub fn from_bits(bits: impl Into<Bits<u64>>) -> Tcgen05SmemDesc {
        Tcgen05SmemDesc { bits: bits.into() }
    }

    /// Pack fields already in their encoded units (the single place the
    /// bit positions are written; the UMMA atoms' encoder calls it too).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pack(
        start: u64,
        lbo: u64,
        sbo: u64,
        version: u64,
        base_offset: u64,
        absolute_lbo: bool,
        k_segment_offset: bool,
        swizzle: Tcgen05SmemSwizzle,
    ) -> u64 {
        start
            | lbo << Self::LEADING_BYTE_OFFSET.0
            | sbo << Self::STRIDE_BYTE_OFFSET.0
            | version << Self::VERSION.0
            | base_offset << Self::BASE_OFFSET.0
            | (absolute_lbo as u64) << Self::LBO_MODE.0
            | (k_segment_offset as u64) << Self::K_SEGMENT_OFFSET.0
            | (swizzle as u64) << Self::LAYOUT_TYPE.0
    }

    /// The descriptor of an operand at shared-memory byte address
    /// `start_address` (a constant, a `Value<u32>`, or a shared-memory
    /// pointer) with `fields`; version 1. Fails, saying which field is out
    /// of range, otherwise.
    #[track_caller]
    pub fn build(
        start_address: impl Into<Bits<u32>>,
        fields: &Tcgen05SmemDescFields,
    ) -> Result<Tcgen05SmemDesc, String> {
        let lbo = units("leading byte offset", fields.leading_byte_offset.into(), 15)?;
        let sbo = units("stride byte offset", fields.stride_byte_offset.into(), 14)?;
        check(fields.base_offset < 8, || {
            format!("the base offset {} does not fit its 3-bit field", fields.base_offset)
        })?;
        let rest = Self::pack(
            0,
            lbo,
            sbo,
            1,
            fields.base_offset.into(),
            fields.absolute_lbo,
            fields.k_segment_offset,
            fields.swizzle,
        );
        match start_address.into() {
            Bits::Static(a) => {
                let start = units("start address", a.into(), 15)?;
                Ok(Tcgen05SmemDesc { bits: Bits::Static(rest | start) })
            }
            Bits::Runtime(a) => Ok(Tcgen05SmemDesc { bits: Bits::Static(rest) }.with_start_address(a)),
        }
    }

    /// This descriptor, whose start address is 0, with the start address
    /// `addr` ORed in at run time (`(addr >> 4) & 0x7FFF`).
    #[track_caller]
    pub fn with_start_address(self, addr: Value<u32>) -> Tcgen05SmemDesc {
        self.with_start_address_bits(addr, Self::START_ADDRESS.1)
    }

    /// [`with_start_address`](Self::with_start_address) masking the start
    /// to `bits` bits (14 for CuTe C++'s sm_100 encoding).
    #[track_caller]
    pub(crate) fn with_start_address_bits(self, addr: Value<u32>, bits: u32) -> Tcgen05SmemDesc {
        let start = (addr.to::<u64>() >> 4u64) & ((1u64 << bits) - 1);
        let bits = match self.bits {
            Bits::Static(c) => start | c,
            Bits::Runtime(v) => start | v,
        };
        Tcgen05SmemDesc { bits: Bits::Runtime(bits) }
    }

    pub fn bits(&self) -> Bits<u64> {
        self.bits
    }

    #[track_caller]
    pub fn value(&self) -> Value<u64> {
        self.bits.value()
    }

    pub fn start_address(&self) -> Bits<u64> {
        self.bits.field(Self::START_ADDRESS.0, Self::START_ADDRESS.1)
    }

    pub fn leading_byte_offset(&self) -> Bits<u64> {
        self.bits.field(Self::LEADING_BYTE_OFFSET.0, Self::LEADING_BYTE_OFFSET.1)
    }

    pub fn stride_byte_offset(&self) -> Bits<u64> {
        self.bits.field(Self::STRIDE_BYTE_OFFSET.0, Self::STRIDE_BYTE_OFFSET.1)
    }

    pub fn version(&self) -> Bits<u64> {
        self.bits.field(Self::VERSION.0, Self::VERSION.1)
    }

    pub fn base_offset(&self) -> Bits<u64> {
        self.bits.field(Self::BASE_OFFSET.0, Self::BASE_OFFSET.1)
    }

    /// The leading-dimension stride mode bit (1: absolute).
    pub fn leading_dim_mode(&self) -> Bits<u64> {
        self.bits.field(Self::LBO_MODE.0, Self::LBO_MODE.1)
    }

    pub fn k_segment_offset(&self) -> Bits<u64> {
        self.bits.field(Self::K_SEGMENT_OFFSET.0, Self::K_SEGMENT_OFFSET.1)
    }

    /// The swizzle field (a [`Tcgen05SmemSwizzle`] code, or a reserved one).
    pub fn layout_type(&self) -> Bits<u64> {
        self.bits.field(Self::LAYOUT_TYPE.0, Self::LAYOUT_TYPE.1)
    }

    /// The swizzle, for a descriptor known when the kernel is built.
    pub fn swizzle(&self) -> Option<Tcgen05SmemSwizzle> {
        self.layout_type().as_static().and_then(Tcgen05SmemSwizzle::from_bits)
    }

    /// The descriptor with its start address advanced by `bytes` (a
    /// multiple of 16; a constant offset with any of its low 4 bits set is
    /// refused): a plain 64-bit add of `bytes >> 4`, as the MMA loops issue
    /// it, so the caller keeps the start address within its field (checked
    /// when everything is constant).
    #[track_caller]
    pub fn advance_start_address(&self, bytes: impl Into<Bits<u64>>) -> Result<Tcgen05SmemDesc, String> {
        let bits = match (self.bits, bytes.into()) {
            (d, Bits::Static(b)) => {
                check(b.is_multiple_of(16), || {
                    format!("the byte increment {b} has low bits the 16-byte start-address field cannot hold")
                })?;
                match d {
                    Bits::Static(d) => {
                        let start = (d & 0x7fff) + (b >> 4);
                        check(start < 1 << 15, || {
                            format!("advancing by {b} bytes takes the start address past its 15-bit field")
                        })?;
                        Bits::Static(d + (b >> 4))
                    }
                    Bits::Runtime(d) => Bits::Runtime(d + (b >> 4)),
                }
            }
            (d, Bits::Runtime(b)) => Bits::Runtime(d.value() + (b >> 4u64)),
        };
        Ok(Tcgen05SmemDesc { bits })
    }
}

// ---------------------------------------------------------------------------------------
// tcgen05 instruction descriptors.
// ---------------------------------------------------------------------------------------

/// The accumulator format of a tcgen05 instruction descriptor (bits 4-5;
/// CuTe DSL's `Tcgen05OutputFormat`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tcgen05OutputFormat {
    F16 = 0,
    F32 = 1,
    S32 = 2,
}

impl Tcgen05OutputFormat {
    /// The format of an accumulator type: Float16, Float32 or Int32.
    pub fn for_dtype(t: DType) -> Option<Tcgen05OutputFormat> {
        Some(match t {
            DType::F16 => Tcgen05OutputFormat::F16,
            DType::F32 => Tcgen05OutputFormat::F32,
            DType::I32 => Tcgen05OutputFormat::S32,
            _ => return None,
        })
    }
}

/// An operand format code of a tcgen05 instruction descriptor (CuTe DSL's
/// `Tcgen05InputFormat`). The codes depend on the instruction kind, so
/// several names share a value: `F16`, `BF16`, `TF32` for `.kind::f16` and
/// `.kind::tf32`; `E4M3`, `E5M2`, `E2M3`, `E3M2`, `E2M1` for
/// `.kind::f8f6f4` and `.kind::mxf8f6f4`; `U8`, `S8` for `.kind::i8`; and
/// `MX_E2M1` (1) for `.kind::mxf4` and `.kind::mxf4nvf4`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tcgen05InputFormat(pub u8);

impl Tcgen05InputFormat {
    pub const F16: Tcgen05InputFormat = Tcgen05InputFormat(0);
    pub const BF16: Tcgen05InputFormat = Tcgen05InputFormat(1);
    pub const TF32: Tcgen05InputFormat = Tcgen05InputFormat(2);
    pub const E4M3: Tcgen05InputFormat = Tcgen05InputFormat(0);
    pub const E5M2: Tcgen05InputFormat = Tcgen05InputFormat(1);
    pub const E2M3: Tcgen05InputFormat = Tcgen05InputFormat(3);
    pub const E3M2: Tcgen05InputFormat = Tcgen05InputFormat(4);
    pub const E2M1: Tcgen05InputFormat = Tcgen05InputFormat(5);
    pub const U8: Tcgen05InputFormat = Tcgen05InputFormat(0);
    pub const S8: Tcgen05InputFormat = Tcgen05InputFormat(1);
    pub const MX_E2M1: Tcgen05InputFormat = Tcgen05InputFormat(1);

    /// The code of an operand of type `t` under `kind`, if the kind takes
    /// the type.
    pub fn for_dtype(kind: UmmaDataKind, t: DType) -> Option<Tcgen05InputFormat> {
        kind.operand_format(t).map(|v| Tcgen05InputFormat(v as u8))
    }
}

/// The scale-factor format of a block-scaled instruction descriptor (CuTe
/// DSL's `MxScaleFormat`; bits 23-24).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MxScaleFormat {
    UE4M3 = 0,
    E8M0 = 1,
    /// sm_107 only.
    UE5M3 = 2,
}

impl MxScaleFormat {
    /// The format of a scale-factor type: `F8E4M3FN`, `F8E8M0FNU` or
    /// `F8E5M3FNU`.
    pub fn for_dtype(t: DType) -> Option<MxScaleFormat> {
        Some(match t {
            DType::F8E4M3FN => MxScaleFormat::UE4M3,
            DType::F8E8M0FNU => MxScaleFormat::E8M0,
            DType::F8E5M3FNU => MxScaleFormat::UE5M3,
            _ => return None,
        })
    }
}

/// The checks of N and M (the instruction's, a CTA pair's for
/// `CtaGroup::TWO`) common to the descriptors: `n` a multiple of 8 below
/// 512 and `m` a multiple of `1 << m_drop` that fits `m_bits`.
fn check_dims(n: u32, m: u32, n_min: u32, m_drop: u32, m_bits: u32) -> Result<(), String> {
    check(n.is_multiple_of(8) && n >= n_min && n >> 3 < 64, || {
        format!("N = {n} is not a multiple of 8 from {n_min} to 504 (the field holds N >> 3 in 6 bits)")
    })?;
    check(m.is_multiple_of(1 << m_drop) && m >> m_drop < 1 << m_bits && m > 0, || {
        format!(
            "M = {m} is not a non-zero multiple of {} that fits the field (M >> {m_drop} in {m_bits} bits)",
            1 << m_drop
        )
    })
}

/// The MMA shapes (the PTX ISA's and CuTe C++'s SM100 MMA structs): with
/// `CtaGroup::ONE`, M of 64 or 128 (dense) or 128 (block-scaled) and N a
/// multiple of 8 from 8 to 256 (`.kind::i8`: 8, or a multiple of 16 from 16);
/// with `CtaGroup::TWO`, M of 128 or 256 (dense) or 256 (block-scaled) and
/// N a multiple of 16 from 16 to 256.
pub(crate) fn check_shape(kind: UmmaDataKind, m: u32, n: u32, cta_group: CtaGroup) -> Result<(), String> {
    let scaled = matches!(kind, UmmaDataKind::Mxf8f6f4 | UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4);
    let two = cta_group == CtaGroup::TWO;
    let ms: &[u32] = match (scaled, two) {
        (false, false) => &[64, 128],
        (false, true) => &[128, 256],
        (true, false) => &[128],
        (true, true) => &[256],
    };
    let k = kind.ptx();
    check(ms.contains(&m), || format!(".kind::{k} with cta_group::{} takes M of {ms:?}, not {m}", cta_group as u32))?;
    let n_ok = if two {
        n.is_multiple_of(16) && (16..=256).contains(&n)
    } else if kind == UmmaDataKind::I8 {
        n == 8 || (n.is_multiple_of(16) && (16..=256).contains(&n))
    } else {
        n.is_multiple_of(8) && (8..=256).contains(&n)
    };
    check(n_ok, || {
        format!(
            ".kind::{k} with cta_group::{} takes N {}, not {n}",
            cta_group as u32,
            match (two, kind) {
                (true, _) => "a multiple of 16 from 16 to 256",
                (false, UmmaDataKind::I8) => "of 8 or a multiple of 16 from 16 to 256",
                _ => "a multiple of 8 from 8 to 256",
            }
        )
    })
}

/// MN-major operands are only for 8-, 16- and 32-bit types (CuTe C++'s
/// descriptor notes: not FP6 or FP4).
fn check_major(t: DType, major: OperandMajorMode, which: &str) -> Result<(), String> {
    check(major == OperandMajorMode::K || t.bits() >= 8, || {
        format!("{which} is {t}, which a tcgen05 MMA only reads K-major")
    })
}

fn input_format(kind: UmmaDataKind, t: DType, which: &str) -> Result<Tcgen05InputFormat, String> {
    Tcgen05InputFormat::for_dtype(kind, t).ok_or_else(|| format!(".kind::{} does not take {t} for {which}", kind.ptx()))
}

/// The fields of a [`Tcgen05InstrDesc`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tcgen05InstrDescFields {
    /// The sparse metadata selector (0 to 3).
    pub sparse_id2: u8,
    pub sparse: bool,
    /// Saturate (`.kind::i8` only).
    pub saturate: bool,
    pub c_format: Tcgen05OutputFormat,
    pub a_format: Tcgen05InputFormat,
    pub b_format: Tcgen05InputFormat,
    pub a_negate: bool,
    pub b_negate: bool,
    pub a_major: OperandMajorMode,
    pub b_major: OperandMajorMode,
    /// N (a multiple of 8).
    pub n: u32,
    /// M (a multiple of 16).
    pub m: u32,
    /// The K-size bit (29): K = 64 of the sm_107 `.kind::f8f6f4` forms
    /// (reserved on sm_100).
    pub k_dim: bool,
    /// The maximum shift of `.ws` B reuse (0 to 3).
    pub max_shift: u8,
}

impl Default for Tcgen05InstrDescFields {
    fn default() -> Self {
        Tcgen05InstrDescFields {
            sparse_id2: 0,
            sparse: false,
            saturate: false,
            c_format: Tcgen05OutputFormat::F32,
            a_format: Tcgen05InputFormat(0),
            b_format: Tcgen05InputFormat(0),
            a_negate: false,
            b_negate: false,
            a_major: OperandMajorMode::K,
            b_major: OperandMajorMode::K,
            n: 0,
            m: 0,
            k_dim: false,
            max_shift: 0,
        }
    }
}

/// The instruction descriptor of `tcgen05.mma` for `.kind::tf32`, `f16`,
/// `f8f6f4` and `i8` (CuTe DSL's `Tcgen05InstrDesc`; CuTe C++'s
/// `UMMA::InstrDescriptor`), 32 bits:
///
/// | bits  | field                         | bits  | field                       |
/// |-------|-------------------------------|-------|-----------------------------|
/// | 0-1   | sparse metadata selector      | 14    | negate B                    |
/// | 2     | sparse                        | 15    | A major (1: MN)             |
/// | 3     | saturate                      | 16    | B major (1: MN)             |
/// | 4-5   | D format                      | 17-22 | N >> 3                      |
/// | 7-9   | A format                      | 24-28 | M >> 4                      |
/// | 10-12 | B format                      | 29    | K size (sm_107)             |
/// | 13    | negate A                      | 30-31 | max shift (`.ws`)           |
///
/// CuTe DSL also names bit 6 (`sparse_format`), which the PTX ISA table
/// and CuTe C++ leave reserved; cute-rs leaves it 0.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05InstrDesc {
    bits: Bits<u32>,
}

impl Tcgen05InstrDesc {
    /// A descriptor from its 32 bits (not checked).
    pub fn from_bits(bits: impl Into<Bits<u32>>) -> Tcgen05InstrDesc {
        Tcgen05InstrDesc { bits: bits.into() }
    }

    /// Pack `f`, checking that each field fits and N and M are multiples of
    /// 8 and 16. The shapes a kind takes are checked by
    /// [`for_mma`](Self::for_mma).
    pub fn build(f: &Tcgen05InstrDescFields) -> Result<Tcgen05InstrDesc, String> {
        check(f.sparse_id2 < 4, || format!("the sparse selector {} does not fit its 2-bit field", f.sparse_id2))?;
        check(f.a_format.0 < 8 && f.b_format.0 < 8, || {
            format!("the operand formats {} and {} do not fit their 3-bit fields", f.a_format.0, f.b_format.0)
        })?;
        check(f.max_shift < 4, || format!("the max shift {} does not fit its 2-bit field", f.max_shift))?;
        check_dims(f.n, f.m, 8, 4, 5)?;
        let bits = u32::from(f.sparse_id2)
            | (f.sparse as u32) << 2
            | (f.saturate as u32) << 3
            | (f.c_format as u32) << 4
            | u32::from(f.a_format.0) << 7
            | u32::from(f.b_format.0) << 10
            | (f.a_negate as u32) << 13
            | (f.b_negate as u32) << 14
            | major_bit(f.a_major) << 15
            | major_bit(f.b_major) << 16
            | (f.n >> 3) << 17
            | (f.m >> 4) << 24
            | (f.k_dim as u32) << 29
            | u32::from(f.max_shift) << 30;
        Ok(Tcgen05InstrDesc { bits: Bits::Static(bits) })
    }

    /// The descriptor of a dense MMA of `kind` (`F16`, `Tf32`, `I8` or
    /// `F8F6F4`) on `a` and `b` into `acc`, `m x n` (a CTA pair's M for
    /// `CtaGroup::TWO`), checking the types the kind takes, the shapes of the
    /// kind and CTA group, and that FP6/FP4 operands are K-major.
    #[allow(clippy::too_many_arguments)]
    pub fn for_mma(
        kind: UmmaDataKind,
        a: DType,
        b: DType,
        acc: DType,
        (m, n): (u32, u32),
        cta_group: CtaGroup,
        a_major: OperandMajorMode,
        b_major: OperandMajorMode,
    ) -> Result<Tcgen05InstrDesc, String> {
        check(
            matches!(kind, UmmaDataKind::F16 | UmmaDataKind::Tf32 | UmmaDataKind::I8 | UmmaDataKind::F8F6F4),
            || format!(".kind::{} is block-scaled: use Tcgen05MxInstrDesc or Tcgen05MxOmmaInstrDesc", kind.ptx()),
        )?;
        let (a_format, b_format) = (input_format(kind, a, "A")?, input_format(kind, b, "B")?);
        let accs: &[DType] = match kind {
            UmmaDataKind::F16 if a == DType::F16 && b == DType::F16 => &[DType::F16, DType::F32],
            UmmaDataKind::F16 | UmmaDataKind::Tf32 => &[DType::F32],
            UmmaDataKind::I8 => &[DType::I32],
            _ => &[DType::F16, DType::F32],
        };
        check(accs.contains(&acc), || {
            format!(".kind::{} on {a} x {b} accumulates in {accs:?}, not {acc}", kind.ptx())
        })?;
        check_shape(kind, m, n, cta_group)?;
        check_major(a, a_major, "A")?;
        check_major(b, b_major, "B")?;
        Tcgen05InstrDesc::build(&Tcgen05InstrDescFields {
            c_format: Tcgen05OutputFormat::for_dtype(acc).expect("checked above"),
            a_format,
            b_format,
            a_major,
            b_major,
            n,
            m,
            ..Default::default()
        })
    }

    pub fn bits(&self) -> Bits<u32> {
        self.bits
    }

    #[track_caller]
    pub fn value(&self) -> Value<u32> {
        self.bits.value()
    }

    pub fn sparse_id2(&self) -> Bits<u32> {
        self.bits.field(0, 2)
    }
    pub fn sparse_flag(&self) -> Bits<u32> {
        self.bits.field(2, 1)
    }
    pub fn saturate(&self) -> Bits<u32> {
        self.bits.field(3, 1)
    }
    pub fn c_format(&self) -> Bits<u32> {
        self.bits.field(4, 2)
    }
    pub fn a_format(&self) -> Bits<u32> {
        self.bits.field(7, 3)
    }
    pub fn b_format(&self) -> Bits<u32> {
        self.bits.field(10, 3)
    }
    pub fn a_negate(&self) -> Bits<u32> {
        self.bits.field(13, 1)
    }
    pub fn b_negate(&self) -> Bits<u32> {
        self.bits.field(14, 1)
    }
    pub fn a_major(&self) -> Bits<u32> {
        self.bits.field(15, 1)
    }
    pub fn b_major(&self) -> Bits<u32> {
        self.bits.field(16, 1)
    }
    /// N >> 3.
    pub fn n_dim(&self) -> Bits<u32> {
        self.bits.field(17, 6)
    }
    /// M >> 4.
    pub fn m_dim(&self) -> Bits<u32> {
        self.bits.field(24, 5)
    }
    pub fn k_dim(&self) -> Bits<u32> {
        self.bits.field(29, 1)
    }
    pub fn max_shift(&self) -> Bits<u32> {
        self.bits.field(30, 2)
    }

    /// The descriptor with the sparse metadata selector set (masked to its
    /// field).
    #[track_caller]
    pub fn set_sparse_id2(&self, id2: impl Into<Bits<u32>>) -> Tcgen05InstrDesc {
        Tcgen05InstrDesc { bits: self.bits.set_field(0, 2, id2.into()) }
    }
}

/// The fields of a [`Tcgen05MxInstrDesc`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tcgen05MxInstrDescFields {
    pub sparse: bool,
    /// B's scale-factor ID (0 to 3).
    pub b_sf_id: u8,
    /// A's scale-factor ID (0 to 3).
    pub a_sf_id: u8,
    pub a_format: Tcgen05InputFormat,
    pub b_format: Tcgen05InputFormat,
    pub scale_format: MxScaleFormat,
    pub a_negate: bool,
    pub b_negate: bool,
    pub a_major: OperandMajorMode,
    pub b_major: OperandMajorMode,
    /// N (a multiple of 8 from 8 to 256).
    pub n: u32,
    /// M: 128 or 256.
    pub m: u32,
    /// The K-size bit (31): K = 64 of the sm_107 form; the sm_103 K = 96
    /// forms set it too.
    pub k_dim: bool,
    /// A's scale-factor layout bit (26; reserved on sm_100).
    pub a_sf_layout: bool,
}

impl Default for Tcgen05MxInstrDescFields {
    fn default() -> Self {
        Tcgen05MxInstrDescFields {
            sparse: false,
            b_sf_id: 0,
            a_sf_id: 0,
            a_format: Tcgen05InputFormat::E4M3,
            b_format: Tcgen05InputFormat::E4M3,
            scale_format: MxScaleFormat::E8M0,
            a_negate: false,
            b_negate: false,
            a_major: OperandMajorMode::K,
            b_major: OperandMajorMode::K,
            n: 0,
            m: 0,
            k_dim: false,
            a_sf_layout: false,
        }
    }
}

/// The scale-factor format of `sf` for `kind`.
fn scale_format(kind: UmmaDataKind, sf: DType) -> Result<MxScaleFormat, String> {
    let f = MxScaleFormat::for_dtype(sf)
        .ok_or_else(|| format!("{sf} is no scale-factor type (F8E4M3FN, F8E8M0FNU or F8E5M3FNU)"))?;
    let ok: &[MxScaleFormat] = match kind {
        UmmaDataKind::Mxf8f6f4 | UmmaDataKind::Mxf4 => &[MxScaleFormat::E8M0],
        _ => &[MxScaleFormat::UE4M3, MxScaleFormat::E8M0, MxScaleFormat::UE5M3],
    };
    check(ok.contains(&f), || format!(".kind::{} does not take {sf} scale factors", kind.ptx()))?;
    Ok(f)
}

/// The instruction descriptor of block-scaled `tcgen05.mma.kind::mxf8f6f4`
/// (CuTe DSL's `Tcgen05MxInstrDesc`; CuTe C++'s
/// `UMMA::InstrDescriptorBlockScaled`), 32 bits: sparse 2, B's
/// scale-factor ID 4-5, A format 7-9, B format 10-12, negate A 13, negate
/// B 14, A major 15, B major 16, N >> 3 in 17-22, scale format 23-24
/// ([`MxScaleFormat`]), A's scale-factor layout 26, M >> 7 in 27-28, A's
/// scale-factor ID 29-30 and the K size 31.
///
/// CuTe C++ writes M >> 4 into bits 24-28 and CuTe DSL M >> 7 into 27-28:
/// for the only Ms block-scaled MMAs take, 128 and 256, the bits are the
/// same.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05MxInstrDesc {
    bits: Bits<u32>,
}

/// The common part of the block-scaled encodings.
#[allow(clippy::too_many_arguments)]
fn mx_bits(
    sparse: bool,
    a_negate: bool,
    b_negate: bool,
    a_major: OperandMajorMode,
    b_major: OperandMajorMode,
    (m, n): (u32, u32),
    scale: MxScaleFormat,
    a_sf_layout: bool,
    (a_sf_id, b_sf_id): (u8, u8),
) -> Result<u32, String> {
    check(a_sf_id < 4 && b_sf_id < 4, || {
        format!("the scale-factor IDs {a_sf_id} and {b_sf_id} do not fit their 2-bit fields")
    })?;
    check_dims(n, m, 8, 7, 2)?;
    check(n <= 256, || format!("N = {n} is past 256"))?;
    Ok((sparse as u32) << 2
        | u32::from(b_sf_id) << 4
        | (a_negate as u32) << 13
        | (b_negate as u32) << 14
        | major_bit(a_major) << 15
        | major_bit(b_major) << 16
        | (n >> 3) << 17
        | (scale as u32) << 23
        | (a_sf_layout as u32) << 26
        | (m >> 7) << 27
        | u32::from(a_sf_id) << 29)
}

fn mx_set_sf_ids(bits: Bits<u32>, a: Bits<u32>, b: Bits<u32>) -> Bits<u32> {
    bits.set_field(4, 2, b).set_field(29, 2, a)
}

impl Tcgen05MxInstrDesc {
    pub fn from_bits(bits: impl Into<Bits<u32>>) -> Tcgen05MxInstrDesc {
        Tcgen05MxInstrDesc { bits: bits.into() }
    }

    /// Pack `f`, checking that each field fits, the formats are FP8, FP6 or
    /// FP4 codes (0, 1, 3, 4, 5), N is a multiple of 8 from 8 to 256 and M is
    /// 128 or 256.
    pub fn build(f: &Tcgen05MxInstrDescFields) -> Result<Tcgen05MxInstrDesc, String> {
        for (which, t) in [("A", f.a_format), ("B", f.b_format)] {
            check(matches!(t.0, 0 | 1 | 3 | 4 | 5), || {
                format!("the {which} format {} is not an FP8, FP6 or FP4 code (0, 1, 3, 4 or 5)", t.0)
            })?;
        }
        let bits = mx_bits(
            f.sparse,
            f.a_negate,
            f.b_negate,
            f.a_major,
            f.b_major,
            (f.m, f.n),
            f.scale_format,
            f.a_sf_layout,
            (f.a_sf_id, f.b_sf_id),
        )? | u32::from(f.a_format.0) << 7
            | u32::from(f.b_format.0) << 10
            | (f.k_dim as u32) << 31;
        Ok(Tcgen05MxInstrDesc { bits: Bits::Static(bits) })
    }

    /// The descriptor of `.kind::mxf8f6f4` on `a` and `b` with `sf` scale
    /// factors, `m x n` (a CTA pair's M for `CtaGroup::TWO`), checking the
    /// types, the shapes of the CTA group and that FP6/FP4 operands are
    /// K-major.
    pub fn for_mma(
        a: DType,
        b: DType,
        sf: DType,
        (m, n): (u32, u32),
        cta_group: CtaGroup,
        a_major: OperandMajorMode,
        b_major: OperandMajorMode,
    ) -> Result<Tcgen05MxInstrDesc, String> {
        let kind = UmmaDataKind::Mxf8f6f4;
        let (a_format, b_format) = (input_format(kind, a, "A")?, input_format(kind, b, "B")?);
        let scale_format = scale_format(kind, sf)?;
        check_shape(kind, m, n, cta_group)?;
        check_major(a, a_major, "A")?;
        check_major(b, b_major, "B")?;
        Tcgen05MxInstrDesc::build(&Tcgen05MxInstrDescFields {
            a_format,
            b_format,
            scale_format,
            a_major,
            b_major,
            n,
            m,
            ..Default::default()
        })
    }

    pub fn bits(&self) -> Bits<u32> {
        self.bits
    }

    #[track_caller]
    pub fn value(&self) -> Value<u32> {
        self.bits.value()
    }

    /// The descriptor with A's and B's scale-factor IDs replaced (masked to
    /// their 2-bit fields), constants or kernel values.
    #[track_caller]
    pub fn set_sf_ids(&self, a_sf_id: impl Into<Bits<u32>>, b_sf_id: impl Into<Bits<u32>>) -> Tcgen05MxInstrDesc {
        Tcgen05MxInstrDesc { bits: mx_set_sf_ids(self.bits, a_sf_id.into(), b_sf_id.into()) }
    }

    pub fn sparse_flag(&self) -> Bits<u32> {
        self.bits.field(2, 1)
    }
    pub fn b_sf_id(&self) -> Bits<u32> {
        self.bits.field(4, 2)
    }
    pub fn a_format(&self) -> Bits<u32> {
        self.bits.field(7, 3)
    }
    pub fn b_format(&self) -> Bits<u32> {
        self.bits.field(10, 3)
    }
    pub fn a_negate(&self) -> Bits<u32> {
        self.bits.field(13, 1)
    }
    pub fn b_negate(&self) -> Bits<u32> {
        self.bits.field(14, 1)
    }
    pub fn a_major(&self) -> Bits<u32> {
        self.bits.field(15, 1)
    }
    pub fn b_major(&self) -> Bits<u32> {
        self.bits.field(16, 1)
    }
    /// N >> 3.
    pub fn n_dim(&self) -> Bits<u32> {
        self.bits.field(17, 6)
    }
    pub fn scale_format(&self) -> Bits<u32> {
        self.bits.field(23, 2)
    }
    pub fn a_sf_layout(&self) -> Bits<u32> {
        self.bits.field(26, 1)
    }
    /// M >> 7.
    pub fn m_dim(&self) -> Bits<u32> {
        self.bits.field(27, 2)
    }
    pub fn a_sf_id(&self) -> Bits<u32> {
        self.bits.field(29, 2)
    }
    pub fn k_dim(&self) -> Bits<u32> {
        self.bits.field(31, 1)
    }
}

/// The fields of a [`Tcgen05MxOmmaInstrDesc`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tcgen05MxOmmaInstrDescFields {
    pub sparse: bool,
    pub a_sf_id: u8,
    pub b_sf_id: u8,
    pub a_sf_layout: bool,
    /// The A and B formats: E2M1 ([`Tcgen05InputFormat::MX_E2M1`], 1).
    pub a_format: Tcgen05InputFormat,
    pub b_format: Tcgen05InputFormat,
    pub scale_format: MxScaleFormat,
    pub a_negate: bool,
    pub b_negate: bool,
    pub a_major: OperandMajorMode,
    pub b_major: OperandMajorMode,
    pub n: u32,
    pub m: u32,
    /// The 2-bit K size, split across bits 3 (high) and 31 (low): 0 for K =
    /// 64, 1 for sm_103's K = 96 (and sm_107's `.kind::mxf4` K = 128), 2 for
    /// sm_107's `.kind::mxf4nvf4` K = 128.
    pub k_dim: u8,
    /// The sparsity-version bit (12). CuTe DSL sets it by default and calls
    /// it irrelevant on SM100; CuTe C++ leaves it 0, as cute-rs does.
    pub sparsity_version: bool,
}

impl Default for Tcgen05MxOmmaInstrDescFields {
    fn default() -> Self {
        Tcgen05MxOmmaInstrDescFields {
            sparse: false,
            a_sf_id: 0,
            b_sf_id: 0,
            a_sf_layout: false,
            a_format: Tcgen05InputFormat::MX_E2M1,
            b_format: Tcgen05InputFormat::MX_E2M1,
            scale_format: MxScaleFormat::E8M0,
            a_negate: false,
            b_negate: false,
            a_major: OperandMajorMode::K,
            b_major: OperandMajorMode::K,
            n: 0,
            m: 0,
            k_dim: 0,
            sparsity_version: false,
        }
    }
}

/// The instruction descriptor of block-scaled `tcgen05.mma.kind::mxf4` and
/// `.kind::mxf4nvf4` (CuTe DSL's `Tcgen05MxOmmaInstrDesc`): the layout of
/// [`Tcgen05MxInstrDesc`] with the K size split across bits 3 and 31, a
/// 2-bit B format (10-11) and the sparsity-version bit 12.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05MxOmmaInstrDesc {
    bits: Bits<u32>,
}

impl Tcgen05MxOmmaInstrDesc {
    pub fn from_bits(bits: impl Into<Bits<u32>>) -> Tcgen05MxOmmaInstrDesc {
        Tcgen05MxOmmaInstrDesc { bits: bits.into() }
    }

    /// Pack `f`, checking that each field fits, both formats are E2M1, the
    /// K size is 0 to 2, N is a multiple of 8 from 8 to 256 and M is 128 or
    /// 256.
    pub fn build(f: &Tcgen05MxOmmaInstrDescFields) -> Result<Tcgen05MxOmmaInstrDesc, String> {
        for (which, t) in [("A", f.a_format), ("B", f.b_format)] {
            check(t == Tcgen05InputFormat::MX_E2M1, || {
                format!("the {which} format {} is not E2M1 (1), the only one .kind::mxf4 and mxf4nvf4 take", t.0)
            })?;
        }
        check(f.k_dim < 3, || format!("the K size {} is not 0, 1 or 2", f.k_dim))?;
        let bits = mx_bits(
            f.sparse,
            f.a_negate,
            f.b_negate,
            f.a_major,
            f.b_major,
            (f.m, f.n),
            f.scale_format,
            f.a_sf_layout,
            (f.a_sf_id, f.b_sf_id),
        )? | u32::from(f.k_dim >> 1) << 3
            | u32::from(f.a_format.0) << 7
            | u32::from(f.b_format.0) << 10
            | (f.sparsity_version as u32) << 12
            | u32::from(f.k_dim & 1) << 31;
        Ok(Tcgen05MxOmmaInstrDesc { bits: Bits::Static(bits) })
    }

    /// The descriptor of `.kind::mxf4` or `.kind::mxf4nvf4` (`kind`) with
    /// `sf` scale factors, `m x n` (a CTA pair's M for `CtaGroup::TWO`),
    /// K-major operands, checking the scale-factor type and the shapes of the
    /// CTA group.
    pub fn for_mma(
        kind: UmmaDataKind,
        sf: DType,
        (m, n): (u32, u32),
        cta_group: CtaGroup,
    ) -> Result<Tcgen05MxOmmaInstrDesc, String> {
        check(matches!(kind, UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4), || {
            format!("Tcgen05MxOmmaInstrDesc is for .kind::mxf4 and mxf4nvf4, not .kind::{}", kind.ptx())
        })?;
        let scale_format = scale_format(kind, sf)?;
        check_shape(kind, m, n, cta_group)?;
        Tcgen05MxOmmaInstrDesc::build(&Tcgen05MxOmmaInstrDescFields { scale_format, n, m, ..Default::default() })
    }

    pub fn bits(&self) -> Bits<u32> {
        self.bits
    }

    #[track_caller]
    pub fn value(&self) -> Value<u32> {
        self.bits.value()
    }

    #[track_caller]
    pub fn set_sf_ids(&self, a_sf_id: impl Into<Bits<u32>>, b_sf_id: impl Into<Bits<u32>>) -> Tcgen05MxOmmaInstrDesc {
        Tcgen05MxOmmaInstrDesc { bits: mx_set_sf_ids(self.bits, a_sf_id.into(), b_sf_id.into()) }
    }

    pub fn sparse_flag(&self) -> Bits<u32> {
        self.bits.field(2, 1)
    }
    pub fn k_dim_upper(&self) -> Bits<u32> {
        self.bits.field(3, 1)
    }
    pub fn b_sf_id(&self) -> Bits<u32> {
        self.bits.field(4, 2)
    }
    pub fn a_format(&self) -> Bits<u32> {
        self.bits.field(7, 3)
    }
    pub fn b_format(&self) -> Bits<u32> {
        self.bits.field(10, 2)
    }
    pub fn sparsity_version(&self) -> Bits<u32> {
        self.bits.field(12, 1)
    }
    pub fn a_negate(&self) -> Bits<u32> {
        self.bits.field(13, 1)
    }
    pub fn b_negate(&self) -> Bits<u32> {
        self.bits.field(14, 1)
    }
    pub fn a_major(&self) -> Bits<u32> {
        self.bits.field(15, 1)
    }
    pub fn b_major(&self) -> Bits<u32> {
        self.bits.field(16, 1)
    }
    pub fn n_dim(&self) -> Bits<u32> {
        self.bits.field(17, 6)
    }
    pub fn scale_format(&self) -> Bits<u32> {
        self.bits.field(23, 2)
    }
    pub fn a_sf_layout(&self) -> Bits<u32> {
        self.bits.field(26, 1)
    }
    pub fn m_dim(&self) -> Bits<u32> {
        self.bits.field(27, 2)
    }
    pub fn a_sf_id(&self) -> Bits<u32> {
        self.bits.field(29, 2)
    }
    pub fn k_dim_lower(&self) -> Bits<u32> {
        self.bits.field(31, 1)
    }
    /// The 2-bit K size (`upper << 1 | lower`), for a constant descriptor.
    pub fn k_dim(&self) -> Option<u32> {
        Some(self.k_dim_upper().as_static()? << 1 | self.k_dim_lower().as_static()?)
    }
}

// ---------------------------------------------------------------------------------------
// Tensor-memory addresses.
// ---------------------------------------------------------------------------------------

/// A tensor-memory address (CuTe DSL's `TmemAddr`): an Int32 with the lane
/// (row) in bits 16-31 and the column in bits 0-15.
#[derive(Clone, Copy, Debug)]
pub struct TmemAddr {
    bits: Bits<i32>,
}

impl TmemAddr {
    pub fn from_bits(bits: impl Into<Bits<i32>>) -> TmemAddr {
        TmemAddr { bits: bits.into() }
    }

    /// `row << 16 | col & 0xFFFF`. A constant row must be 0 to 0x7FFF and a
    /// constant column 0 to 0xFFFF.
    #[track_caller]
    pub fn from_row_col(row: impl Into<Bits<i32>>, col: impl Into<Bits<i32>>) -> Result<TmemAddr, String> {
        let (row, col) = (row.into(), col.into());
        if let Some(r) = row.as_static() {
            check((0..0x8000).contains(&r), || format!("the TMEM lane {r} does not fit bits 16-30"))?;
        }
        if let Some(c) = col.as_static() {
            check((0..0x10000).contains(&c), || format!("the TMEM column {c} does not fit bits 0-15"))?;
        }
        let bits = match (row, col) {
            (Bits::Static(r), Bits::Static(c)) => Bits::Static(r << 16 | c),
            (r, c) => Bits::Runtime((r.value() << 16i32) | (c.value() & 0xFFFFi32)),
        };
        Ok(TmemAddr { bits })
    }

    pub fn bits(&self) -> Bits<i32> {
        self.bits
    }

    #[track_caller]
    pub fn value(&self) -> Value<i32> {
        self.bits.value()
    }

    /// The column (bits 0-15).
    pub fn col_id(&self) -> Bits<i32> {
        self.bits.field(0, 16)
    }

    /// The lane (bits 16-31).
    pub fn row_id(&self) -> Bits<i32> {
        self.bits.field(16, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Layout;
    use crate::nvgpu::OperandMajorMode::{K, MN};

    #[test]
    fn wgmma_fields_follow_the_ptx_table() {
        let d = WgmmaSmemDesc::build(0x1230u32, 0x40, 0x400, 3, WgmmaSwizzle::Swizzle64B).unwrap();
        let want = 0x123 | 4 << 16 | 0x40 << 32 | 3 << 49 | 2u64 << 62;
        assert_eq!(d.bits().as_static(), Some(want));
        assert_eq!(d.start_address().as_static(), Some(0x123));
        assert_eq!(d.leading_byte_offset().as_static(), Some(4));
        assert_eq!(d.stride_byte_offset().as_static(), Some(0x40));
        assert_eq!(d.base_offset().as_static(), Some(3));
        assert_eq!(d.swizzle(), Some(WgmmaSwizzle::Swizzle64B));
        // The encoding drops address bits past 18.
        assert!(WgmmaSmemDesc::build(0x4_0000u32, 16, 16, 0, WgmmaSwizzle::Interleave).is_err());
        assert!(WgmmaSmemDesc::build(0x18u32, 16, 16, 0, WgmmaSwizzle::Interleave).is_err());
        assert!(WgmmaSmemDesc::build(0u32, 8, 16, 0, WgmmaSwizzle::Interleave).is_err());
        assert!(WgmmaSmemDesc::build(0u32, 16, 1 << 18, 0, WgmmaSwizzle::Interleave).is_err());
        assert!(WgmmaSmemDesc::build(0u32, 16, 16, 1, WgmmaSwizzle::Interleave).is_err());
        assert!(WgmmaSmemDesc::build(0u32, 16, 16, 8, WgmmaSwizzle::Swizzle128B).is_err());
        // Advancing: by 32 bytes, confined to the field, and refused off the 16-byte grid.
        let a = d.advance_start_address(32u64).unwrap();
        assert_eq!(a.bits().as_static(), Some(want + 2));
        let top = WgmmaSmemDesc::build(0x3fff0u32, 16, 16, 0, WgmmaSwizzle::Swizzle128B).unwrap();
        assert_eq!(top.advance_start_address(16u64).unwrap().start_address().as_static(), Some(0));
        assert_eq!(top.advance_start_address(16u64).unwrap().leading_byte_offset().as_static(), Some(1));
        assert!(d.advance_start_address(8u64).is_err());
    }

    #[test]
    fn tcgen05_smem_fields_follow_the_ptx_table() {
        let f = Tcgen05SmemDescFields {
            leading_byte_offset: 0x7fff0,
            stride_byte_offset: 0x3fff0,
            base_offset: 5,
            absolute_lbo: true,
            k_segment_offset: true,
            swizzle: Tcgen05SmemSwizzle::Swizzle32B,
        };
        let d = Tcgen05SmemDesc::build(0x7fff0u32, &f).unwrap();
        let want = 0x7fff | 0x7fff << 16 | 0x3fff << 32 | 1 << 46 | 5 << 49 | 1 << 52 | 1 << 53 | 6u64 << 61;
        assert_eq!(d.bits().as_static(), Some(want));
        assert_eq!(d.start_address().as_static(), Some(0x7fff));
        assert_eq!(d.leading_byte_offset().as_static(), Some(0x7fff));
        assert_eq!(d.stride_byte_offset().as_static(), Some(0x3fff));
        assert_eq!(d.version().as_static(), Some(1));
        assert_eq!(d.base_offset().as_static(), Some(5));
        assert_eq!(d.leading_dim_mode().as_static(), Some(1));
        assert_eq!(d.k_segment_offset().as_static(), Some(1));
        assert_eq!(d.swizzle(), Some(Tcgen05SmemSwizzle::Swizzle32B));
        for (s, code) in [
            (Tcgen05SmemSwizzle::SwizzleNone, 0),
            (Tcgen05SmemSwizzle::Swizzle128BBase32B, 1),
            (Tcgen05SmemSwizzle::Swizzle128B, 2),
            (Tcgen05SmemSwizzle::Swizzle64B, 4),
        ] {
            let d = Tcgen05SmemDesc::build(0u32, &Tcgen05SmemDescFields { swizzle: s, ..f }).unwrap();
            assert_eq!(d.layout_type().as_static(), Some(code));
        }
        for code in [3, 5, 7] {
            assert_eq!(Tcgen05SmemDesc::from_bits(code << 61).swizzle(), None);
        }
        assert!(Tcgen05SmemDesc::build(0x8_0000u32, &f).is_err());
        assert!(Tcgen05SmemDesc::build(0u32, &Tcgen05SmemDescFields { stride_byte_offset: 0x4_0000, ..f }).is_err());
        assert!(Tcgen05SmemDesc::build(0u32, &Tcgen05SmemDescFields { leading_byte_offset: 24, ..f }).is_err());
        let d = Tcgen05SmemDesc::build(0x100u32, &Tcgen05SmemDescFields::default()).unwrap();
        assert_eq!(d.advance_start_address(64u64).unwrap().start_address().as_static(), Some(0x14));
        assert!(d.advance_start_address(4u64).is_err());
        let top = Tcgen05SmemDesc::build(0x7fff0u32, &Tcgen05SmemDescFields::default()).unwrap();
        assert!(top.advance_start_address(16u64).is_err());
    }

    #[test]
    fn instruction_descriptor_fields() {
        let f = Tcgen05InstrDescFields {
            sparse_id2: 3,
            sparse: true,
            saturate: true,
            c_format: Tcgen05OutputFormat::S32,
            a_format: Tcgen05InputFormat::S8,
            b_format: Tcgen05InputFormat(7),
            a_negate: true,
            b_negate: true,
            a_major: MN,
            b_major: MN,
            n: 504,
            m: 496,
            k_dim: true,
            max_shift: 3,
        };
        let d = Tcgen05InstrDesc::build(&f).unwrap();
        let want = 3 | 1 << 2 | 1 << 3 | 2 << 4 | 1 << 7 | 7 << 10 | 1 << 13 | 1 << 14 | 1 << 15 | 1 << 16;
        let want = want | 63 << 17 | 31 << 24 | 1 << 29 | 3 << 30;
        assert_eq!(d.bits().as_static(), Some(want));
        let got = [
            d.sparse_id2(),
            d.sparse_flag(),
            d.saturate(),
            d.c_format(),
            d.a_format(),
            d.b_format(),
            d.a_negate(),
            d.b_negate(),
            d.a_major(),
            d.b_major(),
            d.n_dim(),
            d.m_dim(),
            d.k_dim(),
            d.max_shift(),
        ]
        .map(|b| b.as_static().unwrap());
        assert_eq!(got, [3, 1, 1, 2, 1, 7, 1, 1, 1, 1, 63, 31, 1, 3]);
        assert!(Tcgen05InstrDesc::build(&Tcgen05InstrDescFields { n: 12, ..f }).is_err());
        assert!(Tcgen05InstrDesc::build(&Tcgen05InstrDescFields { m: 72, ..f }).is_err());
        assert!(Tcgen05InstrDesc::build(&Tcgen05InstrDescFields { m: 512, ..f }).is_err());
        assert!(Tcgen05InstrDesc::build(&Tcgen05InstrDescFields { sparse_id2: 4, ..f }).is_err());
        assert!(Tcgen05InstrDesc::build(&Tcgen05InstrDescFields { a_format: Tcgen05InputFormat(8), ..f }).is_err());

        // Shapes and types per kind and CTA group.
        let one = CtaGroup::ONE;
        let two = CtaGroup::TWO;
        let mma = |k, a, b, c, mn, g, am| Tcgen05InstrDesc::for_mma(k, a, b, c, mn, g, am, K);
        assert!(mma(UmmaDataKind::F16, DType::F16, DType::F16, DType::F16, (64, 8), one, MN).is_ok());
        assert!(mma(UmmaDataKind::F16, DType::BF16, DType::BF16, DType::F16, (64, 8), one, K).is_err());
        assert!(mma(UmmaDataKind::F16, DType::F16, DType::F16, DType::F32, (256, 8), one, K).is_err());
        assert!(mma(UmmaDataKind::F16, DType::F16, DType::F16, DType::F32, (256, 16), two, K).is_ok());
        assert!(mma(UmmaDataKind::F16, DType::F16, DType::F16, DType::F32, (128, 24), two, K).is_err());
        assert!(mma(UmmaDataKind::F16, DType::F16, DType::F16, DType::F32, (64, 264), one, K).is_err());
        assert!(mma(UmmaDataKind::I8, DType::I8, DType::U8, DType::I32, (128, 24), one, K).is_err());
        assert!(mma(UmmaDataKind::I8, DType::I8, DType::U8, DType::I32, (128, 8), one, K).is_ok());
        assert!(mma(UmmaDataKind::F8F6F4, DType::F4E2M1FN, DType::F8E4M3FN, DType::F32, (128, 8), one, MN).is_err());
        assert!(mma(UmmaDataKind::Tf32, DType::TF32, DType::TF32, DType::F16, (128, 8), one, K).is_err());
        assert!(mma(UmmaDataKind::Mxf8f6f4, DType::F8E4M3FN, DType::F8E4M3FN, DType::F32, (128, 8), one, K).is_err());
    }

    #[test]
    fn block_scaled_descriptor_fields() {
        let f = Tcgen05MxInstrDescFields {
            sparse: true,
            b_sf_id: 2,
            a_sf_id: 3,
            a_format: Tcgen05InputFormat::E2M1,
            b_format: Tcgen05InputFormat::E3M2,
            scale_format: MxScaleFormat::UE5M3,
            a_negate: true,
            b_negate: false,
            a_major: K,
            b_major: MN,
            n: 256,
            m: 256,
            k_dim: true,
            a_sf_layout: true,
        };
        let d = Tcgen05MxInstrDesc::build(&f).unwrap();
        let want = 1 << 2 | 2 << 4 | 5 << 7 | 4 << 10 | 1 << 13 | 1 << 16 | 32 << 17 | 2 << 23 | 1 << 26 | 2 << 27;
        assert_eq!(d.bits().as_static(), Some(want | 3 << 29 | 1 << 31));
        assert_eq!(d.m_dim().as_static(), Some(2));
        assert_eq!(d.scale_format().as_static(), Some(2));
        let d = d.set_sf_ids(1u32, 0u32);
        assert_eq!((d.a_sf_id().as_static(), d.b_sf_id().as_static()), (Some(1), Some(0)));
        assert!(Tcgen05MxInstrDesc::build(&Tcgen05MxInstrDescFields { m: 64, ..f }).is_err());
        assert!(Tcgen05MxInstrDesc::build(&Tcgen05MxInstrDescFields { n: 264, ..f }).is_err());
        assert!(Tcgen05MxInstrDesc::build(&Tcgen05MxInstrDescFields { a_format: Tcgen05InputFormat(2), ..f }).is_err());
        assert!(
            Tcgen05MxInstrDesc::for_mma(
                DType::F8E4M3FN,
                DType::F8E5M2,
                DType::F8E8M0FNU,
                (128, 8),
                CtaGroup::ONE,
                K,
                K
            )
            .is_ok()
        );
        assert!(
            Tcgen05MxInstrDesc::for_mma(DType::F8E4M3FN, DType::F8E5M2, DType::F8E4M3FN, (128, 8), CtaGroup::ONE, K, K)
                .is_err()
        );
        assert!(
            Tcgen05MxInstrDesc::for_mma(
                DType::F8E4M3FN,
                DType::F8E5M2,
                DType::F8E8M0FNU,
                (128, 8),
                CtaGroup::TWO,
                K,
                K
            )
            .is_err()
        );

        let o = Tcgen05MxOmmaInstrDescFields { k_dim: 2, sparsity_version: true, n: 64, m: 128, ..Default::default() };
        let d = Tcgen05MxOmmaInstrDesc::build(&o).unwrap();
        assert_eq!(d.bits().as_static(), Some(1 << 3 | 1 << 7 | 1 << 10 | 1 << 12 | 8 << 17 | 1 << 23 | 1 << 27));
        assert_eq!(d.k_dim(), Some(2));
        let d = Tcgen05MxOmmaInstrDesc::build(&Tcgen05MxOmmaInstrDescFields { k_dim: 1, ..o }).unwrap();
        assert_eq!((d.k_dim_upper().as_static(), d.k_dim_lower().as_static()), (Some(0), Some(1)));
        assert!(Tcgen05MxOmmaInstrDesc::build(&Tcgen05MxOmmaInstrDescFields { k_dim: 3, ..o }).is_err());
        assert!(
            Tcgen05MxOmmaInstrDesc::build(&Tcgen05MxOmmaInstrDescFields { b_format: Tcgen05InputFormat(0), ..o })
                .is_err()
        );
        assert!(Tcgen05MxOmmaInstrDesc::for_mma(UmmaDataKind::Mxf4, DType::F8E4M3FN, (128, 8), CtaGroup::ONE).is_err());
        assert!(
            Tcgen05MxOmmaInstrDesc::for_mma(UmmaDataKind::Mxf4Nvf4, DType::F8E4M3FN, (256, 16), CtaGroup::TWO).is_ok()
        );
    }

    #[test]
    fn tmem_addresses() {
        let a = TmemAddr::from_row_col(32, 0x1ff).unwrap();
        assert_eq!(a.bits().as_static(), Some(32 << 16 | 0x1ff));
        assert_eq!((a.row_id().as_static(), a.col_id().as_static()), (Some(32), Some(0x1ff)));
        assert!(TmemAddr::from_row_col(0, 0x10000).is_err());
        assert!(TmemAddr::from_row_col(-1, 0).is_err());
    }

    /// The WGMMA atoms' descriptor fields are the builder's (the same bits
    /// for the same configuration).
    #[test]
    fn wgmma_atom_fields_match_the_builder() {
        use crate::nvgpu::warpgroup::gmma_desc_fields;
        // K-major, 16-bit, 128-byte swizzle: rows of 64 elements, SBO 8 rows = 1024 bytes.
        let tile = Layout::new(((8, 8), (8, 2)), ((64, 512), (1, 8)));
        let got = gmma_desc_fields(&tile, 64, K, 3, 16).unwrap();
        let want = WgmmaSmemDesc::build(0u32, 16, 1024, 0, WgmmaSwizzle::Swizzle128B).unwrap();
        assert_eq!(Some(got), want.bits().as_static());
        // K-major, interleaved: LBO 128 bytes (8 rows of 16 bytes), SBO 256.
        let tile = Layout::new(((8, 8), (8, 2)), ((8, 128), (1, 64)));
        let got = gmma_desc_fields(&tile, 64, K, 0, 16).unwrap();
        let want = WgmmaSmemDesc::build(0u32, 128, 256, 0, WgmmaSwizzle::Interleave).unwrap();
        assert_eq!(Some(got), want.bits().as_static());
    }

    /// The UMMA atoms' descriptor fields are the builder's.
    #[test]
    fn umma_atom_fields_match_the_builder() {
        use crate::nvgpu::tcgen05::{UmmaDescEncoding, UmmaLayoutType, umma_desc_fields};
        // K-major, 16-bit, 128-byte swizzle, 128 rows: SBO 1024 bytes, LBO written as 16.
        let tile = Layout::new(((8, 16), (8, 2)), ((64, 512), (1, 8)));
        let f = umma_desc_fields(&tile, K, UmmaLayoutType::Swizzle128B, 16, UmmaDescEncoding::Sm100).unwrap();
        let fields = Tcgen05SmemDescFields {
            leading_byte_offset: 16,
            stride_byte_offset: 1024,
            swizzle: Tcgen05SmemSwizzle::Swizzle128B,
            ..Default::default()
        };
        let want = Tcgen05SmemDesc::build(0u32, &fields).unwrap();
        assert_eq!(Some(f.encode(UmmaDescEncoding::Sm100)), want.bits().as_static());
    }

    /// The UMMA atoms' instruction descriptors are the builders'.
    #[test]
    fn umma_atom_instruction_descriptors_match_the_builders() {
        use crate::nvgpu::tcgen05::{MmaF16BF16Op, MmaMXF4NVF4Op, MmaMXF8F6F4Op, OperandSource};
        let s = OperandSource::SMEM;
        let k = MmaF16BF16Op::new(DType::BF16, DType::F32, (128, 64), CtaGroup::ONE, s, MN, K).kind();
        let d = Tcgen05InstrDesc::for_mma(
            UmmaDataKind::F16,
            DType::BF16,
            DType::BF16,
            DType::F32,
            (128, 64),
            CtaGroup::ONE,
            MN,
            K,
        )
        .unwrap();
        assert_eq!(Some(k.idesc()), d.bits().as_static());
        let k = MmaMXF8F6F4Op::new(DType::F4E2M1FN, DType::F8E4M3FN, (128, 256), CtaGroup::ONE, s, K, K).kind();
        let d = Tcgen05MxInstrDesc::for_mma(
            DType::F4E2M1FN,
            DType::F8E4M3FN,
            DType::F8E8M0FNU,
            (128, 256),
            CtaGroup::ONE,
            K,
            K,
        )
        .unwrap();
        assert_eq!(Some(k.idesc()), d.bits().as_static());
        let k = MmaMXF4NVF4Op::new(DType::F8E4M3FN, (256, 128), CtaGroup::TWO, s).kind();
        let d = Tcgen05MxOmmaInstrDesc::for_mma(UmmaDataKind::Mxf4Nvf4, DType::F8E4M3FN, (256, 128), CtaGroup::TWO)
            .unwrap();
        assert_eq!(Some(k.idesc()), d.bits().as_static());
    }
}

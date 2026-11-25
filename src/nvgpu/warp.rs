// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's `cute.nvgpu.warp` block-scaled MMA operations: warp-level `mma.sync ... .block_scale` on
//! GeForce Blackwell (the sm_120a and sm_121a chips with PTX ISA 8.7, and
//! the sm_120f family with 8.8).
//!
//! | operation          | A / B                     | scale factors | shape     | vector |
//! |--------------------|---------------------------|---------------|-----------|--------|
//! | [`MmaMXF4Op`]      | E2M1 / E2M1               | UE8M0         | 16x8x64   | 32     |
//! | [`MmaMXF4NVF4Op`]  | E2M1 / E2M1               | UE4M3         | 16x8x64   | 16     |
//! | [`MmaMXF8Op`]      | E4M3 or E5M2 (both alike) | UE8M0         | 16x8x32   | 32     |
//! | [`MmaMXF8F6F4Op`]  | [`MXF8F6F4_SUPPORTED_PAIRS`] | UE8M0      | 16x8x32   | 32     |
//!
//! All accumulate in Float32. A is row-major and B column-major (both
//! K-major). The scale factors come with the operands, CuTe DSL's
//! `cute.gemm(atom, d, [a, sfa], [b, sfb], c)`: in cute-rs
//! `gemm(&mma, &d, (&a, &sfa), (&b, &sfb), &c)`, where `sfa` and `sfb` are
//! the fragments of [`partition_fragment_sfa`](crate::utils::sm120::partition_fragment_sfa)
//! and [`partition_fragment_sfb`](crate::utils::sm120::partition_fragment_sfb);
//! or set as the atom's [`Field::SFA`] and [`Field::SFB`] with
//! [`TiledMma::set_tensor`](crate::TiledMma::set_tensor).
//!
//! The FP4 operand of [`MmaMXF8F6F4Op`] is a fragment of `F4E2M1FN` values
//! like the others; [`gemm`](crate::gemm) puts each in the 8-bit container
//! the instruction reads (bits 5:2), where a CuTe DSL kernel shifts the
//! loaded bytes itself.
//!
//! Which targets may run them is checked when the kernel is compiled
//! ([`check_target`](crate::validate::check_target)), where CuTe DSL checks
//! the environment's architecture when the operation is made.

use crate::atom::mma::{MmaAtom, MmaOp};
use crate::ir::{BlockScaledMmaKind, DType};

/// CuTe DSL's `cute.nvgpu.warp.Field`: the runtime fields of the warp-level
/// MMA atoms ([`MmaField`](crate::MmaField); the block-scaled atoms take
/// `SFA` and `SFB`).
pub use crate::atom::fields::MmaField as Field;

/// The (A, B) types [`MmaMXF8F6F4Op`] accepts: FP4 with either FP8 format,
/// in either order.
pub const MXF8F6F4_SUPPORTED_PAIRS: [(DType, DType); 4] = [
    (DType::F4E2M1FN, DType::F8E4M3FN),
    (DType::F4E2M1FN, DType::F8E5M2),
    (DType::F8E4M3FN, DType::F4E2M1FN),
    (DType::F8E5M2, DType::F4E2M1FN),
];

/// The checks CuTe DSL's `MmaSM120BlockScaledOp` makes of the same-type
/// operations; panics naming the parameter.
#[track_caller]
fn check_same_type(op: &str, ab: DType, acc: DType, sf: DType, vec: usize) {
    let n = |t: DType| t.cute_name();
    let fp8 = matches!(ab, DType::F8E4M3FN | DType::F8E5M2);
    assert!(
        ab == DType::F4E2M1FN || fp8,
        "cute: {op}: 'ab_dtype' {} is not a block-scaled operand type (Float4E2M1FN, Float8E4M3FN or Float8E5M2)",
        n(ab)
    );
    assert!(acc == DType::F32, "cute: {op} accumulates in Float32 only, but 'acc_dtype' is {}", n(acc));
    let want = if vec == 16 { DType::F8E4M3FN } else { DType::F8E8M0FNU };
    assert!(
        sf == want,
        "cute: {op}: scale-factor vectors of {vec} elements take {} scale factors, but 'sf_type' is {}",
        n(want),
        n(sf)
    );
}

macro_rules! same_type_op {
    ($(#[$doc:meta])* $name:ident, $vec:expr, $fp4:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct $name {
            ab_dtype: DType,
            acc_dtype: DType,
            sf_type: DType,
        }

        impl $name {
            /// The operation, or a panic naming the unsupported parameter (as
            /// CuTe DSL raises at construction).
            #[track_caller]
            pub fn new(ab_dtype: DType, acc_dtype: DType, sf_type: DType) -> $name {
                let op = stringify!($name);
                check_same_type(op, ab_dtype, acc_dtype, sf_type, $vec);
                let fp4 = ab_dtype == DType::F4E2M1FN;
                assert!(
                    fp4 == $fp4,
                    "cute: {op} takes {} operands, but 'ab_dtype' is {}",
                    if $fp4 { "Float4E2M1FN" } else { "Float8E4M3FN or Float8E5M2" },
                    ab_dtype.cute_name()
                );
                $name { ab_dtype, acc_dtype, sf_type }
            }

            pub fn ab_dtype(&self) -> DType {
                self.ab_dtype
            }

            pub fn acc_dtype(&self) -> DType {
                self.acc_dtype
            }

            pub fn sf_type(&self) -> DType {
                self.sf_type
            }

            pub fn sf_vec_size(&self) -> usize {
                $vec
            }

            pub fn shape_mnk(&self) -> (usize, usize, usize) {
                self.kind().shape_mnk()
            }

            /// The instruction it issues.
            pub fn kind(&self) -> BlockScaledMmaKind {
                BlockScaledMmaKind { a: self.ab_dtype, b: self.ab_dtype, sf_vec_size: $vec }
            }
        }

        impl From<$name> for MmaOp {
            fn from(op: $name) -> MmaOp {
                MmaOp::Sm120(op.kind())
            }
        }

        impl From<$name> for MmaAtom {
            fn from(op: $name) -> MmaAtom {
                MmaAtom::new(op.into())
            }
        }
    };
}

same_type_op!(
    /// CuTe DSL's `cute.nvgpu.warp.MmaMXF4Op(ab_dtype, acc_dtype, sf_type)`:
    /// `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`,
    /// Float4E2M1FN A and B packed eight per register, a Float8E8M0FNU scale
    /// per 32 elements of K, Float32 accumulators.
    MmaMXF4Op,
    32,
    true
);

same_type_op!(
    /// CuTe DSL's `cute.nvgpu.warp.MmaMXF4NVF4Op(ab_dtype, acc_dtype,
    /// sf_type)` (NVFP4):
    /// `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`,
    /// a Float8E4M3FN scale (read as UE4M3) per 16 elements of K.
    MmaMXF4NVF4Op,
    16,
    true
);

same_type_op!(
    /// CuTe DSL's `cute.nvgpu.warp.MmaMXF8Op(ab_dtype, acc_dtype, sf_type)`:
    /// `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.<t>.<t>.f32.ue8m0`
    /// with Float8E4M3FN or Float8E5M2 A and B (the same type), the FP8
    /// fragments of [`MmaFP8Op`](crate::MmaFP8Op)'s m16n8k32, and a
    /// Float8E8M0FNU scale per 32 elements of K.
    MmaMXF8Op,
    32,
    false
);

/// CuTe DSL's `cute.nvgpu.warp.MmaMXF8F6F4Op(a_dtype, b_dtype, acc_dtype,
/// sf_type)`: the mixed FP4 x FP8 form of
/// `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X`,
/// with A and B of different types from [`MXF8F6F4_SUPPORTED_PAIRS`], a
/// Float8E8M0FNU scale per 32 elements of K and Float32 accumulators. Its
/// fragments are those of the m16n8k32 FP8 MMA; the FP4 operand's values
/// travel in 8-bit containers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaMXF8F6F4Op {
    a_dtype: DType,
    b_dtype: DType,
    acc_dtype: DType,
    sf_type: DType,
}

impl MmaMXF8F6F4Op {
    /// The operation, or a panic naming the unsupported parameter (same-type
    /// pairs belong to [`MmaMXF4Op`], [`MmaMXF4NVF4Op`] and [`MmaMXF8Op`]).
    #[track_caller]
    pub fn new(a_dtype: DType, b_dtype: DType, acc_dtype: DType, sf_type: DType) -> MmaMXF8F6F4Op {
        let (a, b) = (a_dtype.cute_name(), b_dtype.cute_name());
        assert!(
            acc_dtype == DType::F32,
            "cute: MmaMXF8F6F4Op accumulates in Float32 only, but 'acc_dtype' is {}",
            acc_dtype.cute_name()
        );
        assert!(
            sf_type == DType::F8E8M0FNU,
            "cute: MmaMXF8F6F4Op takes Float8E8M0FNU scale factors, but 'sf_type' is {}",
            sf_type.cute_name()
        );
        if a_dtype == b_dtype {
            let other = if a_dtype == DType::F4E2M1FN { "MmaMXF4Op or MmaMXF4NVF4Op" } else { "MmaMXF8Op" };
            panic!("cute: MmaMXF8F6F4Op mixes two operand types, but A and B are both {a}: use {other}");
        }
        assert!(
            MXF8F6F4_SUPPORTED_PAIRS.contains(&(a_dtype, b_dtype)),
            "cute: MmaMXF8F6F4Op has no form for A {a} and B {b}; it pairs Float4E2M1FN with \
             Float8E4M3FN or Float8E5M2, either way round (MXF8F6F4_SUPPORTED_PAIRS)"
        );
        MmaMXF8F6F4Op { a_dtype, b_dtype, acc_dtype, sf_type }
    }

    pub fn a_dtype(&self) -> DType {
        self.a_dtype
    }

    pub fn b_dtype(&self) -> DType {
        self.b_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        self.acc_dtype
    }

    pub fn sf_type(&self) -> DType {
        self.sf_type
    }

    pub fn sf_vec_size(&self) -> usize {
        32
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        (16, 8, 32)
    }

    /// The instruction it issues.
    pub fn kind(&self) -> BlockScaledMmaKind {
        BlockScaledMmaKind { a: self.a_dtype, b: self.b_dtype, sf_vec_size: 32 }
    }
}

impl From<MmaMXF8F6F4Op> for MmaOp {
    fn from(op: MmaMXF8F6F4Op) -> MmaOp {
        MmaOp::Sm120(op.kind())
    }
}

impl From<MmaMXF8F6F4Op> for MmaAtom {
    fn from(op: MmaMXF8F6F4Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

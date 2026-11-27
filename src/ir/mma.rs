//! The `mma.sync` kinds, dense and block-scaled.

use super::*;

/// Warp-level matrix multiply-accumulate instructions (`mma.sync`, and the
/// sparse `mma.sp::ordered_metadata.sync`). The name gives the shape, then
/// the accumulator type and the operand type; `Sp` kinds are sparse.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmaKind {
    /// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    M16N8K16F32F16,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
    M16N8K8F32F16,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
    M16N8K16F16F16,
    /// `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`
    M16N8K8F16F16,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    M16N8K16F32BF16,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32`
    M16N8K8F32BF16,
    /// `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`: f32 operands
    /// truncated to 19 bits.
    M16N8K4Tf32,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: f32 operands
    /// truncated to 19 bits.
    M16N8K8Tf32,
    /// `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`
    M8N8K4F64,
    /// `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (sm_89, PTX 8.4)
    M16N8K32F32E4M3,
    /// `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` (sm_89, PTX 8.7)
    M16N8K32F16E4M3,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32` (sm_89, PTX 8.7)
    M16N8K16F32E4M3,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16` (sm_89, PTX 8.7)
    M16N8K16F16E4M3,
    /// `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` (sm_89, PTX 8.4)
    M16N8K32F32E5M2,
    /// `mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16` (sm_89, PTX 8.7)
    M16N8K32F16E5M2,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32` (sm_89, PTX 8.7)
    M16N8K16F32E5M2,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16` (sm_89, PTX 8.7)
    M16N8K16F16E5M2,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    SpM16N8K16F32F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
    SpM16N8K16F16F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    SpM16N8K16F32BF16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32`
    SpM16N8K32F32F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16`
    SpM16N8K32F16F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32`
    SpM16N8K32F32BF16,
}

impl MmaKind {
    pub fn shape_mnk(self) -> (usize, usize, usize) {
        use MmaKind::*;
        match self {
            M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16 => (16, 8, 16),
            M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16 | M16N8K8Tf32 => (16, 8, 8),
            M16N8K4Tf32 => (16, 8, 4),
            M8N8K4F64 => (8, 8, 4),
            M16N8K32F32E4M3 | M16N8K32F16E4M3 | M16N8K32F32E5M2 | M16N8K32F16E5M2 => (16, 8, 32),
            M16N8K16F32E4M3 | M16N8K16F16E4M3 | M16N8K16F32E5M2 | M16N8K16F16E5M2 => (16, 8, 16),
            SpM16N8K16F32F16 | SpM16N8K16F16F16 | SpM16N8K16F32BF16 => (16, 8, 16),
            SpM16N8K32F32F16 | SpM16N8K32F16F16 | SpM16N8K32F32BF16 => (16, 8, 32),
        }
    }

    /// Values held per thread for (A, B, C/D). A sparse kind's A fragment
    /// holds the kept half of A (two values of every four along K).
    pub fn fragment_sizes(self) -> (usize, usize, usize) {
        use MmaKind::*;
        match self {
            M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16 => (8, 4, 4),
            M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16 | M16N8K8Tf32 => (4, 2, 4),
            M16N8K4Tf32 => (2, 1, 4),
            M8N8K4F64 => (1, 1, 2),
            M16N8K32F32E4M3 | M16N8K32F16E4M3 | M16N8K32F32E5M2 | M16N8K32F16E5M2 => (16, 8, 4),
            M16N8K16F32E4M3 | M16N8K16F16E4M3 | M16N8K16F32E5M2 | M16N8K16F16E5M2 => (8, 4, 4),
            SpM16N8K16F32F16 | SpM16N8K16F16F16 | SpM16N8K16F32BF16 => (4, 4, 4),
            SpM16N8K32F32F16 | SpM16N8K32F16F16 | SpM16N8K32F32BF16 => (8, 8, 4),
        }
    }

    /// Element types of (A, B, C/D). TF32 operands are passed as `f32`.
    pub fn dtypes(self) -> (DType, DType, DType) {
        use MmaKind::*;
        let (ab, c) = match self {
            M16N8K16F32F16 | M16N8K8F32F16 | SpM16N8K16F32F16 | SpM16N8K32F32F16 => (DType::F16, DType::F32),
            M16N8K16F16F16 | M16N8K8F16F16 | SpM16N8K16F16F16 | SpM16N8K32F16F16 => (DType::F16, DType::F16),
            M16N8K16F32BF16 | M16N8K8F32BF16 | SpM16N8K16F32BF16 | SpM16N8K32F32BF16 => (DType::BF16, DType::F32),
            M16N8K4Tf32 | M16N8K8Tf32 => (DType::F32, DType::F32),
            M8N8K4F64 => (DType::F64, DType::F64),
            M16N8K32F32E4M3 | M16N8K16F32E4M3 => (DType::F8E4M3FN, DType::F32),
            M16N8K32F16E4M3 | M16N8K16F16E4M3 => (DType::F8E4M3FN, DType::F16),
            M16N8K32F32E5M2 | M16N8K16F32E5M2 => (DType::F8E5M2, DType::F32),
            M16N8K32F16E5M2 | M16N8K16F16E5M2 => (DType::F8E5M2, DType::F16),
        };
        (ab, ab, c)
    }

    /// Whether the operands are TF32 (f32 registers whose low 13 mantissa
    /// bits the tensor core ignores).
    pub fn is_tf32(self) -> bool {
        matches!(self, MmaKind::M16N8K4Tf32 | MmaKind::M16N8K8Tf32)
    }

    /// Whether the operands are FP8 (`.e4m3` or `.e5m2`, sm_89).
    pub fn is_fp8(self) -> bool {
        self.dtypes().0.is_fp8()
    }

    /// Whether this is a sparse `mma.sp` (A structured 2:4 sparse, with a
    /// metadata operand).
    pub fn is_sparse(self) -> bool {
        use MmaKind::*;
        matches!(
            self,
            SpM16N8K16F32F16
                | SpM16N8K16F16F16
                | SpM16N8K16F32BF16
                | SpM16N8K32F32F16
                | SpM16N8K32F16F16
                | SpM16N8K32F32BF16
        )
    }

    /// The PTX name of the operand type: `f16`, `tf32`, `e4m3`.
    fn ptx_operand(self) -> &'static str {
        match self.dtypes().0 {
            _ if self.is_tf32() => "tf32",
            DType::F8E4M3FN => "e4m3",
            DType::F8E5M2 => "e5m2",
            a => a.name(),
        }
    }

    /// The PTX instruction, e.g. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        let c = self.dtypes().2.name();
        let ab = self.ptx_operand();
        let op = if self.is_sparse() { "mma.sp::ordered_metadata.sync" } else { "mma.sync" };
        format!("{op}.aligned.m{m}n{n}k{k}.row.col.{c}.{ab}.{ab}.{c}")
    }
}

// SM120 warp-level block-scaled mma.sync kinds.

/// A warp-level block-scaled `mma.sync` (sm_120a, PTX ISA 8.7): its A and B
/// formats and its scale-factor vector size. With FP4 (`F4E2M1FN`) A and B
/// it is `m16n8k64` (`kind::mxf4` with UE8M0 scales per 32 elements, or
/// `kind::mxf4nvf4` with UE4M3 scales per 16), with A and B packed eight
/// values per register; otherwise `m16n8k32 kind::mxf8f6f4` with UE8M0
/// scales per 32 elements, four 8-bit containers per register (an FP4 value
/// in bits 5:2 of its container). Built by the `cute::warp` operations,
/// which check the combination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockScaledMmaKind {
    pub a: DType,
    pub b: DType,
    pub sf_vec_size: u8,
}

impl BlockScaledMmaKind {
    /// Whether A and B are both FP4, packed eight per register (`m16n8k64`).
    pub fn is_fp4_packed(self) -> bool {
        self.a == DType::F4E2M1FN && self.b == DType::F4E2M1FN
    }

    pub fn shape_mnk(self) -> (usize, usize, usize) {
        if self.is_fp4_packed() { (16, 8, 64) } else { (16, 8, 32) }
    }

    /// The scale-factor type: `F8E4M3FN` (read as UE4M3) for 16-element
    /// vectors, `F8E8M0FNU` (UE8M0) for 32.
    pub fn sf_dtype(self) -> DType {
        if self.sf_vec_size == 16 { DType::F8E4M3FN } else { DType::F8E8M0FNU }
    }

    /// Scale factors per row of A (and column of B) per instruction: 1, 2 or 4.
    pub fn scales_per_mma(self) -> usize {
        self.shape_mnk().2 / self.sf_vec_size as usize
    }

    /// `mxf4`, `mxf4nvf4` or `mxf8f6f4`.
    pub fn kind_name(self) -> &'static str {
        match (self.is_fp4_packed(), self.sf_vec_size) {
            (true, 16) => "mxf4nvf4",
            (true, _) => "mxf4",
            _ => "mxf8f6f4",
        }
    }

    /// The PTX name of an operand type: `e2m1`, `e4m3` or `e5m2`.
    pub fn ptx_type(t: DType) -> &'static str {
        match t {
            DType::F4E2M1FN => "e2m1",
            DType::F8E4M3FN => "e4m3",
            _ => "e5m2",
        }
    }

    /// `ue8m0` or `ue4m3`.
    pub fn ptx_sf_type(self) -> &'static str {
        if self.sf_vec_size == 16 { "ue4m3" } else { "ue8m0" }
    }

    /// Values per thread of (A, B, C/D).
    pub fn fragment_sizes(self) -> (usize, usize, usize) {
        if self.is_fp4_packed() { (32, 16, 4) } else { (16, 8, 4) }
    }

    /// 32-bit registers per thread of (A, B): 4 and 2.
    pub fn operand_registers(self) -> (usize, usize) {
        (4, 2)
    }

    /// Values per register of an operand: 8 packed FP4 values, or 4 8-bit
    /// containers.
    pub fn values_per_register(self) -> usize {
        if self.is_fp4_packed() { 8 } else { 4 }
    }

    /// The PTX instruction, e.g.
    /// `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        format!(
            "mma.sync.aligned.m{m}n{n}k{k}.row.col.kind::{}.block_scale.scale_vec::{}X.f32.{}.{}.f32.{}",
            self.kind_name(),
            self.scales_per_mma(),
            Self::ptx_type(self.a),
            Self::ptx_type(self.b),
            self.ptx_sf_type()
        )
    }
}

/// A warp-level sparse block-scaled `mma.sp::ordered_metadata` (sm_120a,
/// PTX ISA 8.7; CuTe C++'s `SM120_SPARSE_16x8x64_TN_VS` and
/// `SM120_SPARSE_16x8x128_TN_VS`): A 2:4 sparse (FP4: pairs, 4:8), stored
/// compressed. With FP4 A and B (`kind::mxf4` or `mxf4nvf4`) it is
/// `m16n8k128`, otherwise `m16n8k64 kind::mxf8f6f4` (8-bit containers).
/// `scale_vec` (1, 2 or 4) scale factors per row of A and column of B, of
/// type `sf` (`F8E8M0FNU`, or `F8E4M3FN` read as UE4M3), each from byte
/// `bid + j` of the register of the lane `tid` selects.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SpBlockScaledMmaKind {
    pub a: DType,
    pub b: DType,
    pub scale_vec: u8,
    pub sf: DType,
    pub bid_a: u8,
    pub tid_a: u8,
    pub bid_b: u8,
    pub tid_b: u8,
}

impl SpBlockScaledMmaKind {
    /// Whether A and B are both FP4, packed eight per register.
    pub fn is_fp4_packed(self) -> bool {
        self.a == DType::F4E2M1FN && self.b == DType::F4E2M1FN
    }

    /// `(16, 8, 128)` for packed FP4, `(16, 8, 64)` otherwise (logical K).
    pub fn shape_mnk(self) -> (usize, usize, usize) {
        if self.is_fp4_packed() { (16, 8, 128) } else { (16, 8, 64) }
    }

    /// Logical K elements per scale factor.
    pub fn vec(self) -> usize {
        self.shape_mnk().2 / self.scale_vec as usize
    }

    /// `mxf4`, `mxf4nvf4` or `mxf8f6f4`.
    pub fn kind_name(self) -> &'static str {
        match (self.is_fp4_packed(), self.scale_vec) {
            (true, 4) => "mxf4nvf4",
            (true, _) if self.sf == DType::F8E4M3FN => "mxf4nvf4",
            (true, _) => "mxf4",
            _ => "mxf8f6f4",
        }
    }

    /// `ue8m0` or `ue4m3`.
    pub fn ptx_sf_type(self) -> &'static str {
        if self.sf == DType::F8E4M3FN { "ue4m3" } else { "ue8m0" }
    }

    /// The PTX instruction.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        format!(
            "mma.sp::ordered_metadata.sync.aligned.m{m}n{n}k{k}.row.col.kind::{}.block_scale.scale_vec::{}X.f32.{}.{}.f32.{}",
            self.kind_name(),
            self.scale_vec,
            BlockScaledMmaKind::ptx_type(self.a),
            BlockScaledMmaKind::ptx_type(self.b),
            self.ptx_sf_type()
        )
    }

    /// Values per register: 8 packed FP4 values or 4 8-bit containers.
    pub fn values_per_register(self) -> usize {
        if self.is_fp4_packed() { 8 } else { 4 }
    }
}

//! Tensor-memory, `tcgen05` and `wgmma` instruction kinds.

use super::*;

/// The reduction of a `tcgen05.ld.red` (CuTe DSL's `tcgen05.TmemLoadRedOp`):
/// the minimum or maximum, of the values or of their absolute values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum TmemLoadRedOp {
    #[default]
    Max,
    Min,
    MaxAbs,
    MinAbs,
}

impl TmemLoadRedOp {
    /// The PTX qualifiers: `.max`, `.min`, `.max.abs` or `.min.abs`.
    pub fn ptx(self) -> &'static str {
        match self {
            TmemLoadRedOp::Max => ".max",
            TmemLoadRedOp::Min => ".min",
            TmemLoadRedOp::MaxAbs => ".max.abs",
            TmemLoadRedOp::MinAbs => ".min.abs",
        }
    }

    /// Whether this takes the minimum.
    pub fn is_min(self) -> bool {
        matches!(self, TmemLoadRedOp::Min | TmemLoadRedOp::MinAbs)
    }

    /// Whether this reduces absolute values.
    pub fn is_abs(self) -> bool {
        matches!(self, TmemLoadRedOp::MaxAbs | TmemLoadRedOp::MinAbs)
    }
}

/// The reduction of an [`Op::Tcgen05Ld`] (`tcgen05.ld.red`): the Float32
/// `result` is `op` over the values the thread loaded, as PTX's `min` and
/// `max` compute it (a NaN is ignored unless `nan`, which makes any NaN the
/// result). `half_split` is the immediate column offset of `.16x32bx2`'s
/// second access.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TmemLdRed {
    pub result: ValueId,
    pub op: TmemLoadRedOp,
    pub nan: bool,
    pub half_split: Option<u32>,
}

/// The 2:4 compression of an [`Op::Tcgen05Ld`] (`tcgen05.ld{.red}.spcompress`,
/// PTX ISA 9.4, sm_107a): of every four
/// consecutive Float32 values a thread loads, the two the row operation `op`
/// prefers (the larger for `max`, the smaller for `min`, by absolute value
/// for the `abs` forms; a NaN is the least preferred unless `nan`, the most;
/// the lower position wins a tie) are kept, in order. The op's `results` are
/// the kept values (half the loaded ones), `mdata` the metadata registers
/// (i32): the 4-bit field of group `g` (the position of the first kept value
/// in bits 0-1, of the second in bits 2-3, the sparse MMA's format) at bit
/// `4 (g % 8)` of register `g / 8`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TmemLdSpCompress {
    pub mdata: Vec<ValueId>,
    pub op: TmemLoadRedOp,
    pub nan: bool,
}

/// The shapes of `tcgen05.cp` (PTX ISA 9.7.18.9.2), with their broadcast (multicast) qualifiers: the
/// source is `rows` rows of `row_bits` bits in shared memory, written to
/// `broadcast()` TMEM lane groups.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemCpShape {
    /// `.128x256b`: 128 rows of 256 bits into 128 lanes.
    S128x256b,
    /// `.4x256b`: 4 rows of 256 bits into lanes 0, 32, 64 and 96.
    S4x256b,
    /// `.128x128b`: 128 rows of 128 bits into 128 lanes.
    S128x128b,
    /// `.64x128b.warpx2::02_13`: 64 rows of 128 bits, row `r` into lanes
    /// `r` and `64 + r`.
    S64x128bWarpx2_02_13,
    /// `.64x128b.warpx2::01_23`: 64 rows of 128 bits, rows 0 to 31 into
    /// lanes 0 to 63 (twice), rows 32 to 63 into lanes 64 to 127.
    S64x128bWarpx2_01_23,
    /// `.32x128b.warpx4`: 32 rows of 128 bits into each 32-lane quadrant.
    S32x128bWarpx4,
}

impl TmemCpShape {
    pub const ALL: [TmemCpShape; 6] = [
        TmemCpShape::S128x256b,
        TmemCpShape::S4x256b,
        TmemCpShape::S128x128b,
        TmemCpShape::S64x128bWarpx2_02_13,
        TmemCpShape::S64x128bWarpx2_01_23,
        TmemCpShape::S32x128bWarpx4,
    ];

    /// The PTX qualifiers, without the leading dot.
    pub fn ptx(self) -> &'static str {
        match self {
            TmemCpShape::S128x256b => "128x256b",
            TmemCpShape::S4x256b => "4x256b",
            TmemCpShape::S128x128b => "128x128b",
            TmemCpShape::S64x128bWarpx2_02_13 => "64x128b.warpx2::02_13",
            TmemCpShape::S64x128bWarpx2_01_23 => "64x128b.warpx2::01_23",
            TmemCpShape::S32x128bWarpx4 => "32x128b.warpx4",
        }
    }

    /// Rows of the shared-memory source.
    pub fn rows(self) -> usize {
        match self {
            TmemCpShape::S128x256b | TmemCpShape::S128x128b => 128,
            TmemCpShape::S4x256b => 4,
            TmemCpShape::S64x128bWarpx2_02_13 | TmemCpShape::S64x128bWarpx2_01_23 => 64,
            TmemCpShape::S32x128bWarpx4 => 32,
        }
    }

    /// Bits of each row: 256 or 128 (8 or 4 TMEM columns).
    pub fn row_bits(self) -> usize {
        match self {
            TmemCpShape::S128x256b | TmemCpShape::S4x256b => 256,
            _ => 128,
        }
    }

    /// TMEM lane groups each source row is written to: 4 for `.warpx4`, 2
    /// for `.warpx2`, else 1.
    pub fn broadcast(self) -> usize {
        match self {
            TmemCpShape::S32x128bWarpx4 => 4,
            TmemCpShape::S64x128bWarpx2_02_13 | TmemCpShape::S64x128bWarpx2_01_23 => 2,
            _ => 1,
        }
    }

    /// The TMEM lanes (after the address's) that source row `row` is
    /// written to.
    pub fn lanes_of_row(self, row: usize) -> Vec<usize> {
        match self {
            TmemCpShape::S128x256b | TmemCpShape::S128x128b => vec![row],
            TmemCpShape::S4x256b => vec![32 * row],
            TmemCpShape::S64x128bWarpx2_02_13 => vec![row, 64 + row],
            TmemCpShape::S64x128bWarpx2_01_23 => {
                let base = 64 * (row / 32) + row % 32;
                vec![base, base + 32]
            }
            TmemCpShape::S32x128bWarpx4 => (0..4).map(|q| 32 * q + row).collect(),
        }
    }
}

/// The data-movement shapes of `tcgen05.ld` and `tcgen05.st` (PTX ISA
/// 9.7.18.2.3).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemShape {
    /// `.16x64b`: 16 lanes of 64 bits per repetition.
    S16x64b,
    /// `.16x128b`: 16 lanes of 128 bits.
    S16x128b,
    /// `.16x256b`: 16 lanes of 256 bits.
    S16x256b,
    /// `.16x32bx2`: two accesses of 16 lanes of 32 bits, the second at the
    /// first's address plus an immediate column offset.
    S16x32bx2,
    /// `.32x32b`: 32 lanes of 32 bits.
    S32x32b,
}

impl TmemShape {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            TmemShape::S16x64b => "16x64b",
            TmemShape::S16x128b => "16x128b",
            TmemShape::S16x256b => "16x256b",
            TmemShape::S16x32bx2 => "16x32bx2",
            TmemShape::S32x32b => "32x32b",
        }
    }

    /// TMEM lanes one instruction accesses (16 or 32).
    pub fn lanes(self) -> u32 {
        if self == TmemShape::S32x32b { 32 } else { 16 }
    }

    /// 32-bit registers per thread per repetition (the PTX ISA's Table 59).
    pub fn registers_per_repetition(self) -> u32 {
        match self {
            TmemShape::S16x128b => 2,
            TmemShape::S16x256b => 4,
            _ => 1,
        }
    }

    /// 32-bit registers per thread of `.x<num>`.
    pub fn registers(self, num: u32) -> u32 {
        self.registers_per_repetition() * num
    }

    /// The largest repetition: at most 128 registers per thread.
    pub fn max_repetition(self) -> u32 {
        128 / self.registers_per_repetition()
    }

    /// The immediate column offset of the second access of `.16x32bx2`
    /// (the columns of the first, as CuTe's SM100_TMEM_LOAD_16dp32b
    /// instructions use), `None` for the other shapes.
    pub fn half_split(self, num: u32, pack: bool) -> Option<u32> {
        (self == TmemShape::S16x32bx2).then_some(num * if pack { 2 } else { 1 })
    }

    /// Where thread `lane` (0 to 31) of the warp finds register `reg` of an
    /// `.x<num>` access without packing: `(TMEM lane, TMEM column)` relative
    /// to the address (PTX ISA 9.7.18.2.3.1, matrix fragments, as CuTe's
    /// SM100 TMEM copy traits lay them out). With `.pack::16b`
    /// (`.unpack::16b`) the register's two halves are the low halves of
    /// columns `2 * column` and the next one: see [`TmemShape::cells`].
    pub fn fragment(self, lane: u32, reg: u32, num: u32) -> (u32, u32) {
        match self {
            TmemShape::S32x32b => (lane, reg),
            TmemShape::S16x64b => (lane / 4 + 8 * (lane % 2), (lane / 2) % 2 + 2 * reg),
            TmemShape::S16x128b => (lane / 4 + 8 * (reg % 2), lane % 4 + 4 * (reg / 2)),
            TmemShape::S16x256b => (lane / 4 + 8 * ((reg / 2) % 2), 2 * (lane % 4) + reg % 2 + 8 * (reg / 4)),
            TmemShape::S16x32bx2 => (lane % 16, num * (lane / 16) + reg),
        }
    }

    /// The TMEM cells (lane, column) relative to the address that hold
    /// register `reg` of thread `lane`: one 32-bit cell, or with packing
    /// the low halves of two adjacent cells (the register's low half
    /// first).
    pub fn cells(self, lane: u32, reg: u32, num: u32, packed: bool) -> Vec<(u32, u32)> {
        let (l, c) = self.fragment(lane, reg, num);
        if packed { vec![(l, 2 * c), (l, 2 * c + 1)] } else { vec![(l, c)] }
    }

    /// The TMEM columns an `.x<num>` access spans from its address.
    pub fn columns(self, num: u32, packed: bool) -> u32 {
        let base = match self {
            TmemShape::S32x32b => num,
            TmemShape::S16x64b | TmemShape::S16x32bx2 => 2 * num,
            TmemShape::S16x128b => 4 * num,
            TmemShape::S16x256b => 8 * num,
        };
        if packed { 2 * base } else { base }
    }
}

/// The instructions of [`Op::Tcgen05`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tcgen05Inst {
    /// `tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32 [dst],
    /// ncols`: `buffer` is the shared buffer of `dst`, `args` are `[index,
    /// ncols]`, `index` counting 32-bit words of the buffer. Warp-collective;
    /// it writes the TMEM address of the allocation to `dst`.
    Alloc,
    /// `tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32 taddr, ncols`:
    /// `args` are `[taddr, ncols]` (both i32). Warp-collective.
    Dealloc,
    /// `tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned`: no
    /// operands. Warp-collective.
    RelinquishAllocPermit,
    /// No instruction: the simulator checks that the TMEM address `args[0]`
    /// (i32), just loaded by `retrieve_tmem_ptr`, lies in a live allocation of
    /// the CTA. The backend emits nothing for it.
    CheckAddress,
    /// `tcgen05.wait::ld.sync.aligned`: the
    /// warp's earlier `tcgen05.ld`s have completed. Warp-collective.
    WaitLd,
    /// `tcgen05.wait::st.sync.aligned`: the warp's earlier `tcgen05.st`s
    /// have completed. Warp-collective.
    WaitSt,
    /// `tcgen05.fence::before_thread_sync`: orders the thread's earlier
    /// tcgen05 operations before a later thread synchronisation. Per thread.
    FenceBeforeThreadSync,
    /// `tcgen05.fence::after_thread_sync`: orders a thread synchronisation
    /// before the thread's later tcgen05 operations. Per thread.
    FenceAfterThreadSync,
}

impl Tcgen05Inst {
    /// The PTX instruction's name (without qualifiers).
    pub fn name(self) -> &'static str {
        match self {
            Tcgen05Inst::Alloc => "tcgen05.alloc",
            Tcgen05Inst::Dealloc => "tcgen05.dealloc",
            Tcgen05Inst::RelinquishAllocPermit => "tcgen05.relinquish_alloc_permit",
            Tcgen05Inst::CheckAddress => "retrieve_tmem_ptr",
            Tcgen05Inst::WaitLd => "tcgen05.wait::ld",
            Tcgen05Inst::WaitSt => "tcgen05.wait::st",
            Tcgen05Inst::FenceBeforeThreadSync => "tcgen05.fence::before_thread_sync",
            Tcgen05Inst::FenceAfterThreadSync => "tcgen05.fence::after_thread_sync",
        }
    }

    /// Whether every lane of the warp executes it together
    /// (`.sync.aligned`): all but the fences and the address check.
    pub fn is_warp_collective(self) -> bool {
        !matches!(
            self,
            Tcgen05Inst::CheckAddress | Tcgen05Inst::FenceBeforeThreadSync | Tcgen05Inst::FenceAfterThreadSync
        )
    }

    /// Whether it carries a `cta_group` (alloc, dealloc, relinquish).
    pub fn has_cta_group(self) -> bool {
        matches!(self, Tcgen05Inst::Alloc | Tcgen05Inst::Dealloc | Tcgen05Inst::RelinquishAllocPermit)
    }
}

/// The data kinds of `tcgen05.mma` (PTX ISA 9.7.18.10, `.kind`): F16, and TF32, I8 and F8F6F4
/// on the same machinery.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UmmaDataKind {
    /// `.kind::f16`: Float16 or BFloat16 operands, Float16 or Float32
    /// accumulators.
    F16,
    /// `.kind::tf32`: TFloat32 operands (32-bit in shared memory, of which
    /// the tensor core uses the top 19 bits), a Float32 accumulator, K = 8.
    Tf32,
    /// `.kind::i8`: Int8 or Uint8 operands, an Int32 accumulator, K = 32.
    I8,
    /// `.kind::f8f6f4`: E4M3, E5M2, E2M3, E3M2 or E2M1 operands (each of A
    /// and B), a Float16 or Float32 accumulator, K = 32. FP6 and FP4
    /// operands are read from the padded shared-memory image (16 values in
    /// each 16-byte slot, PTX's `.b6x16_p32` and `.b4x16_p64`).
    F8F6F4,
    /// `.kind::mxf8f6f4.block_scale`: the operands of `F8F6F4` (FP6 and FP4
    /// from the padded image), UE8M0 scale factors, one per 32 K elements,
    /// a Float32 accumulator, K = 32.
    Mxf8f6f4,
    /// `.kind::mxf4.block_scale`: E2M1 operands packed two per byte, UE8M0
    /// scale factors, one per 32 K elements, a Float32 accumulator, K = 64
    /// (96 on sm_103a).
    Mxf4,
    /// `.kind::mxf4nvf4.block_scale`: E2M1 operands packed two per byte,
    /// UE4M3 or UE8M0 scale factors, one per 16 K elements, a Float32
    /// accumulator, K = 64 (96 on sm_103a).
    Mxf4Nvf4,
}

impl UmmaDataKind {
    /// The PTX qualifier, without `.kind::`.
    pub fn ptx(self) -> &'static str {
        match self {
            UmmaDataKind::F16 => "f16",
            UmmaDataKind::Tf32 => "tf32",
            UmmaDataKind::I8 => "i8",
            UmmaDataKind::F8F6F4 => "f8f6f4",
            UmmaDataKind::Mxf8f6f4 => "mxf8f6f4",
            UmmaDataKind::Mxf4 => "mxf4",
            UmmaDataKind::Mxf4Nvf4 => "mxf4nvf4",
        }
    }

    /// Whether the kind is block-scaled (its
    /// instruction descriptor is CuTe C++'s `InstrDescriptorBlockScaled`).
    pub fn is_block_scaled(self) -> bool {
        matches!(self, UmmaDataKind::Mxf8f6f4 | UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4)
    }

    /// The instruction descriptor's 3-bit format of an A or B operand of
    /// type `t` (PTX ISA Table 51; CuTe C++'s `UMMA::to_UMMAFormat`).
    pub fn operand_format(self, t: DType) -> Option<u32> {
        match (self, t) {
            (UmmaDataKind::F16, DType::F16) => Some(0),
            (UmmaDataKind::F16, DType::BF16) => Some(1),
            (UmmaDataKind::Tf32, DType::TF32) => Some(2),
            (UmmaDataKind::I8, DType::U8) => Some(0),
            (UmmaDataKind::I8, DType::I8) => Some(1),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F8E4M3FN) => Some(0),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F8E5M2) => Some(1),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F6E2M3FN) => Some(3),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F6E3M2FN) => Some(4),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F4E2M1FN) => Some(5),
            // Packed E2M1 (CuTe C++'s MXF4Format::E2M1).
            (UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4, DType::F4E2M1FN) => Some(1),
            _ => None,
        }
    }

    /// K of one instruction.
    pub fn k(self) -> usize {
        match self {
            UmmaDataKind::F16 => 16,
            UmmaDataKind::Tf32 => 8,
            UmmaDataKind::I8 | UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4 => 32,
            UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4 => 64,
        }
    }

    /// The bits one element of an operand of type `t` takes in shared
    /// memory: its width, except FP6 and FP4 under `.kind::f8f6f4`, which
    /// are read from the padded image (16 values per 16 bytes), one byte
    /// per value as far as addresses go.
    pub fn smem_bits(self, t: DType) -> usize {
        if matches!(self, UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4) && t.bits() < 8 { 8 } else { t.bits() }
    }
}

/// One `tcgen05.mma` instruction: shape `m x n
/// x k` (K: 32 bytes of the operand type), A and B of types `a` and `b`
/// from shared memory, accumulated in `acc` in TMEM; A and B are MN-major
/// when `trans_a` and `trans_b` are set. `two_cta` is `.cta_group::2`.
/// With `a_tmem` A comes from tensor memory
/// (the `[a-tmem]` form, always K-major).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UmmaKind {
    pub kind: UmmaDataKind,
    pub m: u32,
    pub n: u32,
    pub a: DType,
    pub b: DType,
    pub acc: DType,
    pub trans_a: bool,
    pub trans_b: bool,
    pub two_cta: bool,
    pub a_tmem: bool,
    /// The scale factors of a block-scaled kind.
    pub scale: Option<UmmaScale>,
    /// `tcgen05.mma.sp`, A 2:4 structured sparse (stored
    /// compressed, half of K), with metadata in tensor memory.
    pub sparse: bool,
    /// The Rubin (sm_107) forms: the doubled K and the
    /// collector-buffer qualifiers.
    pub sm107: Option<Sm107Umma>,
    /// The forms and modifiers only `cute::prims`
    /// issues (weight-stationary `.ws`, `.collector::a`, `.ashift`,
    /// `scale-input-d`).
    pub prims: UmmaPrims,
}

/// The parts of a `tcgen05.mma` only `cute::prims` issues. The default is none of them.
///
/// * `thread_issue`: issued by each thread that executes it (a prims call
///   inside `elect_one`), not by a warp that elects a lane (`gemm`).
/// * `ws`: `tcgen05.mma.ws` (weight stationary, `cta_group::1`, M of 32,
///   64 or 128 in the data-path Layouts G, E and D), with B's collector
///   buffer `collector_b` (`.collector::b<i>::<op>`, `i` 0 to 3) and the
///   instruction descriptor's `max_shift` (bits 30-31).
/// * `collector_a`: `.collector::a::<op>` on a plain (sm_100a) MMA.
/// * `ashift`: `.ashift` (A in TMEM, shifted up one row in each 32-lane
///   sub-partition after the MMA).
/// * `scale_input_d`: `D = A B + D 2^-s` (`.kind::f16` and `tf32`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct UmmaPrims {
    pub thread_issue: bool,
    pub ws: bool,
    pub collector_b: Option<(u8, CollectorOp)>,
    pub max_shift: u8,
    pub collector_a: CollectorOp,
    pub ashift: bool,
    pub scale_input_d: Option<u8>,
}

/// CuTe DSL's `tcgen05.CollectorOp`: what a
/// Rubin `tcgen05.mma` does with the collector buffer of its A or B operand
/// (PTX `.collector::a::<op>` and `.collector::b::<op>`). `FILL` reads the
/// operand from shared memory and keeps a copy in the collector; `USE`
/// reads it from the collector and keeps it there; `LASTUSE` reads it from
/// the collector and releases it; `DISCARD` (the default, no qualifier)
/// reads shared memory and leaves nothing behind.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CollectorOp {
    #[default]
    DISCARD,
    LASTUSE,
    FILL,
    USE,
}

impl CollectorOp {
    /// The PTX qualifier's last part.
    pub fn ptx(self) -> &'static str {
        match self {
            CollectorOp::DISCARD => "discard",
            CollectorOp::LASTUSE => "lastuse",
            CollectorOp::FILL => "fill",
            CollectorOp::USE => "use",
        }
    }

    /// Whether the MMA reads its operand from the collector buffer rather
    /// than shared memory.
    pub fn reads_collector(self) -> bool {
        matches!(self, CollectorOp::USE | CollectorOp::LASTUSE)
    }
}

/// The Rubin parts of a `tcgen05.mma`:
/// `wide_k`, K doubled (`.kind::f8f6f4` and `.kind::mxf8f6f4` K = 64,
/// `.kind::mxf4nvf4` K = 128), and the collector usage of A and B.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sm107Umma {
    pub wide_k: bool,
    pub collector_a: CollectorOp,
    pub collector_b: CollectorOp,
}

/// The scale factors of a block-scaled `tcgen05.mma`: their type (`F8E8M0FNU` or `F8E4M3FN`), the K
/// elements each scales (`vec`, 16 or 32), and whether the instruction is
/// the sm_103a K = 96 form (`.block16` / `.block32`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UmmaScale {
    pub sf: DType,
    pub vec: u32,
    pub k96: bool,
}

impl UmmaKind {
    /// K: 32 bytes of the operand type (as it is held in shared memory);
    /// 96 for the sm_103a K = 96 forms.
    /// Twice that for a sparse kind (A stores half).
    pub fn k(&self) -> usize {
        let k = if self.scale.is_some_and(|s| s.k96) { 96 } else { self.kind.k() };
        // The Rubin forms double it.
        let k = if self.sm107.is_some_and(|r| r.wide_k) { 2 * k } else { k };
        if self.sparse { 2 * k } else { k }
    }

    /// K of A as stored (compressed for a sparse kind).
    pub fn a_k(&self) -> usize {
        if self.sparse { self.k() / 2 } else { self.k() }
    }

    /// The sparse instruction descriptor's flag (bit 2)
    /// and the metadata selector field (bits 0-1, CuTe C++'s `sparse_id2_`).
    pub const SPARSE_BIT: u32 = 2;
    pub const SPARSE_ID2_MASK: u32 = 3;

    /// The scale factors per row of A (column of B)
    /// one instruction reads, K / the vector size.
    pub fn scale_factors(&self) -> usize {
        self.scale.map_or(0, |s| self.k() / s.vec as usize)
    }

    /// The block-scaled instruction descriptor's scale-factor-ID fields:
    /// A's at bit 29 (2 bits), B's at bit 4.
    pub const A_SF_ID_BIT: u32 = 29;
    pub const B_SF_ID_BIT: u32 = 4;

    /// `(M, N, K)`.
    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        (self.m as usize, self.n as usize, self.k())
    }

    /// The 32-bit instruction descriptor (PTX ISA 9.7.18.4.2, Table 51;
    /// CuTe C++'s `UMMA::InstrDescriptor`): dense, no saturation or
    /// negation, `c_format` in bits 4-5 (F16 0, F32 1, S32 2), the A and B
    /// formats in bits 7-9 and 10-12, the A and B majors (1: MN-major) in
    /// bits 15 and 16, `N >> 3` in bits 17-22 and `M >> 4` in bits 24-28.
    /// Panics on a type the kind does not take.
    ///
    /// A block-scaled kind has CuTe C++'s
    /// `InstrDescriptorBlockScaled` (ported from
    /// include/cute/arch/mma_sm100_desc.hpp): no accumulator format (it is
    /// Float32), the B and A scale-factor IDs in bits 4-5 and 29-30 (0
    /// here; the runtime descriptor ORs them in), the scale format in bit 23
    /// (UE4M3 0, UE8M0 1) and K = 96 in bit 31.
    #[track_caller]
    pub fn idesc(&self) -> u32 {
        // The bits are packed by the public
        // builders (`prims::Tcgen05InstrDesc`, `Tcgen05MxInstrDesc`,
        // `Tcgen05MxOmmaInstrDesc`), which check each field.
        use crate::prims::descriptors as d;
        let fmt = |t: DType| {
            d::Tcgen05InputFormat::for_dtype(self.kind, t)
                .unwrap_or_else(|| panic!("tcgen05.mma.kind::{}: no operand format for {t}", self.kind.ptx()))
        };
        let major =
            |trans: bool| if trans { crate::nvgpu::OperandMajorMode::MN } else { crate::nvgpu::OperandMajorMode::K };
        let wide = self.sm107.is_some_and(|r| r.wide_k);
        let fail = |e: String| -> u32 { panic!("tcgen05.mma.kind::{} instruction descriptor: {e}", self.kind.ptx()) };
        let bits = |b: d::Bits<u32>| b.as_static().expect("a descriptor of constants");
        if let Some(sc) = self.scale {
            // UE5M3 is format 2 of the 2-bit field (bits 23-24) of
            // CuTe C++'s Sm107MXF4NVF4InstrDescriptor.
            let scale_format = d::MxScaleFormat::for_dtype(sc.sf)
                .unwrap_or_else(|| panic!("tcgen05.mma: no scale-factor format for {}", sc.sf));
            if self.kind == UmmaDataKind::Mxf8f6f4 {
                // K = 64 of .kind::mxf8f6f4 is bit 31.
                let f = d::Tcgen05MxInstrDescFields {
                    sparse: self.sparse,
                    a_format: fmt(self.a),
                    b_format: fmt(self.b),
                    scale_format,
                    a_major: major(self.trans_a),
                    b_major: major(self.trans_b),
                    n: self.n,
                    m: self.m,
                    k_dim: wide || sc.k96,
                    ..Default::default()
                };
                return d::Tcgen05MxInstrDesc::build(&f).map_or_else(fail, |x| bits(x.bits()));
            }
            // K = 128 of .kind::mxf4nvf4 is the K-size value 2 (bit 3
            // set, bit 31 clear); the other K = 128 and the sm_103a K = 96 are 1.
            let k_dim = match (wide, self.kind) {
                (true, UmmaDataKind::Mxf4Nvf4) => 2,
                (true, _) => 1,
                (false, _) => sc.k96 as u8,
            };
            let f = d::Tcgen05MxOmmaInstrDescFields {
                sparse: self.sparse,
                a_format: fmt(self.a),
                b_format: fmt(self.b),
                scale_format,
                a_major: major(self.trans_a),
                b_major: major(self.trans_b),
                n: self.n,
                m: self.m,
                k_dim,
                ..Default::default()
            };
            return d::Tcgen05MxOmmaInstrDesc::build(&f).map_or_else(fail, |x| bits(x.bits()));
        }
        let c_format = d::Tcgen05OutputFormat::for_dtype(self.acc)
            .unwrap_or_else(|| panic!("tcgen05.mma: no accumulator format for {}", self.acc));
        let f = d::Tcgen05InstrDescFields {
            sparse: self.sparse,
            c_format,
            a_format: fmt(self.a),
            b_format: fmt(self.b),
            a_major: major(self.trans_a),
            b_major: major(self.trans_b),
            n: self.n,
            m: self.m,
            // K = 64 of .kind::f8f6f4 (CuTe C++'s make_sm107_instr_desc).
            k_dim: wide,
            // B reuse of the .ws forms.
            max_shift: self.prims.max_shift,
            ..Default::default()
        };
        d::Tcgen05InstrDesc::build(&f).map_or_else(fail, |x| bits(x.bits()))
    }

    /// Why this combination of form and modifiers is
    /// no `tcgen05.mma` the PTX ISA defines (or cute-rs models), if it is
    /// not: the `.ws` shapes (`cta_group::1`, M of 32, 64 or 128, N of 64,
    /// 128 or 256, the dense kinds), `.ashift` (A in TMEM, M of 128 or 256,
    /// not with `.collector::a::fill` or `::use`, not `.ws`), `scale-input-d`
    /// (0 to 15, `.kind::f16` and `tf32` only), the collector buffers (B's
    /// 0 to 3, only on `.ws`; A's not on `.ws`), and the block-scaled sparse
    /// kinds (one, two or four scale factors per row: K / 64 for
    /// `.kind::mxf8f6f4` and `mxf4`, K / 32 for `mxf4nvf4`).
    pub fn form_error(&self) -> Option<String> {
        let p = self.prims;
        let dense4 =
            matches!(self.kind, UmmaDataKind::F16 | UmmaDataKind::Tf32 | UmmaDataKind::I8 | UmmaDataKind::F8F6F4);
        if self.sparse
            && let Some(sc) = self.scale
        {
            let vec = match self.kind {
                UmmaDataKind::Mxf4Nvf4 => 32,
                _ => 64,
            };
            if sc.vec as usize != vec || sc.k96 || self.sm107.is_some() {
                return Some(format!(
                    "a sparse .kind::{} MMA scales each {vec} K elements (K = {}), not {}",
                    self.kind.ptx(),
                    self.k(),
                    sc.vec
                ));
            }
        }
        if p.ws {
            if self.two_cta || ![32, 64, 128].contains(&self.m) || ![64, 128, 256].contains(&self.n) {
                return Some(format!(
                    "a weight-stationary MMA is cta_group::1 with M of 32, 64 or 128 and N of 64, 128 or 256, not \
                     M = {} and N = {}{}",
                    self.m,
                    self.n,
                    if self.two_cta { " with cta_group::2" } else { "" }
                ));
            }
            if !dense4 || self.scale.is_some() || self.sm107.is_some() {
                return Some(format!(".ws takes .kind::f16, tf32, f8f6f4 or i8, not .kind::{}", self.kind.ptx()));
            }
            if p.collector_a != CollectorOp::DISCARD || p.ashift {
                return Some(".ws has no .collector::a or .ashift".into());
            }
            if self.sparse && self.kind == UmmaDataKind::Tf32 {
                return Some("cute-rs does not model the 1:2 sparsity of .kind::tf32".into());
            }
            if self.a_tmem && self.m != 128 {
                return Some(format!(
                    "cute-rs models A in tensor memory for .ws with M = 128 only (Layout D), not M = {}",
                    self.m
                ));
            }
        } else {
            if p.collector_b.is_some() || p.max_shift != 0 {
                return Some("B's collector buffers and the max-shift field belong to .ws".into());
            }
        }
        if let Some((i, _)) = p.collector_b
            && i > 3
        {
            return Some(format!("B's collector buffer is b0 to b3, not b{i}"));
        }
        if p.max_shift > 3 {
            return Some(format!("the max-shift field is 2 bits, not {}", p.max_shift));
        }
        if p.ashift {
            if !self.a_tmem || ![128, 256].contains(&self.m) {
                return Some(format!(
                    ".ashift reads A from tensor memory with M of 128 or 256, not M = {}{}",
                    self.m,
                    if self.a_tmem { "" } else { " from shared memory" }
                ));
            }
            if matches!(p.collector_a, CollectorOp::FILL | CollectorOp::USE) {
                return Some(format!(".ashift with .collector::a::{}: only lastuse or discard", p.collector_a.ptx()));
            }
        }
        if let Some(s) = p.scale_input_d {
            if !matches!(self.kind, UmmaDataKind::F16 | UmmaDataKind::Tf32) || self.scale.is_some() {
                return Some(format!("scale-input-d is for .kind::f16 and tf32, not .kind::{}", self.kind.ptx()));
            }
            if s > 15 {
                return Some(format!("scale-input-d is 0 to 15, not {s}"));
            }
        }
        if p.collector_a != CollectorOp::DISCARD && self.sm107.is_some() {
            return Some("the A collector is given twice (prims and the sm_107 form)".into());
        }
        None
    }

    /// The dense instruction descriptor's K-size bit (29).
    pub const K64_BIT: u32 = 29;

    /// The instruction's name with its qualifiers.
    pub fn ptx(&self) -> String {
        let wide = self.sm107.is_some_and(|r| r.wide_k);
        let vec32 = self.scale.is_some_and(|s| s.vec == 32);
        let scale = match (self.kind, self.scale.is_some_and(|s| s.k96)) {
            // K = 128 names the vector size (CuTe C++'s SM107_MMA_MXF4NVF4_SS).
            (UmmaDataKind::Mxf4Nvf4, _) if wide && vec32 => ".block_scale.block32",
            (UmmaDataKind::Mxf4Nvf4, _) if wide => ".block_scale.block16",
            (UmmaDataKind::Mxf8f6f4, _) => ".block_scale",
            (UmmaDataKind::Mxf4, false) => ".block_scale.scale_vec::2X",
            (UmmaDataKind::Mxf4, true) => ".block_scale.block32",
            (UmmaDataKind::Mxf4Nvf4, false) => ".block_scale.scale_vec::4X",
            (UmmaDataKind::Mxf4Nvf4, true) => ".block_scale.block16",
            _ => "",
        };
        // The collector qualifiers (DISCARD, the default, is not written).
        let mut collect = String::new();
        // .ashift, then the collector usage (verified against LLVM 23's
        // NVPTX printer: `.kind::tf32.ashift.collector::a::lastuse`, `.collector::b2::use`).
        if self.prims.ashift {
            collect.push_str(".ashift");
        }
        if self.prims.collector_a != CollectorOp::DISCARD {
            collect.push_str(&format!(".collector::a::{}", self.prims.collector_a.ptx()));
        }
        if let Some((i, op)) = self.prims.collector_b
            && op != CollectorOp::DISCARD
        {
            collect.push_str(&format!(".collector::b{i}::{}", op.ptx()));
        }
        if let Some(r) = self.sm107 {
            for (which, op) in [("a", r.collector_a), ("b", r.collector_b)] {
                if op != CollectorOp::DISCARD {
                    collect.push_str(&format!(".collector::{which}::{}", op.ptx()));
                }
            }
        }
        format!(
            "tcgen05.mma{}{}.cta_group::{}.kind::{}{scale}{collect}",
            if self.prims.ws { ".ws" } else { "" },
            if self.sparse { ".sp" } else { "" },
            if self.two_cta { 2 } else { 1 },
            self.kind.ptx()
        )
    }

    /// The instruction descriptor's negate-A bit (13).
    pub const NEGATE_A_BIT: u32 = 13;
    /// The instruction descriptor's negate-B bit (14).
    pub const NEGATE_B_BIT: u32 = 14;

    /// The 32-bit words of `{disable-output-lane}`: 4 for `cta_group::1`, 8
    /// for `::2`.
    pub fn disable_lane_words(&self) -> usize {
        if self.two_cta { 8 } else { 4 }
    }
}

/// One `wgmma.mma_async` instruction: shape `m64nNkK`, operands of types `a` and `b`
/// accumulated in `acc`:
///
/// | A, B                          | accumulator | K  |
/// |-------------------------------|-------------|----|
/// | F16, F16 or BF16, BF16        | F16, F32 (BF16: F32) | 16 |
/// | F8E4M3FN or F8E5M2 (each)     | F16, F32    | 32 |
/// | I8 or U8 (each)               | I32         | 32 |
///
/// A and B are transposed (MN-major) when `trans_a` and `trans_b` are set
/// (the PTX `imm-trans-a` and `imm-trans-b`, which only the 16-bit types
/// have). With `a_regs` A comes from registers (the PTX ISA's register
/// form, CuTe DSL's `OperandSource.RMEM`), and is K-major.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WgmmaKind {
    pub n: u32,
    pub a: DType,
    pub b: DType,
    pub acc: DType,
    pub trans_a: bool,
    pub trans_b: bool,
    pub a_regs: bool,
}

impl WgmmaKind {
    /// `[M, N, K]`: `[64, n, 16]` for 16-bit operands, `[64, n, 32]` for
    /// 8-bit ones.
    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        (64, self.n as usize, self.k())
    }

    /// K: 32 bytes of the operand type.
    pub fn k(&self) -> usize {
        256 / self.a.bits()
    }

    /// Accumulators per thread: `N / 2`.
    pub fn accumulators(&self) -> usize {
        self.n as usize / 2
    }

    /// Elements of A per thread in the register form: 8 (16-bit) or 16
    /// (8-bit), four 32-bit registers.
    pub fn a_fragment(&self) -> usize {
        64 * self.k() / 128
    }

    /// Whether the operands are 8-bit integers (the `.s32` accumulator
    /// form, which takes neither scale nor transpose immediates).
    pub fn is_integer(&self) -> bool {
        self.acc == DType::I32
    }

    /// Whether the operands are 16-bit floats (the forms with transpose
    /// immediates).
    pub fn is_16bit(&self) -> bool {
        self.a.bits() == 16
    }

    /// The instruction's PTX name, without operands.
    pub fn ptx(&self) -> String {
        let t = |d: DType| match d {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F8E4M3FN => "e4m3",
            DType::F8E5M2 => "e5m2",
            DType::I8 => "s8",
            DType::U8 => "u8",
            DType::I32 => "s32",
            _ => "f32",
        };
        format!("wgmma.mma_async.sync.aligned.m64n{}k{}.{}.{}.{}", self.n, self.k(), t(self.acc), t(self.a), t(self.b))
    }
}

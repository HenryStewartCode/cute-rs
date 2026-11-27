// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm100.hpp (the K and scale-factor vector
//   sizes of SM100_MMA_MXF8F6F4_SS_SPARSE and SM100_MMA_MXF4NVF4_SS_SPARSE),
//   include/cute/atom/mma_traits_sm100_frag.hpp (UMMA::tmem_frg_ws, the
//   weight-stationary accumulator layouts, in `tmem_frg_ws_layout`) and
//   include/cute/arch/mma_sm120_sparse.hpp (the operands of the
//   SM120_SPARSE_16x8x{64,128}_TN_VS block-scaled sparse MMAs),
//     mma_traits_sm100.hpp: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES.
//     mma_traits_sm100_frag.hpp: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//     mma_sm120_sparse.hpp: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
//     All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The MMA forms only CuTe DSL's primitives issue (`cutlass.experimental.primitives`): no CuTe MMA
//! atom has them. Each takes raw operands, as the PTX instruction does: D's
//! tensor-memory address, A's shared-memory descriptor or TMEM address,
//! B's descriptor and an instruction descriptor (`prims::descriptors`).
//!
//! * [`tcgen05_mma`]: the dense `tcgen05.mma` with the modifiers the atoms
//!   lack ([`Tcgen05MmaOptions`]: `.collector::a::*`, `.ashift`,
//!   `scale-input-d` and the disable-output-lane mask; `.collector::b::*`
//!   is the sm_107 form).
//! * [`tcgen05_mma_ws`] and [`tcgen05_mma_ws_sp`]: weight-stationary
//!   `tcgen05.mma.ws{.sp}.cta_group::1` with B's four collector buffers
//!   ([`Tcgen05MmaWsOptions`]).
//! * [`tcgen05_mma_sp_block_scale`]: `tcgen05.mma.sp` with `.block_scale`.
//! * [`mma_sp_block_scale`]: the SM120 warp-level sparse block-scaled
//!   `mma.sp::ordered_metadata`.
//!
//! The tcgen05 forms are issued by every thread that executes them, so
//! call them inside [`elect_one`](crate::elect_one), from the warp that
//! commits ([`tcgen05::commit`](crate::tcgen05::commit)). The instruction
//! descriptor must be known when the kernel is built (its shape and format
//! fields fix the instruction); its negate bits, scale-factor IDs and
//! sparse selector may be anything, and reach the instruction as given.
//! [`tmem_frg_ws_layout`] is the TMEM layout of a `.ws` accumulator.

use super::descriptors::{
    Bits, Tcgen05InstrDesc, Tcgen05MxInstrDesc, Tcgen05MxOmmaInstrDesc, Tcgen05SmemDesc, TmemAddr,
};
use crate::builder;
use crate::ir::{DType, Op, SpBlockScaledMmaKind, UmmaDataKind, UmmaKind, UmmaPrims, UmmaScale, ValueId};
use crate::layout::Layout;
use crate::nvgpu::cpasync::CtaGroup;
use crate::tensor::{AddressSpace, Ptr};
use crate::value::{IntoValue, Numeric, Value};

pub use crate::ir::CollectorOp as Tcgen05MmaCollectorOp;

/// CuTe DSL's `Tcgen05MMAKind`: the `.kind` of a `tcgen05.mma`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tcgen05MmaKind {
    F16,
    TF32,
    F8F6F4,
    I8,
    MXF8F6F4,
    MXF4,
    MXF4NVF4,
}

impl Tcgen05MmaKind {
    /// The IR's kind.
    pub fn data_kind(self) -> UmmaDataKind {
        match self {
            Tcgen05MmaKind::F16 => UmmaDataKind::F16,
            Tcgen05MmaKind::TF32 => UmmaDataKind::Tf32,
            Tcgen05MmaKind::F8F6F4 => UmmaDataKind::F8F6F4,
            Tcgen05MmaKind::I8 => UmmaDataKind::I8,
            Tcgen05MmaKind::MXF8F6F4 => UmmaDataKind::Mxf8f6f4,
            Tcgen05MmaKind::MXF4 => UmmaDataKind::Mxf4,
            Tcgen05MmaKind::MXF4NVF4 => UmmaDataKind::Mxf4Nvf4,
        }
    }
}

/// CuTe DSL's `Tcgen05MMACollectorBBuffer`: one of the four B collector
/// buffers of a `.ws` MMA.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Tcgen05MmaCollectorBBuffer {
    #[default]
    B0,
    B1,
    B2,
    B3,
}

/// CuTe DSL's `Tcgen05MMAScaleVecSize` (`.scale_vec::1X`, `2X`, `4X`): the
/// scale factors per row of A and column of B of a block-scaled MMA.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tcgen05MmaScaleVec {
    X1,
    X2,
    X4,
}

impl Tcgen05MmaScaleVec {
    pub fn count(self) -> usize {
        match self {
            Tcgen05MmaScaleVec::X1 => 1,
            Tcgen05MmaScaleVec::X2 => 2,
            Tcgen05MmaScaleVec::X4 => 4,
        }
    }
}

/// A tensor-memory operand: a TMEM pointer, a [`TmemAddr`] or the raw
/// Int32 address.
#[derive(Clone, Copy, Debug)]
pub struct TmemOperand(Value<i32>);

impl<T: Numeric> From<&Ptr<T>> for TmemOperand {
    #[track_caller]
    fn from(p: &Ptr<T>) -> Self {
        assert!(
            p.memspace() == AddressSpace::Tmem,
            "cute::prims: this operand is a tensor-memory address, but the pointer is in {:?}",
            p.memspace()
        );
        TmemOperand(p.toint_i32())
    }
}

impl From<TmemAddr> for TmemOperand {
    #[track_caller]
    fn from(a: TmemAddr) -> Self {
        TmemOperand(a.value())
    }
}

impl From<Value<i32>> for TmemOperand {
    fn from(v: Value<i32>) -> Self {
        TmemOperand(v)
    }
}

impl TmemOperand {
    /// The Int32 address (the `tcgen05_shift` operation's).
    pub(crate) fn value(self) -> Value<i32> {
        self.0
    }
}

/// The A operand of a tcgen05 MMA: a shared-memory descriptor (a
/// [`Tcgen05SmemDesc`] or its raw u64) or a TMEM address (the `[a-tmem]`
/// form).
#[derive(Clone, Copy, Debug)]
pub enum MmaOperandA {
    Smem(Value<u64>),
    Tmem(Value<i32>),
}

impl From<Tcgen05SmemDesc> for MmaOperandA {
    #[track_caller]
    fn from(d: Tcgen05SmemDesc) -> Self {
        MmaOperandA::Smem(d.value())
    }
}

impl From<Value<u64>> for MmaOperandA {
    fn from(v: Value<u64>) -> Self {
        MmaOperandA::Smem(v)
    }
}

impl From<Value<i32>> for MmaOperandA {
    /// A raw Int32 is a TMEM address (a descriptor is a u64).
    fn from(v: Value<i32>) -> Self {
        MmaOperandA::Tmem(v)
    }
}

impl From<TmemOperand> for MmaOperandA {
    fn from(t: TmemOperand) -> Self {
        MmaOperandA::Tmem(t.0)
    }
}

impl From<TmemAddr> for MmaOperandA {
    #[track_caller]
    fn from(a: TmemAddr) -> Self {
        MmaOperandA::Tmem(a.value())
    }
}

impl<T: Numeric> From<&Ptr<T>> for MmaOperandA {
    #[track_caller]
    fn from(p: &Ptr<T>) -> Self {
        MmaOperandA::Tmem(TmemOperand::from(p).0)
    }
}

/// B's shared-memory descriptor: a [`Tcgen05SmemDesc`] or its raw u64.
#[derive(Clone, Copy, Debug)]
pub struct SmemDescOperand(Value<u64>);

impl From<Tcgen05SmemDesc> for SmemDescOperand {
    #[track_caller]
    fn from(d: Tcgen05SmemDesc) -> Self {
        SmemDescOperand(d.value())
    }
}

impl From<Value<u64>> for SmemDescOperand {
    fn from(v: Value<u64>) -> Self {
        SmemDescOperand(v)
    }
}

/// An instruction descriptor: the typed ones of `prims::descriptors`, or
/// raw bits.
#[derive(Clone, Copy, Debug)]
pub struct InstrDescOperand(Bits<u32>);

impl From<Tcgen05InstrDesc> for InstrDescOperand {
    fn from(d: Tcgen05InstrDesc) -> Self {
        InstrDescOperand(d.bits())
    }
}

impl From<Tcgen05MxInstrDesc> for InstrDescOperand {
    fn from(d: Tcgen05MxInstrDesc) -> Self {
        InstrDescOperand(d.bits())
    }
}

impl From<Tcgen05MxOmmaInstrDesc> for InstrDescOperand {
    fn from(d: Tcgen05MxOmmaInstrDesc) -> Self {
        InstrDescOperand(d.bits())
    }
}

impl From<u32> for InstrDescOperand {
    fn from(v: u32) -> Self {
        InstrDescOperand(Bits::Static(v))
    }
}

/// The keyword arguments of [`tcgen05_mma`] (CuTe DSL's `collector_op`,
/// `b_collector_op`, `a_shift`, `scale_input_d`, `write_disable_mask`).
#[derive(Clone, Debug, Default)]
pub struct Tcgen05MmaOptions {
    /// `.collector::a::<op>` (sm_100a, PTX 8.6).
    pub collector_op: Option<Tcgen05MmaCollectorOp>,
    /// `.collector::b::<op>`: the sm_107 form (PTX 9.4, sm_107f), whose
    /// shared-memory descriptors have the sm_107 encoding.
    pub b_collector_op: Option<Tcgen05MmaCollectorOp>,
    /// `.ashift` (A in TMEM, M of 128 or 256).
    pub a_shift: bool,
    /// `scale-input-d` (0 to 15; `.kind::f16` and `tf32`).
    pub scale_input_d: Option<u8>,
    /// The disable-output-lane words: 4 for `cta_group::1`, 8 for `::2`.
    pub write_disable_mask: Option<Vec<Value<i32>>>,
}

/// The keyword arguments of [`tcgen05_mma_ws`] and [`tcgen05_mma_ws_sp`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Tcgen05MmaWsOptions {
    /// B's collector buffer (`b0` when only `collector_op` is given).
    pub collector_b_buffer: Option<Tcgen05MmaCollectorBBuffer>,
    /// What the MMA does with that buffer (`DISCARD` when not given).
    pub collector_op: Option<Tcgen05MmaCollectorOp>,
    /// The zero-column-mask descriptor (u64).
    pub col_b_zero_mask: Option<Value<u64>>,
}

/// The keyword arguments of [`tcgen05_mma_sp_block_scale`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Tcgen05MmaSpBlockScaleOptions {
    /// `.scale_vec::NX`: 1X for `.kind::mxf8f6f4`, 2X for `mxf4`, 4X for
    /// `mxf4nvf4` (the default for each).
    pub scale_vec_size: Option<Tcgen05MmaScaleVec>,
    pub collector_op: Option<Tcgen05MmaCollectorOp>,
    pub b_collector_op: Option<Tcgen05MmaCollectorOp>,
    pub a_shift: bool,
}

/// The operand types a kind takes (in the order of their formats).
fn kind_types(kind: UmmaDataKind) -> &'static [DType] {
    match kind {
        UmmaDataKind::F16 => &[DType::F16, DType::BF16],
        UmmaDataKind::Tf32 => &[DType::TF32],
        UmmaDataKind::I8 => &[DType::U8, DType::I8],
        UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4 => {
            &[DType::F8E4M3FN, DType::F8E5M2, DType::F6E2M3FN, DType::F6E3M2FN, DType::F4E2M1FN]
        }
        UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4 => &[DType::F4E2M1FN],
    }
}

/// The instruction a prims call issues, from its kind, form and static
/// instruction descriptor: the [`UmmaKind`] whose descriptor equals `bits`
/// outside the bits `runtime` may set. Panics naming `what` when no
/// instruction has that descriptor.
#[track_caller]
fn decode(
    what: &str,
    kind: UmmaDataKind,
    bits: Bits<u32>,
    cta_group: CtaGroup,
    a_tmem: bool,
    sparse: bool,
    scale: Option<(Tcgen05MmaScaleVec, usize)>,
    prims: UmmaPrims,
) -> UmmaKind {
    let Some(bits) = bits.as_static() else {
        panic!(
            "cute::prims::{what}: the instruction descriptor is a runtime value; cute-rs needs its shape and format \
             fields when the kernel is built (build it from constants)"
        )
    };
    let fail = |why: String| -> ! { panic!("cute::prims::{what}: instruction descriptor {bits:#010x}: {why}") };
    let scaled = kind.is_block_scaled();
    if scaled != scale.is_some() {
        fail(if scaled {
            format!(".kind::{} is block-scaled: issue it with tcgen05_mma_sp_block_scale", kind.ptx())
        } else {
            format!(".kind::{} is not block-scaled", kind.ptx())
        });
    }
    // The bits the instruction takes at run time: negate A/B, and the sparse selector or the
    // scale-factor IDs; .ws: the max shift.
    let mut runtime = 1u32 << UmmaKind::NEGATE_A_BIT | 1 << UmmaKind::NEGATE_B_BIT;
    if sparse {
        runtime |= UmmaKind::SPARSE_ID2_MASK;
    }
    if scaled {
        runtime |= 3 << UmmaKind::A_SF_ID_BIT | 3 << UmmaKind::B_SF_ID_BIT;
    }
    let mut prims = prims;
    if prims.ws {
        prims.max_shift = (bits >> 30) as u8;
    }
    if bits >> 2 & 1 != sparse as u32 {
        fail(format!(
            "its sparse flag (bit 2) is {}, but the instruction is {}",
            bits >> 2 & 1,
            if sparse { "sparse" } else { "dense" }
        ));
    }
    let n = (bits >> 17 & 0x3f) << 3;
    let m = if scaled { (bits >> 27 & 3) << 7 } else { (bits >> 24 & 0x1f) << 4 };
    let m_ok = if scaled { m == 128 || m == 256 } else { m > 0 };
    if !m_ok || !(8..=256).contains(&n) {
        fail(format!("M = {m} and N = {n} are no .kind::{} shape", kind.ptx()));
    }
    let types = kind_types(kind);
    let accs: &[DType] = match kind {
        UmmaDataKind::I8 => &[DType::I32],
        UmmaDataKind::F16 | UmmaDataKind::F8F6F4 => &[DType::F32, DType::F16],
        _ => &[DType::F32],
    };
    let sfs: &[DType] = match (scale, kind) {
        (None, _) => &[DType::F32],
        (Some(_), UmmaDataKind::Mxf4Nvf4) => &[DType::F8E4M3FN, DType::F8E8M0FNU],
        (Some(_), _) => &[DType::F8E8M0FNU],
    };
    let want = bits & !runtime;
    for &a in types {
        for &b in types {
            for &acc in accs {
                for &sf in sfs {
                    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                        let cand = UmmaKind {
                            kind,
                            m,
                            n,
                            a,
                            b,
                            acc,
                            trans_a: ta,
                            trans_b: tb,
                            two_cta: cta_group == CtaGroup::TWO,
                            a_tmem,
                            scale: scale.map(|(_, vec)| UmmaScale { sf, vec: vec as u32, k96: false }),
                            sparse,
                            sm107: None,
                            prims,
                        };
                        if (a.bits() < 8 && ta) || (b.bits() < 8 && tb) {
                            continue;
                        }
                        if cand.idesc() == want {
                            return cand;
                        }
                    }
                }
            }
        }
    }
    fail(format!(
        "no .kind::{} instruction of M = {m}, N = {n} has these format, major-ness, K-size and reserved bits (A from \
         {})",
        kind.ptx(),
        if a_tmem { "tensor memory" } else { "shared memory" }
    ))
}

/// Push one prims `tcgen05.mma`.
#[track_caller]
fn issue(
    what: &str,
    kind: UmmaKind,
    idesc: Bits<u32>,
    d: TmemOperand,
    a: MmaOperandA,
    b: SmemDescOperand,
    enable_input_d: Value<bool>,
    disable_lane: Vec<ValueId>,
    sf: Option<(ValueId, ValueId)>,
    meta: Option<ValueId>,
    zero_col_mask: Option<ValueId>,
) {
    let loc = std::panic::Location::caller();
    if let Some(why) = kind.form_error() {
        panic!("cute::prims::{what}: {}: {why}", kind.ptx());
    }
    if !kind.two_cta && kind.m == 32 && !kind.prims.ws {
        panic!("cute::prims::{what}: M = 32 is a .ws shape");
    }
    let desc_a = match a {
        MmaOperandA::Smem(v) => v.checked(),
        MmaOperandA::Tmem(v) => v.checked(),
    };
    let elect = IntoValue::<bool>::into_value(true).checked();
    let op = Op::Tcgen05Mma {
        kind,
        taddr: d.0.checked(),
        desc_a,
        desc_b: b.0.checked(),
        enable_input_d: enable_input_d.checked(),
        elect,
        idesc: Some(idesc.value().checked()),
        disable_lane,
        sf,
        meta,
        zero_col_mask,
    };
    builder::with(|bs| bs.push(op, loc));
}

/// CuTe DSL's `tcgen05_mma(mma_kind, cta_group, d, a, b, idesc,
/// enable_input_d, *, collector_op, b_collector_op, a_shift, scale_input_d,
/// write_disable_mask)`: `tcgen05.mma.cta_group::{1,2}.kind::{f16, tf32,
/// f8f6f4, i8}` (sm_100a; PTX ISA 8.6), `D = A B` or, with
/// `enable_input_d`, `D = A B + D` (`D 2^-scale_input_d` with
/// `scale_input_d`). `d` is D's TMEM address, `a` A's descriptor or TMEM
/// address, `b` B's descriptor and `idesc` a [`Tcgen05InstrDesc`]. Issued
/// by the executing thread (call it inside `elect_one`).
#[track_caller]
pub fn tcgen05_mma(
    mma_kind: Tcgen05MmaKind,
    cta_group: CtaGroup,
    d: impl Into<TmemOperand>,
    a: impl Into<MmaOperandA>,
    b: impl Into<SmemDescOperand>,
    idesc: impl Into<InstrDescOperand>,
    enable_input_d: impl IntoValue<bool>,
    options: Tcgen05MmaOptions,
) {
    let a = a.into();
    let idesc = idesc.into().0;
    let prims = UmmaPrims {
        thread_issue: true,
        collector_a: if options.b_collector_op.is_some() {
            Tcgen05MmaCollectorOp::DISCARD
        } else {
            options.collector_op.unwrap_or_default()
        },
        ashift: options.a_shift,
        scale_input_d: options.scale_input_d,
        ..Default::default()
    };
    let a_tmem = matches!(a, MmaOperandA::Tmem(_));
    let mut kind = decode("tcgen05_mma", mma_kind.data_kind(), idesc, cta_group, a_tmem, false, None, prims);
    if let Err(why) = super::descriptors::check_shape(kind.kind, kind.m, kind.n, cta_group) {
        panic!("cute::prims::tcgen05_mma: {why}");
    }
    // .collector::b::* is the sm_107 form.
    if let Some(bop) = options.b_collector_op {
        kind.sm107 = Some(crate::ir::Sm107Umma {
            wide_k: false,
            collector_a: options.collector_op.unwrap_or_default(),
            collector_b: bop,
        });
    }
    let disable_lane: Vec<ValueId> =
        options.write_disable_mask.unwrap_or_default().into_iter().map(|v| v.checked()).collect();
    if !disable_lane.is_empty() && disable_lane.len() != kind.disable_lane_words() {
        panic!(
            "cute::prims::tcgen05_mma: write_disable_mask has {} words; cta_group::{} takes {}",
            disable_lane.len(),
            cta_group as u32,
            kind.disable_lane_words()
        );
    }
    issue(
        "tcgen05_mma",
        kind,
        idesc,
        d.into(),
        a,
        b.into(),
        enable_input_d.into_value(),
        disable_lane,
        None,
        None,
        None,
    );
}

/// The `.ws` prims part of a weight-stationary call.
fn ws_prims(options: &Tcgen05MmaWsOptions) -> UmmaPrims {
    let buffer = options.collector_b_buffer.unwrap_or_default() as u8;
    UmmaPrims {
        thread_issue: true,
        ws: true,
        collector_b: Some((buffer, options.collector_op.unwrap_or_default())),
        ..Default::default()
    }
}

/// CuTe DSL's `tcgen05_mma_ws(mma_kind, d, a, b, idesc, enable_input_d, *,
/// collector_b_buffer, collector_op, col_b_zero_mask)`:
/// `tcgen05.mma.ws.cta_group::1.kind::{f16, tf32, f8f6f4, i8}` (sm_100a;
/// PTX ISA 8.6), weight stationary: M of 32, 64 or 128 (D in the data-path
/// Layouts G, E and D, [`tmem_frg_ws_layout`]), N of 64, 128 or 256. B may
/// be kept in one of four collector buffers (`FILL` it, then `USE` it and
/// `LASTUSE` it with the same descriptor).
#[track_caller]
pub fn tcgen05_mma_ws(
    mma_kind: Tcgen05MmaKind,
    d: impl Into<TmemOperand>,
    a: impl Into<MmaOperandA>,
    b: impl Into<SmemDescOperand>,
    idesc: impl Into<InstrDescOperand>,
    enable_input_d: impl IntoValue<bool>,
    options: Tcgen05MmaWsOptions,
) {
    ws_common("tcgen05_mma_ws", mma_kind, d.into(), a.into(), b.into(), idesc.into(), enable_input_d, None, options)
}

/// CuTe DSL's `tcgen05_mma_ws_sp(mma_kind, d, a, b, idesc, enable_input_d,
/// sparse_metadata, *, ...)`: `tcgen05.mma.ws.sp.cta_group::1`, A 2:4
/// structured sparse (`M x K/2` as stored) with its metadata in TMEM at
/// `sparse_metadata` (the selector in the descriptor's bits 0-1).
/// `.kind::tf32` (1:2 sparsity) is not modelled.
#[track_caller]
pub fn tcgen05_mma_ws_sp(
    mma_kind: Tcgen05MmaKind,
    d: impl Into<TmemOperand>,
    a: impl Into<MmaOperandA>,
    b: impl Into<SmemDescOperand>,
    idesc: impl Into<InstrDescOperand>,
    enable_input_d: impl IntoValue<bool>,
    sparse_metadata: impl Into<TmemOperand>,
    options: Tcgen05MmaWsOptions,
) {
    let meta = sparse_metadata.into();
    ws_common(
        "tcgen05_mma_ws_sp",
        mma_kind,
        d.into(),
        a.into(),
        b.into(),
        idesc.into(),
        enable_input_d,
        Some(meta),
        options,
    )
}

#[track_caller]
fn ws_common(
    what: &str,
    mma_kind: Tcgen05MmaKind,
    d: TmemOperand,
    a: MmaOperandA,
    b: SmemDescOperand,
    idesc: InstrDescOperand,
    enable_input_d: impl IntoValue<bool>,
    meta: Option<TmemOperand>,
    options: Tcgen05MmaWsOptions,
) {
    let a_tmem = matches!(a, MmaOperandA::Tmem(_));
    let kind =
        decode(what, mma_kind.data_kind(), idesc.0, CtaGroup::ONE, a_tmem, meta.is_some(), None, ws_prims(&options));
    let zcm = options.col_b_zero_mask.map(|v| v.checked());
    issue(
        what,
        kind,
        idesc.0,
        d,
        a,
        b,
        enable_input_d.into_value(),
        Vec::new(),
        None,
        meta.map(|m| m.0.checked()),
        zcm,
    );
}

/// CuTe DSL's `tcgen05_mma_sp_block_scale(mma_kind, cta_group, d, a, b,
/// idesc, enable_input_d, sparse_metadata, scale_a, scale_b, *,
/// scale_vec_size, collector_op, b_collector_op, a_shift)`:
/// `tcgen05.mma.sp.cta_group::{1,2}.kind::{mxf8f6f4, mxf4,
/// mxf4nvf4}.block_scale{.scale_vec::NX}`: A structured sparse (FP4 by
/// pairs) with metadata in TMEM, the scale factors of A and B in TMEM at
/// `scale_a` and `scale_b` (their IDs in the descriptor). K is 64 for
/// `.kind::mxf8f6f4` (one scale factor per row, 1X) and 128 for the FP4
/// kinds (2X, UE8M0, for `mxf4`; 4X, UE4M3 or UE8M0, for `mxf4nvf4`), as
/// CuTe C++'s `SM100_MMA_MXF8F6F4_SS_SPARSE` and
/// `SM100_MMA_MXF4NVF4_SS_SPARSE`. `idesc` is a [`Tcgen05MxInstrDesc`] or
/// [`Tcgen05MxOmmaInstrDesc`] with the sparse flag.
#[track_caller]
pub fn tcgen05_mma_sp_block_scale(
    mma_kind: Tcgen05MmaKind,
    cta_group: CtaGroup,
    d: impl Into<TmemOperand>,
    a: impl Into<MmaOperandA>,
    b: impl Into<SmemDescOperand>,
    idesc: impl Into<InstrDescOperand>,
    enable_input_d: impl IntoValue<bool>,
    sparse_metadata: impl Into<TmemOperand>,
    scale_a: impl Into<TmemOperand>,
    scale_b: impl Into<TmemOperand>,
    options: Tcgen05MmaSpBlockScaleOptions,
) {
    let what = "tcgen05_mma_sp_block_scale";
    let data = mma_kind.data_kind();
    let (k, default_vec) = match data {
        UmmaDataKind::Mxf8f6f4 => (64, Tcgen05MmaScaleVec::X1),
        UmmaDataKind::Mxf4 => (128, Tcgen05MmaScaleVec::X2),
        UmmaDataKind::Mxf4Nvf4 => (128, Tcgen05MmaScaleVec::X4),
        _ => panic!("cute::prims::{what}: .kind::{} is not block-scaled (mxf8f6f4, mxf4 or mxf4nvf4)", data.ptx()),
    };
    let sv = options.scale_vec_size.unwrap_or(default_vec);
    assert!(
        sv == default_vec,
        "cute::prims::{what}: .kind::{} takes .scale_vec::{}X when sparse (cute-rs models CuTe C++'s sparse forms), \
         not {}X",
        data.ptx(),
        default_vec.count(),
        sv.count()
    );
    let a = a.into();
    let a_tmem = matches!(a, MmaOperandA::Tmem(_));
    let prims = UmmaPrims {
        thread_issue: true,
        collector_a: if options.b_collector_op.is_some() {
            Tcgen05MmaCollectorOp::DISCARD
        } else {
            options.collector_op.unwrap_or_default()
        },
        ashift: options.a_shift,
        ..Default::default()
    };
    let idesc = idesc.into().0;
    let mut kind = decode(what, data, idesc, cta_group, a_tmem, true, Some((sv, k / sv.count())), prims);
    if let Err(why) = super::descriptors::check_shape(kind.kind, kind.m, kind.n, cta_group) {
        panic!("cute::prims::{what}: {why}");
    }
    if let Some(bop) = options.b_collector_op {
        kind.sm107 = Some(crate::ir::Sm107Umma {
            wide_k: false,
            collector_a: options.collector_op.unwrap_or_default(),
            collector_b: bop,
        });
    }
    let (sa, sb, e) = (scale_a.into(), scale_b.into(), sparse_metadata.into());
    issue(
        what,
        kind,
        idesc,
        d.into(),
        a,
        b.into(),
        enable_input_d.into_value(),
        Vec::new(),
        Some((sa.0.checked(), sb.0.checked())),
        Some(e.0.checked()),
        None,
    );
}

/// CuTe DSL's `MMABlockScaleKind` of the warp-level block-scaled MMAs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmaBlockScaleKind {
    MXF8F6F4,
    MXF4,
    MXF4NVF4,
}

/// The scale-factor registers of [`mma_sp_block_scale`]: the register
/// (`sfa` or `sfb`), and the byte and thread ids selecting the factors in
/// it (byte `byte_id + j` holds factor `j`).
#[derive(Clone, Copy, Debug)]
pub struct ScaleOperand {
    pub reg: Value<u32>,
    pub byte_id: u8,
    pub thread_id: u8,
}

impl From<Value<u32>> for ScaleOperand {
    fn from(reg: Value<u32>) -> Self {
        ScaleOperand { reg, byte_id: 0, thread_id: 0 }
    }
}

/// Why a warp-level sparse block-scaled kind is no
/// instruction the PTX ISA defines, if it is not: the types and scale
/// vector of each kind (`mxf8f6f4` 1X UE8M0; `mxf4` 2X UE8M0; `mxf4nvf4`
/// 2X UE8M0 or 4X UE4M3/UE8M0), byte-id (1X: 0 to 3; 2X: 0 or 2; 4X: 0) and
/// thread-id (A: 0 or 1, B: 0 to 3).
pub fn sp_block_scale_error(k: SpBlockScaledMmaKind) -> Option<String> {
    let fp8 = [DType::F8E4M3FN, DType::F8E5M2, DType::F6E2M3FN, DType::F6E3M2FN, DType::F4E2M1FN];
    if !fp8.contains(&k.a) || !fp8.contains(&k.b) {
        return Some(format!("A and B are FP8, FP6 or FP4, not {} and {}", k.a, k.b));
    }
    let ok = matches!(
        (k.is_fp4_packed(), k.scale_vec, k.sf),
        (false, 1, DType::F8E8M0FNU) | (true, 2, DType::F8E8M0FNU) | (true, 4, DType::F8E4M3FN | DType::F8E8M0FNU)
    );
    if !ok {
        return Some(format!(
            ".scale_vec::{}X with {} scale factors is not a form of these operand types",
            k.scale_vec, k.sf
        ));
    }
    let bid_ok = |b: u8| match k.scale_vec {
        1 => b < 4,
        2 => b == 0 || b == 2,
        _ => b == 0,
    };
    if !bid_ok(k.bid_a) || !bid_ok(k.bid_b) {
        return Some(format!("byte-ids {} and {} do not fit .scale_vec::{}X", k.bid_a, k.bid_b, k.scale_vec));
    }
    if k.tid_a > 1 || k.tid_b > 3 {
        return Some(format!("thread-id-a is 0 or 1 and thread-id-b 0 to 3, not {} and {}", k.tid_a, k.tid_b));
    }
    None
}

/// CuTe DSL's `mma_sp_block_scale(res, shape, scale_vec_size,
/// block_scale_format, kind, ...)`: the SM120 warp-level
/// `mma.sp::ordered_metadata.sync.aligned.m16n8k{64,128}.row.col.kind::<kind>.block_scale.scale_vec::<n>X`
/// (sm_120a, PTX ISA 8.7; 4X with UE8M0 PTX 9.1), executed by a converged
/// warp. `shape` is `(16, 8, 64)` (`mxf8f6f4`, 8-bit containers) or
/// `(16, 8, 128)` (packed FP4), `a` the four registers of the compressed A,
/// `b` four of B, `c` the four accumulators, `metadata` the sparsity
/// metadata register (`sparsity_selector` 0), and `scale_a`, `scale_b` the
/// scale-factor registers with their byte and thread ids. Returns D. The
/// inline PTX is not convergent to LLVM: keep it out of divergent code.
#[track_caller]
pub fn mma_sp_block_scale(
    shape: (usize, usize, usize),
    scale_vec_size: Tcgen05MmaScaleVec,
    block_scale_format: DType,
    kind: MmaBlockScaleKind,
    (a_type, b_type): (DType, DType),
    a: [Value<u32>; 4],
    b: [Value<u32>; 4],
    c: [Value<f32>; 4],
    metadata: Value<u32>,
    sparsity_selector: u8,
    scale_a: impl Into<ScaleOperand>,
    scale_b: impl Into<ScaleOperand>,
) -> [Value<f32>; 4] {
    let loc = std::panic::Location::caller();
    let (sa, sb) = (scale_a.into(), scale_b.into());
    let k = SpBlockScaledMmaKind {
        a: a_type,
        b: b_type,
        scale_vec: scale_vec_size.count() as u8,
        sf: block_scale_format,
        bid_a: sa.byte_id,
        tid_a: sa.thread_id,
        bid_b: sb.byte_id,
        tid_b: sb.thread_id,
    };
    let fail = |why: String| -> ! { panic!("cute::prims::mma_sp_block_scale: {why}") };
    if let Some(why) = sp_block_scale_error(k) {
        fail(why);
    }
    let name = match kind {
        MmaBlockScaleKind::MXF8F6F4 => "mxf8f6f4",
        MmaBlockScaleKind::MXF4 => "mxf4",
        MmaBlockScaleKind::MXF4NVF4 => "mxf4nvf4",
    };
    if k.kind_name() != name {
        fail(format!(
            "kind::{name} with {a_type} x {b_type}, .scale_vec::{}X and {} scales is kind::{}",
            k.scale_vec,
            block_scale_format,
            k.kind_name()
        ));
    }
    if shape != k.shape_mnk() {
        fail(format!("kind::{name} has the shape {:?}, not {shape:?}", k.shape_mnk()));
    }
    if sparsity_selector != 0 {
        fail(format!("the sparsity selector of m16n8k{} is 0, not {sparsity_selector}", shape.2));
    }
    let ids = |v: &[Value<u32>]| -> Vec<ValueId> { v.iter().map(|x| x.checked()).collect() };
    let (a, b) = (ids(&a), ids(&b));
    let c: Vec<ValueId> = c.iter().map(|x| x.checked()).collect();
    let (e, sfa, sfb) = (metadata.checked(), sa.reg.checked(), sb.reg.checked());
    let d = builder::with(|bs| {
        let d: Vec<ValueId> = (0..4).map(|_| bs.new_value(DType::F32)).collect();
        bs.push(Op::MmaSpBlockScaled { kind: k, d: d.clone(), a, b, c, e, sfa, sfb }, loc);
        d
    });
    [0, 1, 2, 3].map(|i| Value::from_id(d[i]))
}

/// The TMEM layout of a `.ws` accumulator of `m` rows (32, 64 or 128) and
/// `n` columns (64, 128 or 256) tiled over `rest` (MMA_M, MMA_N, ...), on
/// elements of `bits` bits each in a 32-bit column (CuTe C++'s
/// `UMMA::tmem_frg_ws_1sm<T>`): `((M, N), MMA_M, MMA_N, ...) ->` TMEM
/// element offset (lane at a stride of `2^16` 32-bit columns). M = 32 is
/// the PTX ISA's Layout G (1x4 data paths: column `j` of D in lanes
/// `32 (j / (N/4))` onwards), M = 64 Layout E (2x2), M = 128 Layout D.
#[track_caller]
pub fn tmem_frg_ws_layout(m: usize, n: usize, rest: &[usize], bits: usize) -> Layout {
    assert!([32, 64, 128].contains(&m), "cute::prims::tmem_frg_ws_layout: M is 32, 64 or 128, not {m}");
    assert!([64, 128, 256].contains(&n), "cute::prims::tmem_frg_ws_layout: N is 64, 128 or 256, not {n}");
    let atom = match m {
        32 => Layout::new((32, (n / 4, 4)), (1, (128, 32))),
        64 => Layout::new((64, (n / 2, 2)), (1, (128, 64))),
        _ => Layout::new((128, n), (1, 128)),
    };
    let rest_layout = Layout::col_major(rest.to_vec());
    let logical = crate::layout::tiled_product(&atom, &rest_layout);
    let restride = Layout::new((128, 16384), ((1usize << 21) / bits, 32 / bits));
    restride.compose(&logical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_accumulator_layouts() {
        let lane = 1 << 16;
        // Layout G: row i, column j in lane i + 32 (j / 16), column j % 16 (N = 64).
        let l = tmem_frg_ws_layout(32, 64, &[1, 1], 32);
        for (row, col) in [(0, 0), (5, 7), (31, 15), (0, 16), (17, 40), (31, 63)] {
            assert_eq!(l.call(row + 32 * col), (row + 32 * (col / 16)) * lane + col % 16);
        }
        // Layout E: lane i + 64 (j / 64), column j % 64 (N = 128).
        let l = tmem_frg_ws_layout(64, 128, &[1, 1], 32);
        for (row, col) in [(0, 0), (63, 63), (0, 64), (40, 127)] {
            assert_eq!(l.call(row + 64 * col), (row + 64 * (col / 64)) * lane + col % 64);
        }
        // Layout D.
        let l = tmem_frg_ws_layout(128, 64, &[1, 1], 32);
        assert_eq!(l.call(5 + 128 * 7), 5 * lane + 7);
        // A second N tile after the first's columns.
        let l = tmem_frg_ws_layout(32, 64, &[1, 2], 32);
        assert_eq!(l.call(32 * 64), 16);
    }

    #[test]
    fn sparse_block_scaled_kinds() {
        let k = |a, sv, sf, bid, tid| SpBlockScaledMmaKind {
            a,
            b: a,
            scale_vec: sv,
            sf,
            bid_a: bid,
            tid_a: tid,
            bid_b: 0,
            tid_b: 0,
        };
        let f4 = DType::F4E2M1FN;
        assert!(sp_block_scale_error(k(f4, 4, DType::F8E4M3FN, 0, 0)).is_none());
        assert_eq!(k(f4, 4, DType::F8E4M3FN, 0, 0).kind_name(), "mxf4nvf4");
        assert_eq!(k(f4, 2, DType::F8E8M0FNU, 2, 1).kind_name(), "mxf4");
        assert!(sp_block_scale_error(k(f4, 2, DType::F8E8M0FNU, 2, 1)).is_none());
        assert!(sp_block_scale_error(k(f4, 2, DType::F8E8M0FNU, 1, 0)).is_some());
        assert!(sp_block_scale_error(k(DType::F8E4M3FN, 1, DType::F8E8M0FNU, 3, 2)).is_some());
        assert_eq!(
            k(DType::F8E4M3FN, 1, DType::F8E8M0FNU, 0, 0).ptx(),
            "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.\
             e4m3.f32.ue8m0"
        );
    }
}

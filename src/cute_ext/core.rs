//! The operation types of cute_ext's core dialect that cute-rs's cute_ext
//! uses (CuTe DSL's `OperationTypeEnum`): the hardware operations its
//! copies, MMAs and reductions name.

use std::fmt;

/// An operation of cute_ext's core dialect. [`warp_reduce`](super::warp_reduce)
/// takes [`ReduceDownShuffle`](Self::ReduceDownShuffle) or
/// [`ReduceSwapShuffle`](Self::ReduceSwapShuffle); later cute_ext parts
/// use the TMA, MMA and TMEM copy kinds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OperationType {
    /// `SM90_TMA_LOAD`.
    Sm90TmaLoad,
    /// `SM90_TMA_LOAD_MULTICAST`.
    Sm90TmaLoadMulticast,
    /// `SM100_TMA_LOAD_2SM`.
    Sm100TmaLoad2Sm,
    /// `SM100_TMA_LOAD_2SM_MULTICAST`.
    Sm100TmaLoad2SmMulticast,
    /// `SM100_TMA_LOAD_2SM_IM2COL`.
    Sm100TmaLoad2SmIm2col,
    /// `SM100_TMA_LOAD_2SM_IM2COL_MULTICAST`.
    Sm100TmaLoad2SmIm2colMulticast,
    /// `SM100_MMA_1SM_SS`.
    Sm100Mma1SmSs,
    /// `SM100_MMA_1SM_TS`.
    Sm100Mma1SmTs,
    /// `SM100_MMA_2SM_SS`.
    Sm100Mma2SmSs,
    /// `SM100_MMA_2SM_TS`.
    Sm100Mma2SmTs,
    /// `SM100_MMA_SCALED_1SM_SS`.
    Sm100MmaScaled1SmSs,
    /// `SM100_MMA_SCALED_1SM_TS`.
    Sm100MmaScaled1SmTs,
    /// `SM100_MMA_SCALED_2SM_SS`.
    Sm100MmaScaled2SmSs,
    /// `SM100_MMA_SCALED_2SM_TS`.
    Sm100MmaScaled2SmTs,
    /// `SM100_COPY_T2R`.
    Sm100CopyT2r,
    /// `SM100_COPY_R2T`.
    Sm100CopyR2t,
    /// `ST_SHARED`.
    StShared,
    /// `LD_SHARED`.
    LdShared,
    /// `SREG_CTAID`.
    SregCtaid,
    /// `SM100_LAUNCH_CONTROL`.
    Sm100LaunchControl,
    /// `REDUCE_DOWN_SHUFFLE`: a reduction across lanes whose result every
    /// lane of the group holds.
    ReduceDownShuffle,
    /// `REDUCE_SWAP_SHUFFLE`: a transposing reduce-scatter that leaves
    /// lane `i` of the group with slice `i` of the result.
    ReduceSwapShuffle,
}

impl OperationType {
    /// Every operation type, in declaration order.
    pub const ALL: [OperationType; 22] = [
        OperationType::Sm90TmaLoad,
        OperationType::Sm90TmaLoadMulticast,
        OperationType::Sm100TmaLoad2Sm,
        OperationType::Sm100TmaLoad2SmMulticast,
        OperationType::Sm100TmaLoad2SmIm2col,
        OperationType::Sm100TmaLoad2SmIm2colMulticast,
        OperationType::Sm100Mma1SmSs,
        OperationType::Sm100Mma1SmTs,
        OperationType::Sm100Mma2SmSs,
        OperationType::Sm100Mma2SmTs,
        OperationType::Sm100MmaScaled1SmSs,
        OperationType::Sm100MmaScaled1SmTs,
        OperationType::Sm100MmaScaled2SmSs,
        OperationType::Sm100MmaScaled2SmTs,
        OperationType::Sm100CopyT2r,
        OperationType::Sm100CopyR2t,
        OperationType::StShared,
        OperationType::LdShared,
        OperationType::SregCtaid,
        OperationType::Sm100LaunchControl,
        OperationType::ReduceDownShuffle,
        OperationType::ReduceSwapShuffle,
    ];

    /// The enumerator's name in CuTe DSL (`SM90_TMA_LOAD`, ...).
    pub fn name(self) -> &'static str {
        use OperationType::*;
        match self {
            Sm90TmaLoad => "SM90_TMA_LOAD",
            Sm90TmaLoadMulticast => "SM90_TMA_LOAD_MULTICAST",
            Sm100TmaLoad2Sm => "SM100_TMA_LOAD_2SM",
            Sm100TmaLoad2SmMulticast => "SM100_TMA_LOAD_2SM_MULTICAST",
            Sm100TmaLoad2SmIm2col => "SM100_TMA_LOAD_2SM_IM2COL",
            Sm100TmaLoad2SmIm2colMulticast => "SM100_TMA_LOAD_2SM_IM2COL_MULTICAST",
            Sm100Mma1SmSs => "SM100_MMA_1SM_SS",
            Sm100Mma1SmTs => "SM100_MMA_1SM_TS",
            Sm100Mma2SmSs => "SM100_MMA_2SM_SS",
            Sm100Mma2SmTs => "SM100_MMA_2SM_TS",
            Sm100MmaScaled1SmSs => "SM100_MMA_SCALED_1SM_SS",
            Sm100MmaScaled1SmTs => "SM100_MMA_SCALED_1SM_TS",
            Sm100MmaScaled2SmSs => "SM100_MMA_SCALED_2SM_SS",
            Sm100MmaScaled2SmTs => "SM100_MMA_SCALED_2SM_TS",
            Sm100CopyT2r => "SM100_COPY_T2R",
            Sm100CopyR2t => "SM100_COPY_R2T",
            StShared => "ST_SHARED",
            LdShared => "LD_SHARED",
            SregCtaid => "SREG_CTAID",
            Sm100LaunchControl => "SM100_LAUNCH_CONTROL",
            ReduceDownShuffle => "REDUCE_DOWN_SHUFFLE",
            ReduceSwapShuffle => "REDUCE_SWAP_SHUFFLE",
        }
    }

    /// The operation type named `name` (as [`name`](Self::name) gives it).
    pub fn from_name(name: &str) -> Option<OperationType> {
        OperationType::ALL.into_iter().find(|op| op.name() == name)
    }
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! dtype interop (CuTe DSL's `cutlass.torch.dtype`): the DLPack `DLDataType` of every
//! cute-rs element type and back, and the names PyTorch gives the types (for diagnostics
//! and for matching a tensor a Python caller describes).
//!
//! The DLPack codes follow `dlpack.h` 1.x (kDLInt 0, kDLUInt 1, kDLFloat 2, kDLBfloat 4,
//! kDLBool 6, kDLFloat8_* 7 to 14, kDLFloat6_* 15 and 16, kDLFloat4_e2m1fn 17, all checked
//! against [`crate::external::dlpack::type_code`]). The packed views are DLPack lanes:
//! `Float4E2M1FNx2` is `float4_e2m1fn` with 2 lanes and `Float6E2M3FNx4` is
//! `float6_e2m3fn` with 4.
//!
//! TFloat32 has no DLPack or PyTorch type of its own: it exports as float32 and
//! imports back as Float32. FloatNV8E5M3FNU, Float8E4M3 (the IEEE-style one), Int128 and
//! Uint128 have no PyTorch name.

use crate::external::DLDataType;
use crate::external::DlpackError;
use crate::external::dlpack::type_code::*;
use crate::ir::DType;

/// The DLPack type of `dtype`, or none if DLPack has no code for it.
pub fn dl_data_type(dtype: DType) -> Option<DLDataType> {
    let (code, bits, lanes) = match dtype {
        DType::Bool => (KDL_BOOL, 8, 1),
        DType::I4 => (KDL_INT, 4, 1),
        DType::I8 => (KDL_INT, 8, 1),
        DType::I16 => (KDL_INT, 16, 1),
        DType::I32 => (KDL_INT, 32, 1),
        DType::I64 => (KDL_INT, 64, 1),
        DType::I128 => (KDL_INT, 128, 1),
        DType::U8 => (KDL_UINT, 8, 1),
        DType::U16 => (KDL_UINT, 16, 1),
        DType::U32 => (KDL_UINT, 32, 1),
        DType::U64 => (KDL_UINT, 64, 1),
        DType::U128 => (KDL_UINT, 128, 1),
        DType::F16 => (KDL_FLOAT, 16, 1),
        DType::BF16 => (KDL_BFLOAT, 16, 1),
        DType::F32 | DType::TF32 => (KDL_FLOAT, 32, 1),
        DType::F64 => (KDL_FLOAT, 64, 1),
        DType::F8E4M3 => (KDL_FLOAT8_E4M3, 8, 1),
        DType::F8E4M3B11FNUZ => (KDL_FLOAT8_E4M3B11FNUZ, 8, 1),
        DType::F8E4M3FN => (KDL_FLOAT8_E4M3FN, 8, 1),
        DType::F8E5M2 => (KDL_FLOAT8_E5M2, 8, 1),
        DType::F8E8M0FNU => (KDL_FLOAT8_E8M0FNU, 8, 1),
        DType::F6E2M3FN => (KDL_FLOAT6_E2M3FN, 6, 1),
        DType::F6E3M2FN => (KDL_FLOAT6_E3M2FN, 6, 1),
        DType::F4E2M1FN => (KDL_FLOAT4_E2M1FN, 4, 1),
        DType::F4E2M1FNx2 => (KDL_FLOAT4_E2M1FN, 4, 2),
        DType::F6E2M3FNx4 => (KDL_FLOAT6_E2M3FN, 6, 4),
        DType::F6E3M2FNx4 => (KDL_FLOAT6_E3M2FN, 6, 4),
        DType::F8E5M3FNU => return None,
    };
    Some(DLDataType { code, bits, lanes })
}

/// The cute-rs type of a DLPack type: the scalar types, and the packed views for the
/// `float4_e2m1fn` x 2 and `float6` x 4 lane forms. Unknown codes, widths and other lane
/// counts are refused.
pub fn dtype_from_dl(t: DLDataType) -> Result<DType, DlpackError> {
    let (scalar, lanes) = crate::external::dlpack::dtype_of(t, 0)?;
    Ok(match (scalar, lanes) {
        (d, 1) => d,
        (DType::F4E2M1FN, 2) => DType::F4E2M1FNx2,
        (DType::F6E2M3FN, 4) => DType::F6E2M3FNx4,
        (DType::F6E3M2FN, 4) => DType::F6E3M2FNx4,
        (d, n) => return Err(DlpackError(format!("{n} lanes of {} are not a cute-rs type", d.cute_name()))),
    })
}

/// PyTorch's name for `dtype` (`"torch.float16"`), if it has one.
pub fn torch_dtype_name(dtype: DType) -> Option<&'static str> {
    Some(match dtype {
        DType::Bool => "torch.bool",
        DType::I4 => "torch.int4",
        DType::I8 => "torch.int8",
        DType::I16 => "torch.int16",
        DType::I32 => "torch.int32",
        DType::I64 => "torch.int64",
        DType::U8 => "torch.uint8",
        DType::U16 => "torch.uint16",
        DType::U32 => "torch.uint32",
        DType::U64 => "torch.uint64",
        DType::F16 => "torch.float16",
        DType::BF16 => "torch.bfloat16",
        DType::F32 | DType::TF32 => "torch.float32",
        DType::F64 => "torch.float64",
        DType::F8E4M3FN => "torch.float8_e4m3fn",
        DType::F8E5M2 => "torch.float8_e5m2",
        DType::F8E8M0FNU => "torch.float8_e8m0fnu",
        DType::F4E2M1FNx2 => "torch.float4_e2m1fn_x2",
        _ => return None,
    })
}

/// The type PyTorch's name `name` stands for (`"torch.float32"` is [`DType::F32`]).
pub fn dtype_from_torch_name(name: &str) -> Option<DType> {
    DType::ALL.into_iter().filter(|&t| t != DType::TF32).find(|&t| torch_dtype_name(t) == Some(name))
}

/// A short description of `dtype` for diagnostics: its cute-rs name, then the PyTorch name
/// and DLPack type where there are some.
pub fn describe_dtype(dtype: DType) -> String {
    let mut s = dtype.cute_name().to_string();
    if let Some(n) = torch_dtype_name(dtype) {
        s += &format!(" ({n}");
        if let Some(d) = dl_data_type(dtype) {
            s += &format!(", DLPack code {} bits {} lanes {}", d.code, d.bits, d.lanes);
        }
        s += ")";
    } else if let Some(d) = dl_data_type(dtype) {
        s += &format!(" (DLPack code {} bits {} lanes {})", d.code, d.bits, d.lanes);
    }
    s
}

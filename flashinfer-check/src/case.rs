// SPDX-License-Identifier: Apache-2.0

//! A case directory written by `reference/gen.py`: its parameters, its input tensors, and
//! where the port's outputs go.

// The readers serve every runner; a given build may not use each of them.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use cute::cuda::CudaContext;
use cute::prelude::*;

pub type Res<T> = Result<T, Box<dyn Error>>;

/// The device and compiler every case runs on.
pub struct Gpu {
    pub ctx: CudaContext,
    pub tc: Toolchain,
}

/// One case: `meta.txt` and the files beside it.
pub struct Case {
    pub dir: PathBuf,
    pub name: String,
    meta: BTreeMap<String, String>,
}

/// A tensor read from `in.<name>.bin`: raw little-endian bytes in C order.
pub struct Tensor {
    /// PyTorch's name of the element type (`float16`, `bfloat16`, `int32`, ...).
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Case {
    pub fn load(dir: &Path) -> Res<Case> {
        let text = std::fs::read_to_string(dir.join("meta.txt"))?;
        let meta =
            text.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
        Ok(Case { dir: dir.to_path_buf(), name: dir.file_name().unwrap().to_string_lossy().into_owned(), meta })
    }

    /// A parameter, if the case has it.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(String::as_str)
    }

    /// A parameter; panics, naming it, if the case has none.
    pub fn str(&self, key: &str) -> &str {
        self.get(key).unwrap_or_else(|| panic!("case {} has no `{key}`", self.name))
    }

    pub fn int(&self, key: &str) -> usize {
        self.parsed(key)
    }

    pub fn i64(&self, key: &str) -> i64 {
        self.parsed(key)
    }

    pub fn float(&self, key: &str) -> f64 {
        self.parsed(key)
    }

    /// A parameter written from a Python `bool` (`0` or `1`).
    pub fn flag(&self, key: &str) -> bool {
        self.int(key) != 0
    }

    /// A parameter written from a Python list of integers.
    pub fn ints(&self, key: &str) -> Vec<usize> {
        self.str(key).split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
    }

    fn parsed<T: std::str::FromStr>(&self, key: &str) -> T {
        let s = self.str(key);
        s.parse().unwrap_or_else(|_| panic!("case {}: `{key}={s}` does not parse", self.name))
    }

    /// Whether the case has an input called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.meta.contains_key(&format!("in.{name}"))
    }

    /// The input called `name`.
    pub fn tensor(&self, name: &str) -> Tensor {
        let line = self.str(&format!("in.{name}"));
        let (dtype, shape) = line.split_once(' ').unwrap();
        let bytes = std::fs::read(self.dir.join(format!("in.{name}.bin")))
            .unwrap_or_else(|e| panic!("case {}: in.{name}.bin: {e}", self.name));
        Tensor { dtype: dtype.to_string(), shape: parse_shape(shape), bytes }
    }

    /// The element type FlashInfer's output `name` has.
    pub fn ref_dtype(&self, name: &str) -> &str {
        self.str(&format!("ref.{name}")).split(' ').next().unwrap()
    }

    /// The shape FlashInfer's output `name` has.
    pub fn ref_shape(&self, name: &str) -> Vec<usize> {
        parse_shape(self.str(&format!("ref.{name}")).split(' ').nth(1).unwrap())
    }

    /// Save the port's output `name` (`out.<name>.bin`): raw bytes of the same type and
    /// shape as FlashInfer's.
    pub fn write(&self, name: &str, bytes: &[u8]) -> Res<()> {
        let want: usize = self.ref_shape(name).iter().product::<usize>() * dtype_bytes(self.ref_dtype(name));
        if bytes.len() != want {
            return Err(format!("output `{name}` is {} bytes; FlashInfer's is {want}", bytes.len()).into());
        }
        Ok(std::fs::write(self.dir.join(format!("out.{name}.bin")), bytes)?)
    }

    /// Save output `name` from Float32 values, converted to FlashInfer's output type (each
    /// value must already be one of that type's).
    pub fn write_f32(&self, name: &str, values: &[f32]) -> Res<()> {
        self.write(name, &encode(self.ref_dtype(name), values))
    }
}

fn parse_shape(s: &str) -> Vec<usize> {
    if s == "scalar" { Vec::new() } else { s.split('x').map(|n| n.parse().unwrap()).collect() }
}

/// Bytes per element of a PyTorch type name.
pub fn dtype_bytes(dtype: &str) -> usize {
    match dtype {
        "uint8" | "int8" | "bool" | "float8_e4m3fn" | "float8_e5m2" => 1,
        "float16" | "bfloat16" | "int16" | "uint16" => 2,
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        other => panic!("no size known for dtype `{other}`"),
    }
}

/// Float32 values as the bytes of a `float16`, `bfloat16`, `float32` or FP8 tensor.
pub fn encode(dtype: &str, values: &[f32]) -> Vec<u8> {
    match dtype {
        "float16" => values.iter().flat_map(|&x| F16::from_f32(x).0.to_le_bytes()).collect(),
        "bfloat16" => values.iter().flat_map(|&x| BF16::from_f32(x).0.to_le_bytes()).collect(),
        "float32" => values.iter().flat_map(|&x| x.to_le_bytes()).collect(),
        "float8_e4m3fn" => values.iter().map(|&x| F8E4M3FN::from_f32(x).0).collect(),
        "float8_e5m2" => values.iter().map(|&x| F8E5M2::from_f32(x).0).collect(),
        other => panic!("cannot write Float32 values as `{other}`"),
    }
}

impl Tensor {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.bytes.len() / dtype_bytes(&self.dtype)
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn chunks<const N: usize>(&self) -> impl Iterator<Item = [u8; N]> + '_ {
        assert_eq!(dtype_bytes(&self.dtype), N, "{} elements are not {N} bytes", self.dtype);
        self.bytes.as_chunks::<N>().0.iter().copied()
    }

    /// A floating-point tensor's values, each converted exactly to Float32 (`float16`,
    /// `bfloat16`, `float32` and the FP8 types).
    pub fn f32s(&self) -> Vec<f32> {
        match self.dtype.as_str() {
            "float16" => self.chunks().map(|b| F16(u16::from_le_bytes(b)).to_f32()).collect(),
            "bfloat16" => self.chunks().map(|b| BF16(u16::from_le_bytes(b)).to_f32()).collect(),
            "float32" => self.chunks().map(f32::from_le_bytes).collect(),
            "float8_e4m3fn" => self.bytes.iter().map(|&b| F8E4M3FN(b).to_f32()).collect(),
            "float8_e5m2" => self.bytes.iter().map(|&b| F8E5M2(b).to_f32()).collect(),
            other => panic!("`{other}` is not a floating-point tensor"),
        }
    }

    /// An integer tensor's values as `i64` (any integer type, and `bool`).
    pub fn i64s(&self) -> Vec<i64> {
        match self.dtype.as_str() {
            "uint8" | "bool" => self.bytes.iter().map(|&b| b as i64).collect(),
            "int8" => self.bytes.iter().map(|&b| b as i8 as i64).collect(),
            "int16" => self.chunks().map(|b| i16::from_le_bytes(b) as i64).collect(),
            "uint16" => self.chunks().map(|b| u16::from_le_bytes(b) as i64).collect(),
            "int32" => self.chunks().map(|b| i32::from_le_bytes(b) as i64).collect(),
            "uint32" => self.chunks().map(|b| u32::from_le_bytes(b) as i64).collect(),
            "int64" => self.chunks().map(i64::from_le_bytes).collect(),
            "uint64" => self.chunks().map(|b| u64::from_le_bytes(b) as i64).collect(),
            other => panic!("`{other}` is not an integer tensor"),
        }
    }

    /// An integer tensor's values as `i32`; panics if one does not fit.
    pub fn i32s(&self) -> Vec<i32> {
        self.i64s().into_iter().map(|v| i32::try_from(v).expect("an index that does not fit i32")).collect()
    }

    /// An integer tensor's values as `usize`; panics on a negative one.
    pub fn usizes(&self) -> Vec<usize> {
        self.i64s().into_iter().map(|v| usize::try_from(v).expect("a negative index")).collect()
    }
}

/// Run `f`'s kernel once on host arguments (copied to the device and back) and return
/// them, so their bytes can be saved.
pub fn launch(gpu: &Gpu, kernel: &Kernel, cfg: LaunchConfig, mut args: Vec<KernelArg>) -> Res<Vec<KernelArg>> {
    let arch = gpu.ctx.best_target(&gpu.tc).ok_or("no target for this GPU")?;
    let f = gpu.ctx.load(&kernel.compile_with(&gpu.tc, &CompileOptions::new().gpu_arch(arch))?)?;
    gpu.ctx.launch(kernel, &f, cfg, &mut args)?;
    Ok(args)
}

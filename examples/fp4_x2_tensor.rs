// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// This example follows a CuTe DSL example of NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/tvm_ffi/fp4_x2_tensor.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and takes its tensors (a Float4E2M1FN tensor and the Uint8 tensor holding the
// same torch.float4_e2m1fn_x2 bytes, recast to Float4E2M1FN) and its 16 x 16
// random bytes from it. Those parts are under the BSD-3-Clause licence, and the
// rest of the file is cute-rs's own, under MIT OR Apache-2.0.
// THIRD-PARTY-NOTICES.md has the licence text and describes what is taken.

//! FP4 data from PyTorch: the recast half of CuTe DSL's
//! `cute/tvm_ffi/fp4_x2_tensor.py`.
//!
//! PyTorch stores `torch.float4_e2m1fn_x2` as bytes, two E2M1 values per
//! byte (element 0 in the low nibble). The example hands a kernel the same
//! data twice: as a Float4E2M1FN tensor and as the Uint8 tensor under it,
//! which the kernel recasts (`cute.recast_tensor(t_uint8, Float4E2M1FN)`).
//! Here the first is a `Float4E2M1FNx2` buffer (CuTe DSL's packed view,
//! one unit per byte) recast to Float4E2M1FN, and the second a Uint8 buffer
//! recast through `Float4E2M1FNx2` to Float4E2M1FN. The example's TVM-FFI
//! half (passing torch tensors straight to a compiled function) is not ported.
//!
//! The example only prints the tensors' layouts; this kernel also decodes
//! every value to Float16, eight at a time (one 32-bit vector of FP4 per
//! thread):
//!
//! * with integer operations (`build(.., Decode::Integer)`), which run on
//!   every target (the SM89 GPU test runs it): sign, exponent and mantissa
//!   of each nibble moved into Float16's fields;
//! * with `cvt_f4e2m1x8_to_f16x8` (`cvt.rn.f16x2.e2m1x2`, `Decode::Cvt`),
//!   which needs sm_100a or sm_120a (simulated and assembled by `ptxas`).
//!
//! Run with `cargo run --release --example fp4_x2_tensor`.

use cute::prelude::*;

/// The example's tensor: 16 x 16 bytes, 16 x 32 FP4 values.
pub const ROWS: usize = 16;
pub const BYTES_PER_ROW: usize = 16;

/// How the kernel decodes FP4 to Float16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode {
    /// Integer operations on the nibbles (every target).
    Integer,
    /// `cvt.rn.f16x2.e2m1x2` (sm_100a, sm_120a).
    Cvt,
}

/// Eight FP4 E2M1 values (a 32-bit word of them, element 0 in the low
/// nibble) to Float16 with integer operations: for a nibble `s e1 e0 m`, a
/// normal value (`e > 0`) is `(-1)^s 2^(e - 1) (1 + m / 2)`, Float16 bits
/// `s << 15 | (e + 14) << 10 | m << 9`; a subnormal one (`e = 0`) is `m /
/// 2`, `0x3800` or zero, with the sign.
pub fn decode_e2m1x8(word: Uint32) -> [Float16; 8] {
    std::array::from_fn(|k| {
        let n = (word >> (4 * k as u32)) & 0xfu32;
        let (sign, exp, man) = ((n >> 3u32) << 15u32, (n >> 1u32) & 3u32, n & 1u32);
        let normal = ((exp + 14u32) << 10u32) | (man << 9u32);
        let subnormal = man * 0x3800u32;
        let bits = sign | select(exp.eq(0u32), subnormal, normal);
        bits.to::<u16>().bitcast::<F16>()
    })
}

/// The kernel: thread `t` decodes the eight FP4 values of row `t / 4`,
/// columns `8 (t % 4)` to `+7`, from both tensors, into `from_f4` and
/// `from_uint8` (`ROWS` x 32 Float16).
pub fn build(decode: Decode) -> Kernel {
    let name = match decode {
        Decode::Integer => "fp4_x2_tensor",
        Decode::Cvt => "fp4_x2_tensor_cvt",
    };
    Kernel::builder(name).block_dim([(ROWS * BYTES_PER_ROW / 4) as u32, 1, 1]).build(move || {
        let t_f4x2 = param_tensor::<F4E2M1FNx2>("t_f4", Layout::row_major((ROWS, BYTES_PER_ROW)));
        let t_uint8 = param_tensor::<u8>("t_uint8", Layout::row_major((ROWS, BYTES_PER_ROW)));
        let from_f4 = param_tensor::<F16>("from_f4", Layout::row_major((ROWS, 2 * BYTES_PER_ROW)));
        let from_uint8 = param_tensor::<F16>("from_uint8", Layout::row_major((ROWS, 2 * BYTES_PER_ROW)));
        // The logical Float4E2M1FN tensors, (ROWS, 32): the packed view
        // recast to its lanes, and the bytes recast through it.
        let t_f4 = recast_tensor::<F4E2M1FN, _>(&t_f4x2);
        let t_f4_casted = recast_tensor::<F4E2M1FN, _>(&recast_tensor::<F4E2M1FNx2, _>(&t_uint8));
        assert_eq!(t_f4.shape(), t_f4_casted.shape());
        let t = thread_idx().x;
        let (row, chunk) = (t / 4, t % 4);
        for (src, out) in [(&t_f4, &from_f4), (&t_f4_casted, &from_uint8)] {
            // Sub-byte values move as whole 32-bit vectors.
            let frag = make_fragment::<F4E2M1FN>(8);
            autovec_copy(&local_tile(src, (1, 8), (row, chunk)), &frag);
            let values: [Value<F4E2M1FN>; 8] = std::array::from_fn(|k| frag.load(k));
            let halves = match decode {
                Decode::Integer => decode_e2m1x8(pack(values)),
                Decode::Cvt => cvt_f4e2m1x8_to_f16x8(values),
            };
            for (k, h) in halves.into_iter().enumerate() {
                out.store((row, chunk * 8 + k as i32), h);
            }
        }
    })
}

/// One block of a thread per eight FP4 values.
pub fn launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [(ROWS * BYTES_PER_ROW / 4) as u32, 1, 1])
}

/// The example's data: `torch.randint(0, 256, (16, 16), dtype=torch.uint8)`
/// (from a seeded generator here).
pub fn random_bytes(seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..ROWS * BYTES_PER_ROW)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// The arguments for `bytes`: the same bytes as Float4E2M1FNx2 and as
/// Uint8, and two zeroed outputs.
pub fn args(bytes: &[u8]) -> Vec<KernelArg> {
    let zeros = vec![F16::from_f32(0.0); 2 * bytes.len()];
    vec![
        KernelArg::Bytes { elem: DType::F4E2M1FNx2, data: bytes.to_vec() },
        KernelArg::buffer(bytes),
        KernelArg::buffer(&zeros),
        KernelArg::buffer(&zeros),
    ]
}

/// The values of `bytes` as PyTorch reads `float4_e2m1fn_x2`, element 0
/// from the low nibble, decoded on the host.
pub fn reference(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().flat_map(|&b| [F4E2M1FN(b & 0xf).to_f32(), F4E2M1FN(b >> 4).to_f32()]).collect()
}

/// Simulate the kernel on `bytes` and check both outputs against the host.
pub fn run(decode: Decode, bytes: &[u8]) {
    let mut args = args(bytes);
    build(decode).simulate_all_orders(launch(), &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<u16> = reference(bytes).iter().map(|&x| F16::from_f32(x).0).collect();
    for out in [2, 3] {
        let got: Vec<u16> = args[out].to_vec::<F16>().iter().map(|h| h.0).collect();
        assert_eq!(got, want, "{decode:?}: output {out}");
    }
}

fn main() {
    let bytes = random_bytes(1);
    let all: Vec<u8> = (0..=255).cycle().take(ROWS * BYTES_PER_ROW).collect();
    for decode in [Decode::Integer, Decode::Cvt] {
        run(decode, &bytes);
        run(decode, &all);
        println!("fp4_x2_tensor ({decode:?}): the Float4E2M1FN and recast Uint8 tensors decode correctly");
    }
    println!("row 0 of the example's data: {:?}", &reference(&bytes)[..2 * BYTES_PER_ROW]);
    if let Some(tc) = Toolchain::discover() {
        for (decode, arch) in [(Decode::Integer, Arch::Sm80), (Decode::Cvt, Arch::Sm100a)] {
            match build(decode).compile_ptx(&tc, arch) {
                Ok(ptx) => {
                    let cvt = ptx.lines().filter(|l| l.contains("cvt.rn.f16x2.e2m1x2")).count();
                    println!(
                        "PTX for {decode:?} ({}): {} lines, {cvt} cvt.rn.f16x2.e2m1x2",
                        arch.chip(),
                        ptx.lines().count()
                    );
                }
                Err(e) => eprintln!("PTX compilation failed: {e}"),
            }
        }
    }
}

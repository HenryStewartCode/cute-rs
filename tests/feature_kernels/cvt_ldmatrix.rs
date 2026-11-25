// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/copy_traits_sm100.hpp (the thread-value layouts of
//   SM100_U8x8_LDSM_T and SM100_U8x16_LDSM_T that the LdMatrix16x8x8b check
//   reads its expected values from),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The fast int4 conversions (to BFloat16 and to E4M3 bytes) and the 8-bit
//! `ldmatrix` shapes (`.m16n16`, `.m8n16`, unpacking padded 4- and 6-bit
//! rows), each held to plain Rust.

use cute::prelude::*;

use super::{Case, T, launch};

/// The int4 value `(t + k) mod 16` (as a signed nibble): over 16
/// consecutive threads every value reaches every element position.
fn sweep(t: usize, k: usize) -> i8 {
    (((t + k) as u8 & 0xf) << 4) as i8 >> 4
}

/// BFloat16 bits of an int4 value (exact).
fn bf(v: i8) -> u16 {
    BF16::from_f32(v as f32).0
}

/// Columns of [`int4_fast_conversions`]: per word (the sweep and the input) x8,
/// x4, x2 and eight scalars; then the intrinsic over 17 values at every
/// length.
const I4_COLS: usize = 2 * (8 + 4 + 2 + 8) + 17 * 18 / 2;

/// The int4 values of thread `t`: the sweep, its input row, and the sweep's
/// first value again (17 for the intrinsic).
fn int4_values(t: usize, src: &[u32]) -> Vec<i8> {
    let mut v: Vec<i8> = (0..8).map(|k| sweep(t, k)).collect();
    v.extend((0..8).map(|k| (((src[t] >> (4 * k)) as u8) << 4) as i8 >> 4));
    v.push(sweep(t, 0));
    v
}

/// The int4 conversions on their default paths, which need only sm_80
/// (`lop3`, `prmt`, `fma.rn.bf16x2`, `prmt` for FP8): `cvt_i4x8_to_bf16x8`,
/// `cvt_i4x4_to_bf16x4`, `cvt_i4x2_to_bf16x2` and `cvt_i4_bf16` on a sweep
/// that gives every element position all 16 values and on random input,
/// `cvt_i4_bf16_intrinsic` at every length from 1 to 17, and
/// `cvt_i4x8_to_mxf8x8` / `cvt_i4_mxf8_intrinsic` to E4M3 (and to Int8
/// bytes). Every int4 value is exact in BFloat16 and E4M3, so the results
/// are bit-exact.
pub fn int4_fast_conversions() -> Case {
    let kernel = Kernel::builder("int4_fast_conversions").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(T));
        let out = param_tensor::<BF16>("out", Layout::row_major((T, I4_COLS)));
        let fp8 = param_tensor::<F8E4M3FN>("fp8", Layout::row_major((T, 32)));
        let bytes = param_tensor::<i8>("bytes", Layout::row_major((T, 8)));
        let t = thread_idx().x;
        let swept: [Int4; 8] = std::array::from_fn(|k| ((t + k as i32) & 0xf).to::<I4>());
        let input: [Int4; 8] = unpack(src.load(t));
        let mut results: Vec<BFloat16> = Vec::new();
        for word in [swept, input] {
            results.extend(cvt_i4x8_to_bf16x8(word, false));
            results.extend(cvt_i4x4_to_bf16x4([word[0], word[1], word[2], word[3]], false));
            let (lo, hi): (BFloat16, BFloat16) = cvt_i4x2_to_bf16x2([word[0], word[1]], false);
            results.extend([lo, hi]);
            results.extend(word.map(cvt_i4_bf16));
        }
        let mut seventeen: Vec<Int4> = swept.to_vec();
        seventeen.extend(input);
        seventeen.push(swept[0]);
        for len in 1..=17 {
            results.extend(cvt_i4_bf16_intrinsic(&seventeen, len, false));
        }
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j as i32), v);
        }
        let mut e4m3: Vec<Value<F8E4M3FN>> = cvt_i4x8_to_mxf8x8::<F8E4M3FN>(swept).to_vec();
        e4m3.extend(cvt_i4x8_to_mxf8x8::<F8E4M3FN>(input));
        e4m3.extend(cvt_i4_mxf8_intrinsic::<F8E4M3FN>(&seventeen, 16));
        for (j, v) in e4m3.into_iter().enumerate() {
            fp8.store((t, j as i32), v);
        }
        for (j, v) in cvt_i4x8_to_mxf8x8::<i8>(input).into_iter().enumerate() {
            bytes.store((t, j as i32), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let src = args[0].to_vec::<u32>();
        let (out, fp8, bytes) = (args[1].to_vec::<BF16>(), args[2].to_vec::<F8E4M3FN>(), args[3].to_vec::<i8>());
        for t in 0..T {
            let v = int4_values(t, &src);
            let mut want: Vec<u16> = Vec::new();
            for word in [&v[..8], &v[8..16]] {
                want.extend(word.iter().map(|&x| bf(x)));
                want.extend(word[..4].iter().map(|&x| bf(x)));
                want.extend(word[..2].iter().map(|&x| bf(x)));
                want.extend(word.iter().map(|&x| bf(x)));
            }
            for len in 1..=17 {
                want.extend(v[..len].iter().map(|&x| bf(x)));
            }
            let got: Vec<u16> = out[t * I4_COLS..(t + 1) * I4_COLS].iter().map(|x| x.0).collect();
            assert_eq!(got, want, "thread {t}: BFloat16");
            let e4m3 = |x: i8| F8E4M3FN::from_f32(x as f32).0;
            let mut want8: Vec<u8> = v[..16].iter().map(|&x| e4m3(x)).collect();
            want8.extend(v[..16].iter().map(|&x| e4m3(x)));
            let got8: Vec<u8> = fp8[t * 32..(t + 1) * 32].iter().map(|x| x.0).collect();
            assert_eq!(got8, want8, "thread {t}: E4M3");
            let raw: Vec<u8> = bytes[t * 8..(t + 1) * 8].iter().map(|&x| x as u8).collect();
            assert_eq!(raw, want8[8..16].to_vec(), "thread {t}: E4M3 bytes as Int8");
        }
    }
    Case { name: "int4_fast_conversions", kernel, cfg: launch(1), small_only: false, check }
}

/// The shuffled order CuTe DSL's `with_shuffle` path takes: `(0, 2, 1, 3,
/// 4, 6, 5, 7)` per group of eight, `(0, 2, 1, 3)` for a trailing four.
fn shuffled<V: Copy>(v: &[V]) -> Vec<V> {
    let mut out = Vec::with_capacity(v.len());
    let mut rest = v;
    while rest.len() >= 8 {
        out.extend([0, 2, 1, 3, 4, 6, 5, 7].map(|k| rest[k]));
        rest = &rest[8..];
    }
    if rest.len() >= 4 {
        out.extend([0, 2, 1, 3].map(|k| rest[k]));
        rest = &rest[4..];
    }
    out.extend_from_slice(rest);
    out
}

/// Columns of [`int4_bf16_shuffle`]: x8, x4, x2, then the intrinsic over 13
/// values at every length.
const SHUFFLE_COLS: usize = 8 + 4 + 2 + 13 * 14 / 2;

/// The `with_shuffle` path of the int4-to-BFloat16 conversions (the scaled
/// `.s2f6x2` cvt with scale `0x81`): x8, x4 and x2 on the sweep and the
/// intrinsic at every length from 1 to 13, each fed the shuffled order and
/// giving natural order. Like [`super::cvt_i8_bf16_s2f6x2`] it needs PTX 9.1
/// on sm_100a-class targets: in no kernel list, but simulated and checked as
/// MLIR (and as PTX when the `mlir-opt` is LLVM 23).
pub fn int4_bf16_shuffle() -> Case {
    let kernel = Kernel::builder("int4_bf16_shuffle").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<BF16>("out", Layout::row_major((T, SHUFFLE_COLS)));
        let t = thread_idx().x;
        let values: Vec<Int4> = (0..13).map(|k| ((t + k) & 0xf).to::<I4>()).collect();
        let order = shuffled(&values);
        let mut results: Vec<BFloat16> = Vec::new();
        results.extend(cvt_i4x8_to_bf16x8(std::array::from_fn(|k| order[k]), true));
        let four = shuffled(&values[..4]);
        results.extend(cvt_i4x4_to_bf16x4([four[0], four[1], four[2], four[3]], true));
        let (lo, hi): (BFloat16, BFloat16) = cvt_i4x2_to_bf16x2([values[0], values[1]], true);
        results.extend([lo, hi]);
        for len in 1..=13 {
            results.extend(cvt_i4_bf16_intrinsic(&shuffled(&values[..len]), len, true));
        }
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j as i32), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<BF16>();
        for t in 0..T {
            let v: Vec<i8> = (0..13).map(|k| sweep(t, k)).collect();
            let mut want: Vec<u16> = v[..8].iter().map(|&x| bf(x)).collect();
            want.extend(v[..4].iter().map(|&x| bf(x)));
            want.extend(v[..2].iter().map(|&x| bf(x)));
            for len in 1..=13 {
                want.extend(v[..len].iter().map(|&x| bf(x)));
            }
            let got: Vec<u16> = out[t * SHUFFLE_COLS..(t + 1) * SHUFFLE_COLS].iter().map(|x| x.0).collect();
            assert_eq!(got, want, "thread {t}");
        }
    }
    Case { name: "int4_bf16_shuffle", kernel, cfg: launch(1), small_only: false, check }
}

/// One 8-bit `ldmatrix` atom the [`ldmatrix_b8_forms`] kernel applies.
#[derive(Clone, Copy)]
enum Form {
    /// `LdMatrix8x16x8bOp(num, unpack_bits)` (`.m8n16`).
    M8n16(usize, u32),
    /// `LdMatrix16x16x8bOp(num, transpose=True, unpack_bits)` (`.m16n16`).
    M16n16(usize, Option<u32>),
    /// `LdMatrix16x8x8bOp(num, transpose=True, unpack_bits)`.
    M16n8(usize, Option<u32>),
}

impl Form {
    fn op(self) -> CopyOp {
        match self {
            Form::M8n16(num, bits) => CopyOp::LdMatrix8x16x8b { num, unpack_bits: Some(bits) },
            Form::M16n16(num, unpack_bits) => CopyOp::LdMatrix16x16x8b { num, trans: true, unpack_bits },
            Form::M16n8(num, unpack_bits) => CopyOp::LdMatrix16x8x8b { num, trans: true, unpack_bits },
        }
    }

    /// The 16-byte rows the atom reads.
    fn rows(self) -> usize {
        match self {
            Form::M8n16(num, _) => 8 * num,
            Form::M16n16(num, _) => 16 * num,
            Form::M16n8(num, _) => 8 * num,
        }
    }

    /// The values (bytes) each lane receives.
    fn values(self) -> usize {
        match self {
            Form::M8n16(num, _) => 4 * num,
            Form::M16n16(num, _) => 8 * num,
            Form::M16n8(num, _) => 4 * num,
        }
    }

    fn unpack(self) -> Option<u32> {
        match self {
            Form::M8n16(_, bits) => Some(bits),
            Form::M16n16(_, bits) | Form::M16n8(_, bits) => bits,
        }
    }
}

/// The forms [`ldmatrix_b8_forms`] covers: every shape, every
/// `num_matrices` and every unpacking at least once.
const FORMS: [Form; 12] = [
    Form::M8n16(1, 4),
    Form::M8n16(2, 6),
    Form::M8n16(4, 4),
    Form::M8n16(4, 6),
    Form::M16n16(1, None),
    Form::M16n16(2, None),
    Form::M16n16(1, Some(6)),
    Form::M16n16(2, Some(4)),
    Form::M16n8(2, None),
    Form::M16n8(4, None),
    Form::M16n8(4, Some(4)),
    Form::M16n8(2, Some(6)),
];

/// The shared-memory image: 64 rows of 16 bytes.
const IMAGE_ROWS: usize = 64;

/// Bytes of the outputs of [`ldmatrix_b8_forms`]: each form's values, then
/// the two-warp tiled copy's 16.
const LDSM_COLS: usize = 4 + 8 + 16 + 16 + 8 + 16 + 8 + 16 + 8 + 16 + 16 + 8 + 16;

/// Row `r` of the image as 16 8-bit containers, written from the PTX ISA's
/// tables independently of the simulator: the bytes themselves, or 16
/// 4-bit values packed from the least significant bit of the first 8 bytes
/// (each in bits 0 to 3 of its container, as a B200 loads them) or 16 6-bit values of the
/// first 12 (bits 0 to 5); the rest is padding.
fn expanded(image: &[u8], r: usize, unpack: Option<u32>) -> [u8; 16] {
    let row = &image[16 * r..16 * r + 16];
    let bits = |e: usize, width: usize| -> u8 {
        (0..width).map(|b| ((row[(width * e + b) / 8] >> ((width * e + b) % 8)) & 1) << b).sum()
    };
    std::array::from_fn(|e| match unpack {
        None => row[e],
        Some(4) => bits(e, 4),
        Some(_) => bits(e, 6),
    })
}

/// What lane `l` receives as value `v` of `form` (register `v / 4`, byte
/// `v % 4`), from the PTX ISA's fragment figures for `.m8n16` and `.m16n16`
/// (of the transposed matrix) and, for `LdMatrix16x8x8bOp`, CuTe C++'s
/// thread-value layouts of `SM100_U8x8_LDSM_T` / `SM100_U8x16_LDSM_T` (in
/// bits: `((4, 8), (8, 2, 2, n)) : ((256, 8), (1, 128, 64, 1024))`).
fn ldsm_expected(form: Form, image: &[u8], l: usize, v: usize) -> u8 {
    let (r, j) = (v / 4, v % 4);
    let (row, byte) = match form {
        Form::M8n16(..) => (8 * r + l / 4, 4 * (l % 4) + j),
        Form::M16n16(..) => (16 * (r / 2) + 4 * (l % 4) + j, l / 4 + 8 * (r % 2)),
        Form::M16n8(..) => {
            let (v1, v2, v3) = (v % 2, (v / 2) % 2, v / 4);
            let bit = 256 * (l % 4) + 8 * (l / 4) + 128 * v1 + 64 * v2 + 1024 * v3;
            (bit / 8 / 16, bit / 8 % 16)
        }
    };
    expanded(image, row, form.unpack())[byte]
}

/// Every 8-bit `ldmatrix` form on shared memory filled from a random image
/// (padding bytes included): one atom per form applied by a tiled copy of
/// its own thread-value layout, and `.m8n16.x4` with 4-bit unpacking tiled
/// over two warps with `make_tiled_copy`. Each lane's values are held to the
/// PTX ISA's fragment tables. Needs sm_100a or sm_120a.
pub fn ldmatrix_b8_forms() -> Case {
    let kernel = Kernel::builder("ldmatrix_b8_forms").block_dim([T as u32, 1, 1]).build(|| {
        let image = param_tensor::<u8>("image", Layout::col_major(16 * IMAGE_ROWS));
        let out = param_tensor::<u8>("out", Layout::row_major((T, LDSM_COLS)));
        let t = thread_idx().x;
        let lane = t % 32;
        let mut col = 0i32;
        let mut emit = |frag: &Tensor<u8>| {
            for v in 0..frag.size() {
                out.store((t, col + v as i32), frag.load(v));
            }
            col += frag.size() as i32;
        };
        for form in FORMS {
            let rows = form.rows();
            // (byte, row): offset 16 row + byte, the atoms' element space.
            let smem = make_smem_tensor::<u8>(Layout::col_major((16, rows)));
            for i in 0..(16 * rows).div_ceil(T) {
                let k = t + (i * T) as i32;
                if_(k.lt((16 * rows) as i32), || smem.store((k % 16, k / 16), image.load(k)));
            }
            sync_threads();
            let atom = make_copy_atom::<u8>(form.op());
            let tiled = make_tiled_copy(&atom, atom.layout_ref_tv(), (16, rows));
            let frag = make_fragment::<u8>(form.values());
            copy(&tiled, &tiled.get_slice(lane).partition_s(&smem), &tiled.retile(&frag));
            emit(&frag);
        }
        // .m8n16.x4 unpacking 4-bit rows, over two warps: warp w takes rows
        // 32 w to 32 w + 31.
        let smem = make_smem_tensor::<u8>(Layout::col_major((16, IMAGE_ROWS)));
        for i in 0..16 {
            let k = t + (i * T) as i32;
            smem.store((k % 16, k / 16), image.load(k));
        }
        sync_threads();
        let atom = make_copy_atom::<u8>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: Some(4) });
        let tv = Layout::new(((32, 2), (4, 4)), ((4, 512), (1, 128)));
        let tiled = make_tiled_copy(&atom, &tv, (16, IMAGE_ROWS));
        let frag = make_fragment::<u8>(16);
        copy(&tiled, &tiled.get_slice(t).partition_s(&smem), &tiled.retile(&frag));
        emit(&frag);
    });
    fn check(args: &[KernelArg]) {
        let (image, out) = (args[0].to_vec::<u8>(), args[1].to_vec::<u8>());
        for t in 0..T {
            let (l, w) = (t % 32, t / 32);
            let row = &out[t * LDSM_COLS..(t + 1) * LDSM_COLS];
            let mut col = 0;
            for form in FORMS {
                for v in 0..form.values() {
                    assert_eq!(row[col + v], ldsm_expected(form, &image, l, v), "thread {t} {form:?} value {v}");
                }
                col += form.values();
            }
            let tiled = &image[512 * w..];
            for v in 0..16 {
                let want = ldsm_expected(Form::M8n16(4, 4), tiled, l, v);
                assert_eq!(row[col + v], want, "thread {t}: two-warp tiled copy value {v}");
            }
        }
    }
    Case { name: "ldmatrix_b8_forms", kernel, cfg: launch(1), small_only: false, check }
}

impl std::fmt::Debug for Form {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Form::M8n16(n, b) => write!(f, "LdMatrix8x16x8b(num={n}, unpack_bits={b})"),
            Form::M16n16(n, b) => write!(f, "LdMatrix16x16x8b(num={n}, unpack_bits={b:?})"),
            Form::M16n8(n, b) => write!(f, "LdMatrix16x8x8b(num={n}, unpack_bits={b:?})"),
        }
    }
}

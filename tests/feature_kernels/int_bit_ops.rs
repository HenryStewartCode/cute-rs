// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! The integer and bit-field instructions (bfe, bfi, lop3, shf, prmt, multiplies, counts) with their reference implementations.

use super::*;
// References for the integer and bit instructions, written from the PTX
// ISA's definitions (bit by bit where it gives pseudo-code), independently
// of the simulator.

/// An odd multiplier that spreads test data over every bit of its type (as
/// a wrapping product, a bijection). The generated inputs of the 64-bit
/// types are below 2^48 and never negative, so the integer and bit kernels
/// multiply their data by it to reach the high bits and the sign.
pub(super) const SPREAD: u64 = 0x9e37_79b9_7f4a_7c15;

/// `bfe`: the PTX ISA's pseudo-code.
pub fn bfe_ref(a: u64, b: i32, c: i32, width: u32, signed: bool) -> u64 {
    let msb = width - 1;
    let (pos, len) = (b as u32 & 0xff, c as u32 & 0xff);
    let sbit = if !signed || len == 0 { 0 } else { a >> (pos + len - 1).min(msb) & 1 };
    (0..=msb).fold(0, |d, i| d | (if i < len && pos + i <= msb { a >> (pos + i) & 1 } else { sbit }) << i)
}

/// `bfi`: the PTX ISA's pseudo-code (`a` is the field, `b` the base).
pub fn bfi_ref(a: u64, b: u64, c: i32, d: i32, width: u32) -> u64 {
    let msb = width - 1;
    let (pos, len) = (c as u32 & 0xff, d as u32 & 0xff);
    let mut f = b;
    let mut i = 0;
    while i < len && pos + i <= msb {
        f = f & !(1 << (pos + i)) | (a >> i & 1) << (pos + i);
        i += 1;
    }
    f
}

/// `lop3.b32`: each result bit looked up in the table.
pub fn lop3_ref(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    (0..32).fold(0, |d, i| {
        let index = (a >> i & 1) << 2 | (b >> i & 1) << 1 | (c >> i & 1);
        d | (lut as u32 >> index & 1) << i
    })
}

/// `shf`: the PTX ISA's formulas, where a shift by 32 gives 0.
pub fn shf_ref(a: u32, b: u32, c: u32, kind: ShfKind) -> u32 {
    let (left, clamp) = match kind {
        ShfKind::Left => (true, false),
        ShfKind::Right => (false, false),
        ShfKind::ClampLeft => (true, true),
        ShfKind::ClampRight => (false, true),
    };
    let n = if clamp { c.min(32) } else { c & 0x1f };
    let shl = |x: u32, n: u32| x.checked_shl(n).unwrap_or(0);
    let shr = |x: u32, n: u32| x.checked_shr(n).unwrap_or(0);
    if left { shl(b, n) | shr(a, 32 - n) } else { shl(b, 32 - n) | shr(a, n) }
}

/// `prmt.b32{.mode}`: the generic form from the ISA's description, the
/// modes from formulas for its table.
pub fn prmt_ref(a: u32, b: u32, c: u32, mode: PrmtMode) -> u32 {
    let byte = |k: u32| ((b as u64) << 32 | a as u64) >> (8 * k) & 0xff;
    let s = c & 3;
    let source = |i: u32| match mode {
        PrmtMode::Generic => unreachable!(),
        PrmtMode::F4e => s + i,
        PrmtMode::B4e => (s + 8 - i) % 8,
        PrmtMode::Rc8 => s,
        PrmtMode::Ecl => i.max(s),
        PrmtMode::Ecr => i.min(s),
        PrmtMode::Rc16 => (i & 1) + 2 * (s & 1),
    };
    (0..4).fold(0, |d, i| {
        let value = if mode == PrmtMode::Generic {
            let select = c >> (4 * i) & 0xf;
            let picked = byte(select & 7);
            match (select >> 3, picked >> 7) {
                (0, _) => picked,
                (_, 0) => 0,
                _ => 0xff,
            }
        } else {
            byte(source(i))
        };
        d | (value as u32) << (8 * i)
    })
}

/// The bit-field starts and lengths every thread tries on a type of
/// `width` bits, as constants: empty and full fields, fields reaching past
/// the msb or starting beyond it, fields longer than the type starting inside
/// it, and operands with bits above the low byte (which only count mod 256),
/// including a zero low byte (0x100, 0x200: a start of 0 or an empty field).
/// Constant starts and lengths are folded by `ptxas`, which on its own gets
/// some of these wrong (`bfe.s32` from beyond the msb, `bfi.b32` fields longer
/// than 31 bits).
fn bit_field_edges(width: i32) -> Vec<(i32, i32)> {
    let w = width;
    vec![
        (0, 0),
        (0, 1),
        (0, w),
        (0, 255),
        (3, 0),
        (5, 7),
        (w - 8, 8),
        (w - 4, 8),
        (w - 1, 1),
        (w - 1, 2),
        (w, 0),
        (w, 1),
        (w + 3, 0),
        (w + 8, 5),
        (200, 100),
        (255, 255),
        (1, w),
        (3, w + 8),
        (17, 100),
        (w - 2, w + 1),
        (9, 254),
        (8, w),
        (w / 2 + 3, w),
        (0x103, 0x204),
        (-1, -1),
        (0x7fff_ff07, 0x109),
        (i32::MIN, 0x7fff_ffff),
        (5, 0x100),
        (w + 8, 0x100),
        (0x100, 3),
        (0x200, w),
        (0x100, 0x100),
        (0x100 + w - 4, 8),
    ]
}

/// Constants paired with a runtime start or length in [`bit_field_case`].
const HALF_CONSTANT: [i32; 5] = [5, 40, 70, 300, 0x100];

/// Iterations of the bit-field loop in [`bit_field_case`], whose starts and
/// lengths become constants only once LLVM or `ptxas` unrolls it.
const BIT_FIELD_LOOP: i32 = 6;

/// The start, `bfe` length and `bfi` length of iteration `i` of that loop:
/// starts from 9 to 74 and `bfi` lengths from 30 to 85.
fn bit_field_loop(i: i32) -> (i32, i32, i32) {
    (i * 13 + 9, 9, i * 11 + 30)
}

/// Pairs of a runtime start and length in [`bit_field_case`]: from the data,
/// from the thread index, and two with a zero low byte on some threads.
const BIT_FIELD_RUNTIME: usize = 4;

/// The output columns of [`bit_field_case`] on a type of `width` bits.
fn bit_field_columns(width: i32) -> usize {
    2 * (BIT_FIELD_RUNTIME + bit_field_edges(width).len() + 2 * HALF_CONSTANT.len() + BIT_FIELD_LOOP as usize)
}

/// `bfe` and `bfi` on one type, on data spread over every bit (so fields
/// past the msb fill with a set sign bit as well as a clear one, and `bfi`
/// writes into values with their high bits set): starts and lengths from
/// the input data (any 32-bit value, of which only the low byte counts),
/// from the thread index (fields inside, across and beyond the type; and
/// starts or lengths of 0x100 to 0x700, whose low byte is 0, on every eighth
/// thread), the constant edge cases of [`bit_field_edges`], a constant start
/// or length with the other from the data, and a runtime loop whose starts
/// and lengths are constants once unrolled.
macro_rules! bit_field_case {
    ($name:ident, $t:ty, $u:ty) => {
        pub fn $name() -> Case {
            const W: i32 = <$t>::BITS as i32;
            let cols = bit_field_columns(W);
            let kernel =
                Kernel::builder(concat!("bit_fields_", stringify!($t))).block_dim([T as u32, 1, 1]).build(|| {
                    let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                    let ys = param_tensor::<$t>("ys", Layout::col_major(T));
                    let ps = param_tensor::<i32>("ps", Layout::col_major(T));
                    let ls = param_tensor::<i32>("ls", Layout::col_major(T));
                    let out = param_tensor::<$t>("out", Layout::row_major((T, cols)));
                    let t = thread_idx().x;
                    let (x, y) = (xs.load(t) * SPREAD as $t, ys.load(t) * SPREAD as $t);
                    let (p, l) = (ps.load(t), ls.load(t));
                    let (tp, tl) = (t * 5 % (2 * W + 6), t * 11 % (2 * W + 6));
                    // As in the check: a zero low byte when t % 8 == 0 and
                    // when t < 8 respectively.
                    let (zp, zl) = (((t >> 3) << 8) | ((t & 7) * 9), ((t & 7) << 8) | ((t >> 3) * 9));
                    let mut results = vec![bfe(x, p, l), bfi(y, x, p, l), bfe(x, tp, tl), bfi(y, x, tp, tl)];
                    results.extend([bfe(x, zp, tl), bfi(y, x, zp, tl), bfe(x, tp, zl), bfi(y, x, tp, zl)]);
                    for (start, length) in bit_field_edges(W) {
                        results.push(bfe(x, start, length));
                        results.push(bfi(y, x, start, length));
                    }
                    for c in HALF_CONSTANT {
                        results.extend([bfe(x, c, l), bfi(y, x, c, l), bfe(x, p, c), bfi(y, x, p, c)]);
                    }
                    let looped = results.len();
                    for (j, v) in results.into_iter().enumerate() {
                        out.store((t, j), v);
                    }
                    for_range(0, BIT_FIELD_LOOP, 1, |i| {
                        // As in bit_field_loop.
                        let (start, length_e, length_i) = (i * 13 + 9, 9, i * 11 + 30);
                        let column = i * 2 + looped as i32;
                        out.store((t, column), bfe(x, start, length_e));
                        out.store((t, column + 1), bfi(y, x, start, length_i));
                    });
                    assert_eq!(looped + 2 * BIT_FIELD_LOOP as usize, cols);
                });
            fn check(args: &[KernelArg]) {
                let (xs, ys) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>());
                let (ps, ls, out) = (args[2].to_vec::<i32>(), args[3].to_vec::<i32>(), args[4].to_vec::<$t>());
                let signed = <$t>::MIN != 0;
                let (w, cols) = (W as u32, bit_field_columns(W));
                let spread = |v: $t| v.wrapping_mul(SPREAD as $t) as $u as u64;
                let msb_set = (0..T).filter(|&t| spread(xs[t]) >> (w - 1) == 1).count();
                assert!(msb_set > 0 && msb_set < T, "{}: {msb_set} values with the msb set", stringify!($t));
                for t in 0..T {
                    let (x, y) = (spread(xs[t]), spread(ys[t]));
                    let ti = t as i32;
                    // (start, bfe length, bfi length) for each pair of columns.
                    let mut fields = vec![(ps[t], ls[t], ls[t])];
                    let (tp, tl) = (ti * 5 % (2 * W + 6), ti * 11 % (2 * W + 6));
                    let (zp, zl) = (((ti >> 3) << 8) | ((ti & 7) * 9), ((ti & 7) << 8) | ((ti >> 3) * 9));
                    fields.extend([(tp, tl, tl), (zp, tl, tl), (tp, zl, zl)]);
                    fields.extend(bit_field_edges(W).into_iter().map(|(p, l)| (p, l, l)));
                    for c in HALF_CONSTANT {
                        fields.extend([(c, ls[t], ls[t]), (ps[t], c, c)]);
                    }
                    fields.extend((0..BIT_FIELD_LOOP).map(bit_field_loop));
                    assert_eq!(2 * fields.len(), cols);
                    for (k, (p, le, li)) in fields.into_iter().enumerate() {
                        let got_e = out[t * cols + 2 * k] as $u as u64;
                        let got_i = out[t * cols + 2 * k + 1] as $u as u64;
                        let what = format!("{}: x = {x:#x}, start {p}", stringify!($t));
                        assert_eq!(got_e, bfe_ref(x, p, le, w, signed), "bfe, {what}, length {le}");
                        assert_eq!(got_i, bfi_ref(y, x, p, li, w), "bfi, {what}, length {li}, replacement {y:#x}");
                    }
                }
            }
            Case { name: concat!("bit_fields_", stringify!($t)), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

bit_field_case!(bit_fields_i32, i32, u32);
bit_field_case!(bit_fields_u32, u32, u32);
bit_field_case!(bit_fields_i64, i64, u64);
bit_field_case!(bit_fields_u64, u64, u64);

/// The `lop3` tables every thread applies: constants, the PTX ISA's
/// examples, xor, majority and select.
const LOP3_TABLES: [u8; 9] = [0x00, 0xff, 0x80, 0xfe, 0x40, 0x1a, 0x96, 0xe8, 0xca];

/// Constant funnel-shift amounts: around 0 and 32, and amounts that wrap
/// and clamp differently (`-1` is 2^32 - 1 unsigned).
const SHIFTS: [i32; 8] = [0, 1, 13, 31, 32, 33, 64, -1];

/// Constant `prmt` selectors: identity, the high word, a byte reversal, an
/// interleave, and sign replication from each byte.
const SELECTORS: [u32; 8] = [0x3210, 0x7654, 0x0123, 0x5140, 0x8888, 0xba98, 0xfedc, 0x7f80];

const SHF_KINDS: [ShfKind; 4] = [ShfKind::Left, ShfKind::Right, ShfKind::ClampLeft, ShfKind::ClampRight];

const PRMT_MODES: [PrmtMode; 6] =
    [PrmtMode::F4e, PrmtMode::B4e, PrmtMode::Rc8, PrmtMode::Ecl, PrmtMode::Ecr, PrmtMode::Rc16];

/// `lop3`, `shf` (rotates included) and `prmt` (every mode) on Uint32 and
/// Int32, with runtime and constant tables, shift amounts and selectors.
pub fn logic_and_shifts() -> Case {
    const COLS: usize = 9 + 1 + 4 + 4 * 8 + 3 + 1 + 8 + 1 + 6 + 6;
    let kernel = Kernel::builder("logic_and_shifts").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let zs = param_tensor::<u32>("zs", Layout::col_major(T));
        let ss = param_tensor::<i32>("ss", Layout::col_major(T));
        let ws = param_tensor::<i32>("ws", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let (x, y, z, s, w) = (xs.load(t), ys.load(t), zs.load(t), ss.load(t), ws.load(t));
        let mut results: Vec<Uint32> = LOP3_TABLES.iter().map(|&lut| lop3(x, y, z, lut)).collect();
        results.push(lop3(w, w ^ 0x5a5a_5a5a, s, 0xd8).bitcast());
        results.extend(SHF_KINDS.map(|kind| shf(x, y, s, kind)));
        for kind in SHF_KINDS {
            results.extend(SHIFTS.map(|n| shf(x, y, n, kind)));
        }
        results.push(shf(x, x, s, ShfKind::Left));
        results.push(shf(x, x, s, ShfKind::Right));
        results.push(shf(w, s, w, ShfKind::ClampRight).bitcast());
        results.push(prmt(x, y, z));
        results.extend(SELECTORS.map(|c| prmt(x, y, c)));
        results.push(prmt(w, s, w).bitcast());
        results.extend(PRMT_MODES.map(|mode| prmt_with(x, y, z, mode)));
        results.extend(PRMT_MODES.map(|mode| prmt_with(x, y, 3u32, mode)));
        assert_eq!(results.len(), COLS);
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, zs) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        let (ss, ws, out) = (args[3].to_vec::<i32>(), args[4].to_vec::<i32>(), args[5].to_vec::<u32>());
        for t in 0..T {
            let (x, y, z, s, w) = (xs[t], ys[t], zs[t], ss[t] as u32, ws[t] as u32);
            let mut want: Vec<u32> = LOP3_TABLES.iter().map(|&lut| lop3_ref(x, y, z, lut)).collect();
            want.push(lop3_ref(w, w ^ 0x5a5a_5a5a, s, 0xd8));
            want.extend(SHF_KINDS.map(|kind| shf_ref(x, y, s, kind)));
            for kind in SHF_KINDS {
                want.extend(SHIFTS.map(|n| shf_ref(x, y, n as u32, kind)));
            }
            want.push(x.rotate_left(s % 32));
            want.push(x.rotate_right(s % 32));
            want.push(shf_ref(w, s, w, ShfKind::ClampRight));
            want.push(prmt_ref(x, y, z, PrmtMode::Generic));
            want.extend(SELECTORS.map(|c| prmt_ref(x, y, c, PrmtMode::Generic)));
            want.push(prmt_ref(w, s, w, PrmtMode::Generic));
            want.extend(PRMT_MODES.map(|mode| prmt_ref(x, y, z, mode)));
            want.extend(PRMT_MODES.map(|mode| prmt_ref(x, y, 3, mode)));
            for (j, w) in want.into_iter().enumerate() {
                let got = out[t * COLS + j];
                assert_eq!(got, w, "column {j}, thread {t}: x = {x:#x}, y = {y:#x}, z = {z:#x}, s = {s:#x}");
            }
        }
    }
    Case { name: "logic_and_shifts", kernel, cfg: launch(1), small_only: false, check }
}

/// Constant operands of the 64-bit `mul_hi` in [`int_multiplies`], each
/// multiplied by the data: negative values (with high bits set and clear),
/// the extremes and -1; and unsigned values with bit 63 set.
const MUL_HI_I64: [i64; 5] = [i64::MIN, i64::MIN + 3, -0x1234_5678_9abc_def0, -1, i64::MAX];
const MUL_HI_U64: [u64; 3] = [u64::MAX, u64::MAX - 5, 0x8000_0000_0000_0001];

/// `mul_hi` on the four 32- and 64-bit types (the 64-bit data spread over
/// all 64 bits, so negative and with all 128 bits of the product in use, and
/// times negative and high-bit constants), `mul_wide` on the four 16- and
/// 32-bit types, `mul24` and `mad24` (both halves, both signednesses, with
/// garbage in bits 31:24 and bit 23 set at random) and the saturating
/// add and subtract, on the input data and on edge constants.
pub fn int_multiplies() -> Case {
    const N64: usize = 4 + MUL_HI_I64.len() + MUL_HI_U64.len() + 2;
    const N32: usize = 2 + 4 * 2 + 2 + 8 + 6;
    let kernel = Kernel::builder("int_multiplies").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let ls = param_tensor::<i64>("ls", Layout::col_major(T));
        let ms = param_tensor::<i64>("ms", Layout::col_major(T));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let t = thread_idx().x;
        let (x, y, z) = (xs.load(t), ys.load(t), zs.load(t));
        let (l, m) = (ls.load(t) * SPREAD as i64, ms.load(t) * SPREAD as i64);
        let (ux, uy, uz) = (x.bitcast::<u32>(), y.bitcast::<u32>(), z.bitcast::<u32>());
        let (ul, um) = (l.bitcast::<u64>(), m.bitcast::<u64>());
        let mut wide = vec![mul_hi(l, m).bitcast::<u64>(), mul_hi(ul, um), mul_wide(x, y).bitcast(), mul_wide(ux, uy)];
        wide.extend(MUL_HI_I64.map(|c| mul_hi(l, c).bitcast::<u64>()));
        wide.extend(MUL_HI_U64.map(|c| mul_hi(ul, c)));
        wide.push(mul_hi(Value::constant(i64::MIN + 3), -0x1234_5678_9abc_def0i64).bitcast());
        wide.push(mul_hi(Value::constant(u64::MAX - 5), 0x8000_0000_0000_0001u64));
        assert_eq!(wide.len(), N64);
        for (j, v) in wide.into_iter().enumerate() {
            out64.store((t, j), v);
        }
        let (hx, hy) = (x.to::<i16>(), y.to::<i16>());
        let mut narrow = vec![mul_hi(x, y).bitcast::<u32>(), mul_hi(ux, uy)];
        for hi in [false, true] {
            narrow.push(mul24(x, y, hi).bitcast());
            narrow.push(mul24(ux, uy, hi));
            narrow.push(mad24(x, y, z, hi).bitcast());
            narrow.push(mad24(ux, uy, uz, hi));
        }
        narrow.push(mul_wide(hx, hy).bitcast());
        narrow.push(mul_wide(hx.bitcast::<u16>(), hy.bitcast::<u16>()));
        for c in [i32::MAX, i32::MIN, -1, 1] {
            narrow.push(add_sat_int(x, c).bitcast());
            narrow.push(sub_sat_int(x, c).bitcast());
        }
        narrow.push(add_sat_int(x, y).bitcast());
        narrow.push(sub_sat_int(x, y).bitcast());
        // The sign extension from bit 23 and the 47:16 slice, on constants.
        narrow.push(mul24(0x00ff_ffffu32, 0x00ff_ffffu32, true));
        narrow.push(mul24(0x0080_0000i32, x, true).bitcast());
        narrow.push(mul24(0x0080_0000i32, x, false).bitcast());
        narrow.push(mul24(-0x0080_0000i32, -1, false).bitcast());
        assert_eq!(narrow.len(), N32);
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let (ls, ms) = (args[3].to_vec::<i64>(), args[4].to_vec::<i64>());
        let (out64, out32) = (args[5].to_vec::<u64>(), args[6].to_vec::<u32>());
        // 24-bit operands, sign-extended from bit 23 or not, and the
        // product's slices.
        let s24 = |v: i32| ((v as u32 & 0xff_ffff) ^ 0x80_0000) as i64 - 0x80_0000;
        let u24 = |v: i32| (v as u32 & 0xff_ffff) as i64;
        let slice = |p: i64, hi: bool| if hi { (p >> 16) as u32 } else { p as u32 };
        // The high halves of 128-bit products.
        let hi_s = |a: i64, b: i64| ((a as i128 * b as i128) >> 64) as u64;
        let hi_u = |a: u64, b: u64| ((a as u128 * b as u128) >> 64) as u64;
        let spread = |v: i64| v.wrapping_mul(SPREAD as i64);
        // The 64-bit operands come in every combination of signs.
        let signs: std::collections::HashSet<_> = (0..T).map(|t| (spread(ls[t]) < 0, spread(ms[t]) < 0)).collect();
        assert_eq!(signs.len(), 4, "64-bit mul_hi operands lack a combination of signs: {signs:?}");
        for t in 0..T {
            let (x, y, z, l, m) = (xs[t], ys[t], zs[t], spread(ls[t]), spread(ms[t]));
            let mut want64 = vec![
                hi_s(l, m),
                hi_u(l as u64, m as u64),
                (x as i64 * y as i64) as u64,
                x as u32 as u64 * y as u32 as u64,
            ];
            want64.extend(MUL_HI_I64.map(|c| hi_s(l, c)));
            want64.extend(MUL_HI_U64.map(|c| hi_u(l as u64, c)));
            want64.extend([hi_s(i64::MIN + 3, -0x1234_5678_9abc_def0), hi_u(u64::MAX - 5, 0x8000_0000_0000_0001)]);
            assert_eq!(out64[t * N64..(t + 1) * N64], want64[..], "64-bit products, thread {t}: {l:#x} {m:#x} {x} {y}");
            let mut want =
                vec![((x as i64 * y as i64) >> 32) as u32, ((x as u32 as u64 * y as u32 as u64) >> 32) as u32];
            for hi in [false, true] {
                want.push(slice(s24(x) * s24(y), hi));
                want.push(slice(u24(x) * u24(y), hi));
                want.push(slice(s24(x) * s24(y), hi).wrapping_add(z as u32));
                want.push(slice(u24(x) * u24(y), hi).wrapping_add(z as u32));
            }
            let (hx, hy) = (x as i16, y as i16);
            want.push((hx as i32 * hy as i32) as u32);
            want.push(hx as u16 as u32 * hy as u16 as u32);
            let clamp = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32;
            for c in [i32::MAX, i32::MIN, -1, 1] {
                want.push(clamp(x as i64 + c as i64));
                want.push(clamp(x as i64 - c as i64));
            }
            want.push(clamp(x as i64 + y as i64));
            want.push(clamp(x as i64 - y as i64));
            want.push(0xffff_fe00);
            want.push(slice(-0x80_0000 * s24(x), true));
            want.push(slice(-0x80_0000 * s24(x), false));
            want.push(slice(-0x80_0000 * -1, false));
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "32-bit products, thread {t}: {x:#x} {y:#x} {z:#x}");
        }
    }
    Case { name: "int_multiplies", kernel, cfg: launch(1), small_only: false, check }
}

/// `bfind` from the PTX ISA's pseudo-code: the highest bit of `v` (a
/// `width`-bit value) unlike the sign bit when `signed`, else the highest 1,
/// or 0xFFFFFFFF.
pub fn bfind_ref(v: u64, width: u32, signed: bool) -> u32 {
    let msb = width - 1;
    let v = if signed && v >> msb & 1 == 1 { !v } else { v };
    (0..=msb).rev().find(|&i| v >> i & 1 == 1).unwrap_or(u32::MAX)
}

/// Constant operands of [`bit_counts`]: zero, one, all ones, and the
/// extremes of the signed types.
const BIT_COUNT_CONSTANTS: [u64; 5] = [0, 1, u64::MAX, 0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff];

/// CuTe DSL's names for the bit counts (`clz`, `brev`, `popc`, `bfind`,
/// `log2_of_pow2_int`) on the four 32- and 64-bit types, on the input data
/// (the 64-bit data spread over all 64 bits on odd threads) and on
/// constants, and `globaltimer_lo` (whose two reads must be close together
/// and agree with `globaltimer`).
pub fn bit_counts() -> Case {
    const N32: usize = 12 + 4 + 10 * BIT_COUNT_CONSTANTS.len() + 3 + 2;
    const N64: usize = 4 + 4 * BIT_COUNT_CONSTANTS.len();
    let kernel = Kernel::builder("bit_counts").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let t = thread_idx().x;
        let (x, l) = (xs.load(t), ls.load(t));
        let l = select((t & 1).eq(1), l * SPREAD, l);
        let (sx, sl) = (x.bitcast::<i32>(), l.bitcast::<i64>());
        let mut narrow: Vec<Uint32> = vec![
            clz(x),
            clz(sx),
            clz(l),
            clz(sl),
            brev(x),
            brev(sx).bitcast(),
            popc(x),
            popc(sx).bitcast(),
            bfind(x),
            bfind(sx),
            bfind(l),
            bfind(sl),
        ];
        narrow.extend([popc(x.to::<u8>()).to(), popc(sx.to::<i16>()).to(), bfind(x >> 7u32), bfind(sx >> 9i32)]);
        for c in BIT_COUNT_CONSTANTS {
            narrow.extend([clz(c as u32), clz(c as i32), bfind(c as u32), bfind(c as i32), brev(c as u32)]);
            narrow.extend([clz(c), clz(c as i64), bfind(c), bfind(c as i64), popc(c as i32).bitcast()]);
        }
        let power = Value::constant(1i32) << (sx & 31);
        narrow.extend([log2_of_pow2_int(sx), log2_of_pow2_int(power), log2_of_pow2_int(0)].map(|v| v.bitcast()));
        // Two reads of the low timer word, and the whole timer between them:
        // within a second of each other, in whichever order they ran.
        let before = globaltimer_lo().bitcast::<u32>();
        let whole = globaltimer();
        let after = globaltimer_lo().bitcast::<u32>();
        let second = 1_000_000_000u32;
        let close = |a: Uint32, b: Uint32| select((b - a).lt(second) | (a - b).lt(second), 1u32, 0u32);
        narrow.extend([close(before, after), close(before, whole.to::<u32>())]);
        assert_eq!(narrow.len(), N32);
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
        let mut wide = vec![brev(l), brev(sl).bitcast(), popc(l), popc(sl).bitcast()];
        for c in BIT_COUNT_CONSTANTS {
            wide.extend([brev(c), brev(c as i64).bitcast(), popc(c), popc(c as i64).bitcast()]);
        }
        assert_eq!(wide.len(), N64);
        for (j, v) in wide.into_iter().enumerate() {
            out64.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ls) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>());
        let (out32, out64) = (args[2].to_vec::<u32>(), args[3].to_vec::<u64>());
        for t in 0..T {
            let (x, l) = (xs[t], if t % 2 == 1 { ls[t].wrapping_mul(SPREAD) } else { ls[t] });
            let mut want = vec![
                x.leading_zeros(),
                x.leading_zeros(),
                l.leading_zeros(),
                l.leading_zeros(),
                x.reverse_bits(),
                x.reverse_bits(),
                x.count_ones(),
                x.count_ones(),
                bfind_ref(x as u64, 32, false),
                bfind_ref(x as u64, 32, true),
                bfind_ref(l, 64, false),
                bfind_ref(l, 64, true),
            ];
            want.extend([
                (x as u8).count_ones(),
                (x as u16).count_ones(),
                bfind_ref((x >> 7) as u64, 32, false),
                bfind_ref(((x as i32) >> 9) as u32 as u64, 32, true),
            ]);
            for c in BIT_COUNT_CONSTANTS {
                let c32 = c as u32 as u64;
                let (zeros32, zeros64) = ((c as u32).leading_zeros(), c.leading_zeros());
                want.extend([zeros32, zeros32, bfind_ref(c32, 32, false), bfind_ref(c32, 32, true)]);
                want.push((c as u32).reverse_bits());
                want.extend([zeros64, zeros64, bfind_ref(c, 64, false), bfind_ref(c, 64, true)]);
                want.push((c as u32).count_ones());
            }
            let log2 = |a: u32| if a == 0 { u32::MAX } else { a.trailing_zeros() };
            want.extend([log2(x), x & 31, u32::MAX]);
            want.extend([1, 1]);
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "thread {t}: x = {x:#x}, l = {l:#x}");
            let mut wide = vec![l.reverse_bits(), l.reverse_bits(), l.count_ones() as u64, l.count_ones() as u64];
            for c in BIT_COUNT_CONSTANTS {
                wide.extend([c.reverse_bits(), c.reverse_bits(), c.count_ones() as u64, c.count_ones() as u64]);
            }
            assert_eq!(out64[t * N64..(t + 1) * N64], wide[..], "thread {t}: l = {l:#x}");
        }
    }
    Case { name: "bit_counts", kernel, cfg: launch(1), small_only: false, check }
}

/// Kernels that need SM90 or newer: simulated, and compiled for sm_90, but
/// not run on older GPUs.
pub fn sm90() -> Vec<Case> {
    vec![
        mbarriers_sm90(),
        cluster_barriers_sm90([4, 1, 1], [2, 1, 1]),
        cluster_barriers_sm90([4, 1, 1], [4, 1, 1]),
        cluster_barriers_sm90([1, 2, 2], [1, 2, 2]),
        bulk_copy_producer_warp(),
        half_sm90(),
        cvt_i8_bf16_sm90(),
        cluster_ranks(),
        cluster_ranks_2x2(),
        carry_around_smem_registers_sm90(),
        cluster_scope_handoff(),
        atomics_sm90(),
        consumer_first_try_wait(),
        pointer_cluster_scope(),
        stmatrix_forms(),
        stmatrix_epilogue(),
    ]
    .into_iter()
    .chain(bulk_cases_sm90())
    .chain(pipeline_cases_sm90())
    .chain(async_sched_cases_sm90())
    .chain(tma_load_cases_sm90())
    .chain(tma_store_cases_sm90())
    .chain(tma_im2col_cases_sm90())
    .chain(cluster_copy_cases_sm90())
    // Distributed shared memory.
    .chain(dsmem_cases_sm90())
    // The GEMM host helper cases.
    .chain(gemm_helper_cases_sm90())
    // The TMA pipelines.
    .chain(tma_pipeline_cases_sm90())
    .collect()
}

pub fn all() -> Vec<Case> {
    vec![
        int_i8(),
        int_i16(),
        int_i32(),
        int_i64(),
        int_u8(),
        int_u16(),
        int_u32(),
        int_u64(),
        float_f32(),
        float_f64(),
        float_f16(),
        float_bf16(),
        casts(),
        wide_multiply(),
        while_loop(),
        stack_array(),
        reinterpreted_memory(),
        runtime_sized_buffer(),
        dynamic_shared_memory(),
        warp_operations(),
        atomics(),
        printing(),
        inline_ptx_fragment(),
        mma_k8(),
        mma_f16_accumulate(),
        mma_tf32(),
        mma_f64(),
        mma_bf16(),
        mma_bf16_k8(),
        mma_f16_accumulate_k8(),
        mma_tf32_k8(),
        tiled_copy_tv_ldmatrix(),
        mma_tiled_copies(),
        named_barriers(),
        mbarriers_sm80(),
        warp_helpers(),
        increment_coords(),
        dynamic_layouts(),
        basis_layouts(),
        math_f32(),
        math_f64(),
        math_half(),
        vars(),
        loop_hints(),
        carry_structs(),
        merged_facts(),
        bit_fields_i32(),
        bit_fields_u32(),
        bit_fields_i64(),
        bit_fields_u64(),
        logic_and_shifts(),
        int_multiplies(),
        bit_counts(),
        carry_words_u32(),
        carry_words_u64(),
        carry_multiply(),
        carry_in_loop(),
        inline_ptx_named(),
        inline_asm_raw(),
        mlir_ops(),
        carry_around_division(),
        carry_around_collectives(),
        dynamic_tiles(),
        dynamic_region(),
        cp_async_mbarrier(),
        cp_async_consumer_first(),
        cp_async_pipeline(),
        carry_around_smem_registers(),
        ordered_handoffs(),
        cache_hints(),
        copy_atom_cache_hints(),
        prefetches(),
        global_copy_atoms(),
        cotiled_copies(),
        composed_layout_inner(),
        pointer_swizzle(),
        strong_vectors(),
        carry_around_memory_accesses(),
        assume_vector_copy(),
        int64_strides(),
        fast_divmod(),
        atomic_orders(),
        atomic_lock(),
        atomic_float_adds(),
        atomic_float_extrema(),
        reductions(),
        consumer_first_named_barrier(),
        consumer_first_mbarrier(),
        mbarrier_ring(),
        shape_arith(),
        coord_tensor_copy(),
        boolean_conversions(),
        tf32_conversions(),
        mma_tf32_fragments(),
        mixed_promotion(),
        int128_ops(),
        int4_conversions(),
        int4_fast_conversions(),
    ]
    .into_iter()
    .chain(warp_collectives())
    .chain(packed_half_and_conversions())
    .chain(mma_sparse())
    .chain(subbyte_tensors())
    .chain([mx_storage()])
    .chain(autovec_cases())
    .chain([fence_proxy_alias()])
    .chain(tensor_ssa_cases())
    .chain(vector_cases())
    .chain(masked_cases())
    .chain(extras_cases())
    .chain(raw_ptr_cases())
    .chain(struct_cases())
    .chain(pipeline_cases())
    .collect()
}

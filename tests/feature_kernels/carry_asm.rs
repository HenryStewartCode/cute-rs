// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! Carry chains (through registers, shared memory and memory accesses) and the inline PTX and MLIR kernels.

use super::*;
// Extended-precision arithmetic: carry chains through PTX's carry flag,
// checked against u128 arithmetic.

/// The 256-bit `a + b + carry_in` of numbers held as (low, high) u128
/// halves, and its carry out.
fn add256(a: [u128; 2], b: [u128; 2], carry_in: bool) -> ([u128; 2], bool) {
    let (lo, c1) = a[0].overflowing_add(b[0]);
    let (lo, c2) = lo.overflowing_add(carry_in as u128);
    let (hi, c3) = a[1].overflowing_add(b[1]);
    let (hi, c4) = hi.overflowing_add((c1 | c2) as u128);
    ([lo, hi], c3 | c4)
}

/// The 256-bit `a - b - borrow_in` of (low, high) u128 halves, and its
/// borrow out.
fn sub256(a: [u128; 2], b: [u128; 2], borrow_in: bool) -> ([u128; 2], bool) {
    let (lo, b1) = a[0].overflowing_sub(b[0]);
    let (lo, b2) = lo.overflowing_sub(borrow_in as u128);
    let (hi, b3) = a[1].overflowing_sub(b[1]);
    let (hi, b4) = hi.overflowing_sub((b1 | b2) as u128);
    ([lo, hi], b3 | b4)
}

/// Four 32-bit words (low first) as a u128.
fn u128_of(w: [u32; 4]) -> u128 {
    w.iter().rev().fold(0, |acc, &x| acc << 32 | x as u128)
}

/// A u128 as four 32-bit words, low first.
fn words_of(x: u128) -> [u32; 4] {
    [x as u32, (x >> 32) as u32, (x >> 64) as u32, (x >> 96) as u32]
}

/// `a + b + carry_in` over four 32-bit words, with u128 arithmetic.
fn add_words_u32(a: [u32; 4], b: [u32; 4], carry_in: bool) -> ([u32; 4], bool) {
    let (s, c1) = u128_of(a).overflowing_add(u128_of(b));
    let (s, c2) = s.overflowing_add(carry_in as u128);
    (words_of(s), c1 | c2)
}

/// `a - b - borrow_in` over four 32-bit words, with u128 arithmetic.
fn sub_words_u32(a: [u32; 4], b: [u32; 4], borrow_in: bool) -> ([u32; 4], bool) {
    let (d, b1) = u128_of(a).overflowing_sub(u128_of(b));
    let (d, b2) = d.overflowing_sub(borrow_in as u128);
    (words_of(d), b1 | b2)
}

/// Four 64-bit words (low first) as (low, high) u128 halves.
fn halves_of(w: [u64; 4]) -> [u128; 2] {
    [w[0] as u128 | (w[1] as u128) << 64, w[2] as u128 | (w[3] as u128) << 64]
}

/// `a + b + carry_in` over four 64-bit words, with u128 arithmetic.
fn add_words_u64(a: [u64; 4], b: [u64; 4], carry_in: bool) -> ([u64; 4], bool) {
    let ([lo, hi], c) = add256(halves_of(a), halves_of(b), carry_in);
    ([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64], c)
}

/// `a - b - borrow_in` over four 64-bit words, with u128 arithmetic.
fn sub_words_u64(a: [u64; 4], b: [u64; 4], borrow_in: bool) -> ([u64; 4], bool) {
    let ([lo, hi], br) = sub256(halves_of(a), halves_of(b), borrow_in);
    ([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64], br)
}

/// Columns of the carry-chain kernels: a sum of four words with its carry
/// out, then a difference with its borrow out (5 + 5), a word computed
/// between two carry instructions (1), CuTe DSL's two-word forms, whose
/// `addc` and `subc` do not write the flag, so that two of them add the same
/// carry (3 + 3), the sum and difference on the signed type of the same width
/// (4 + 4), and `add_with_carry` and `sub_with_borrow` chains from a runtime
/// carry (5 + 5).
const CARRY_COLS: usize = 5 + 5 + 1 + 3 + 3 + 4 + 4 + 5 + 5;

/// Multi-word sums and differences of four words of `$t` (signed `$s`): a
/// 128-bit number for 32-bit words, 256 bits for 64-bit words. The words are
/// spread over every bit, and threads shape them so that carries run
/// through every word: `t % 4 == 1` gives `a` all-ones low words and an odd
/// low word of `b` (the sum carries all the way), `t % 4 == 2` zero low words
/// of `a` and an odd low word of `b` (the difference borrows all the way),
/// and `t % 4 == 3` low words of `b` that complement `a`'s (the sum carries
/// all the way only with a carry in). `$add` and `$sub` are the references,
/// in u128 arithmetic.
macro_rules! carry_words_case {
    ($name:ident, $t:ty, $s:ty, $add:expr, $sub:expr) => {
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("a", Layout::row_major((T, 4)));
                let ys = param_tensor::<$t>("b", Layout::row_major((T, 4)));
                let out = param_tensor::<$t>("out", Layout::row_major((T, CARRY_COLS)));
                let t = thread_idx().x;
                let m = t % 4;
                let ones = <$t>::MAX;
                let mut a: [Value<$t>; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD as $t);
                let mut b: [Value<$t>; 4] = std::array::from_fn(|i| ys.load((t, i)) * SPREAD as $t);
                for i in 0..3 {
                    b[i] = select(m.eq(3), !a[i], b[i]);
                    a[i] = select(m.eq(1), Value::constant(ones), select(m.eq(2), Value::constant(0), a[i]));
                }
                b[0] = select(m.eq(1) | m.eq(2), b[0] | 1, b[0]);
                let zero = || Value::constant(0 as $t);
                let mut cols: Vec<Value<$t>> = Vec::new();
                // The sum, with a word computed between two carry instructions.
                cols.push(add_cc(a[0], b[0]));
                let between = (a[1] ^ b[2]) * 3 + (a[0] >> 3);
                cols.extend([addc_cc(a[1], b[1]), addc_cc(a[2], b[2]), addc_cc(a[3], b[3]), addc(zero(), zero())]);
                // The difference; the last subc gives 0 - borrow.
                cols.extend([sub_cc(a[0], b[0]), subc_cc(a[1], b[1]), subc_cc(a[2], b[2]), subc_cc(a[3], b[3])]);
                cols.push(subc(zero(), zero()));
                cols.push(between);
                // CuTe DSL's addc and subc read the flag and leave it.
                cols.extend([add_cc(a[0], b[0]), addc(a[1], b[1]), addc(a[2], b[2])]);
                cols.extend([sub_cc(a[0], b[0]), subc(a[1], b[1]), subc(a[2], b[2])]);
                // The signed type gives the same bits.
                let (sa, sb) = (a.map(|v| v.bitcast::<$s>()), b.map(|v| v.bitcast::<$s>()));
                let signed = [add_cc(sa[0], sb[0]), addc_cc(sa[1], sb[1]), addc_cc(sa[2], sb[2]), addc(sa[3], sb[3])];
                cols.extend(signed.map(|v| v.bitcast::<$t>()));
                let signed = [sub_cc(sa[0], sb[0]), subc_cc(sa[1], sb[1]), subc_cc(sa[2], sb[2]), subc(sa[3], sb[3])];
                cols.extend(signed.map(|v| v.bitcast::<$t>()));
                // Carries as values, from a runtime carry in.
                let mut carry = ((t >> 2) & 1).to::<$t>();
                for i in 0..4 {
                    let (s, c) = add_with_carry(a[i], b[i], carry);
                    cols.push(s);
                    carry = c;
                }
                cols.push(carry);
                let mut borrow = ((t >> 3) & 1).to::<$t>();
                for i in 0..4 {
                    let (d, c) = sub_with_borrow(a[i], b[i], borrow);
                    cols.push(d);
                    borrow = c;
                }
                cols.push(borrow);
                assert_eq!(cols.len(), CARRY_COLS);
                for (j, v) in cols.into_iter().enumerate() {
                    out.store((t, j), v);
                }
            });
            fn check(args: &[KernelArg]) {
                let add: fn([$t; 4], [$t; 4], bool) -> ([$t; 4], bool) = $add;
                let sub: fn([$t; 4], [$t; 4], bool) -> ([$t; 4], bool) = $sub;
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                let mut rippled = [0; 4];
                for t in 0..T {
                    let mut a: [$t; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD as $t));
                    let mut b: [$t; 4] = std::array::from_fn(|i| ys[t * 4 + i].wrapping_mul(SPREAD as $t));
                    let m = t % 4;
                    for i in 0..3 {
                        if m == 3 {
                            b[i] = !a[i];
                        }
                        a[i] = match m {
                            1 => <$t>::MAX,
                            2 => 0,
                            _ => a[i],
                        };
                    }
                    if m == 1 || m == 2 {
                        b[0] |= 1;
                    }
                    let mut want: Vec<$t> = Vec::new();
                    let (s, c) = add(a, b, false);
                    want.extend(s);
                    want.push(c as $t);
                    let (d, br) = sub(a, b, false);
                    want.extend(d);
                    want.push(if br { <$t>::MAX } else { 0 });
                    want.push((a[1] ^ b[2]).wrapping_mul(3).wrapping_add(a[0] >> 3));
                    let (x0, c0) = a[0].overflowing_add(b[0]);
                    want.extend([x0, a[1].wrapping_add(b[1]).wrapping_add(c0 as $t)]);
                    want.push(a[2].wrapping_add(b[2]).wrapping_add(c0 as $t));
                    let (y0, b0) = a[0].overflowing_sub(b[0]);
                    want.extend([y0, a[1].wrapping_sub(b[1]).wrapping_sub(b0 as $t)]);
                    want.push(a[2].wrapping_sub(b[2]).wrapping_sub(b0 as $t));
                    want.extend(s);
                    want.extend(d);
                    let (s, c) = add(a, b, (t >> 2) & 1 == 1);
                    want.extend(s);
                    want.push(c as $t);
                    let (d, br) = sub(a, b, (t >> 3) & 1 == 1);
                    want.extend(d);
                    want.push(br as $t);
                    // The shaped threads' sums carry (and differences borrow)
                    // out of each of the low three words.
                    let carries = (0..3).filter(|&i| {
                        let lower = |x: &[$t; 4]| {
                            let mut w = *x;
                            w[i + 1..].fill(0);
                            w
                        };
                        add(lower(&a), lower(&b), false).0[i + 1] == 1
                    });
                    rippled[m] += carries.count();
                    let got = &out[t * CARRY_COLS..(t + 1) * CARRY_COLS];
                    assert_eq!(got, &want[..], "thread {t}: a = {a:x?}, b = {b:x?}");
                }
                assert_eq!(rippled[1], 3 * T / 4, "every shaped sum should carry out of each low word");
            }
            Case { name: stringify!($name), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

carry_words_case!(carry_words_u32, u32, i32, add_words_u32, sub_words_u32);
carry_words_case!(carry_words_u64, u64, i64, add_words_u64, sub_words_u64);

/// Products through the multiply-adds, on data spread over every bit: the
/// 128-bit product of two 64-bit numbers held as 32-bit words, from
/// `mad_cc`, `madc_cc`, `addc` and `mul_hi` (4 columns); a `mad_cc`,
/// `madc_cc`, `madc` chain on Int32 (whose low product half is the signed
/// product's, 3); the low 128 bits of a 128-bit number times a 64-bit one
/// plus a 128-bit one, from 64-bit `mad_cc`, `madc` and `mul_hi` (2); and a
/// chain on Int64 (3).
pub fn carry_multiply() -> Case {
    const N32: usize = 4 + 3;
    const N64: usize = 2 + 3;
    let kernel = Kernel::builder("carry_multiply").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::row_major((T, 4)));
        let ss = param_tensor::<i32>("ss", Layout::row_major((T, 6)));
        let ls = param_tensor::<u64>("ls", Layout::row_major((T, 5)));
        let ms = param_tensor::<i64>("ms", Layout::row_major((T, 5)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let t = thread_idx().x;
        let x: [Uint32; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD as u32);
        let s: [Int32; 6] = std::array::from_fn(|i| ss.load((t, i)) * SPREAD as i32);
        let l: [Uint64; 5] = std::array::from_fn(|i| ls.load((t, i)) * SPREAD);
        let m: [Int64; 5] = std::array::from_fn(|i| ms.load((t, i)) * SPREAD as i64);
        // (a1 a0) * (b1 b0), a partial product at a time.
        let (a0, a1, b0, b1) = (x[0], x[1], x[2], x[3]);
        let (h00, h01, h10, h11) = (mul_hi(a0, b0), mul_hi(a0, b1), mul_hi(a1, b0), mul_hi(a1, b1));
        let p0 = a0 * b0;
        let p1 = mad_cc(a0, b1, h00);
        let p2 = madc_cc(a1, b1, h01);
        let p3 = addc(h11, 0u32);
        let p1 = mad_cc(a1, b0, p1);
        let p2 = addc_cc(p2, h10);
        let p3 = addc(p3, 0u32);
        let signed = [mad_cc(s[0], s[1], s[2]), madc_cc(s[3], s[4], s[5]), madc(s[0], s[4], s[2])];
        let mut narrow = vec![p0, p1, p2, p3];
        narrow.extend(signed.map(|v| v.bitcast::<u32>()));
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
        // (x1 x0) * y + (z1 z0), modulo 2^128.
        let (x0, x1, y, z0, z1) = (l[0], l[1], l[2], l[3], l[4]);
        let r0 = mad_cc(x0, y, z0);
        let r1 = madc(x1, y, z1) + mul_hi(x0, y);
        let signed = [mad_cc(m[0], m[1], m[2]), madc_cc(m[3], m[4], m[0]), madc(m[1], m[3], m[4])];
        let mut wide = vec![r0, r1];
        wide.extend(signed.map(|v| v.bitcast::<u64>()));
        for (j, v) in wide.into_iter().enumerate() {
            out64.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ss) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        let (ls, ms) = (args[2].to_vec::<u64>(), args[3].to_vec::<i64>());
        let (out32, out64) = (args[4].to_vec::<u32>(), args[5].to_vec::<u64>());
        let mut carries = 0;
        for t in 0..T {
            let x: [u32; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD as u32));
            let s: [i32; 6] = std::array::from_fn(|i| ss[t * 6 + i].wrapping_mul(SPREAD as i32));
            let l: [u64; 5] = std::array::from_fn(|i| ls[t * 5 + i].wrapping_mul(SPREAD));
            let m: [i64; 5] = std::array::from_fn(|i| ms[t * 5 + i].wrapping_mul(SPREAD as i64));
            let product = (x[1] as u128) << 32 | x[0] as u128;
            let product = product * ((x[3] as u128) << 32 | x[2] as u128);
            let mut want = words_of(product).to_vec();
            // Signed: the low half of the signed product, plus the addend and
            // the carry, carrying out of 32 bits.
            let low = |p: i32, q: i32| (p as i64 * q as i64) as u32 as u64;
            let e0 = low(s[0], s[1]) + s[2] as u32 as u64;
            let e1 = low(s[3], s[4]) + s[5] as u32 as u64 + (e0 >> 32);
            let e2 = low(s[0], s[4]) + s[2] as u32 as u64 + (e1 >> 32);
            carries += (e0 >> 32) + (e1 >> 32);
            want.extend([e0 as u32, e1 as u32, e2 as u32]);
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "thread {t}: x = {x:#x?}, s = {s:#x?}");
            let wide = |lo: u64, hi: u64| (hi as u128) << 64 | lo as u128;
            let r = wide(l[0], l[1]).wrapping_mul(l[2] as u128).wrapping_add(wide(l[3], l[4]));
            let mut want = vec![r as u64, (r >> 64) as u64];
            let low = |p: i64, q: i64| (p as i128 * q as i128) as u64 as u128;
            let e0 = low(m[0], m[1]) + m[2] as u64 as u128;
            let e1 = low(m[3], m[4]) + m[0] as u64 as u128 + (e0 >> 64);
            let e2 = low(m[1], m[3]) + m[4] as u64 as u128 + (e1 >> 64);
            carries += ((e0 >> 64) + (e1 >> 64)) as u64;
            want.extend([e0 as u64, e1 as u64, e2 as u64]);
            assert_eq!(out64[t * N64..(t + 1) * N64], want[..], "thread {t}: l = {l:#x?}, m = {m:#x?}");
        }
        assert!(carries > T as u64 / 2, "the multiply-add chains should carry often, not {carries} times");
    }
    Case { name: "carry_multiply", kernel, cfg: launch(1), small_only: false, check }
}

/// Whole carry chains inside control flow, which each body holds from
/// start to end: a runtime loop with a trip count that differs between
/// threads adds a 128-bit number to one accumulator and subtracts it from
/// another in every iteration (the accumulators carried through the loop),
/// and the odd threads add a third chain in an `if_` branch.
pub fn carry_in_loop() -> Case {
    const COLS: usize = 12;
    let kernel = Kernel::builder("carry_in_loop").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, 4)));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x: [Uint32; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD as u32);
        let y: [Uint32; 4] = std::array::from_fn(|i| ys.load((t, i)) * SPREAD as u32);
        let (sum, diff) = for_range_carry(0, t % 4 + 3, 1, (x, x), |i, (s, d)| {
            let z = [y[0] ^ (i.to::<u32>() * 0x9e37_79b9u32), y[1], y[2], y[3] + i.to::<u32>()];
            let s = [add_cc(s[0], z[0]), addc_cc(s[1], z[1]), addc_cc(s[2], z[2]), addc(s[3], z[3])];
            let d = [sub_cc(d[0], z[0]), subc_cc(d[1], z[1]), subc_cc(d[2], z[2]), subc(d[3], z[3])];
            (s, d)
        });
        let extra = Var::new(sum);
        if_((t & 1).eq(1), || {
            let s = extra.get();
            extra.set([add_cc(s[0], x[0]), addc_cc(s[1], x[1]), addc_cc(s[2], x[2]), addc(s[3], x[3])]);
        });
        for (j, v) in sum.into_iter().chain(diff).chain(extra.get()).enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let x: [u32; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD as u32));
            let y: [u32; 4] = std::array::from_fn(|i| ys[t * 4 + i].wrapping_mul(SPREAD as u32));
            let (mut sum, mut diff) = (u128_of(x), u128_of(x));
            for i in 0..(t % 4 + 3) as u32 {
                let z = u128_of([y[0] ^ i.wrapping_mul(0x9e37_79b9), y[1], y[2], y[3].wrapping_add(i)]);
                sum = sum.wrapping_add(z);
                diff = diff.wrapping_sub(z);
            }
            let extra = if t % 2 == 1 { sum.wrapping_add(u128_of(x)) } else { sum };
            let want: Vec<u32> = [sum, diff, extra].into_iter().flat_map(words_of).collect();
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}");
        }
    }
    Case { name: "carry_in_loop", kernel, cfg: launch(1), small_only: false, check }
}

/// Carry chains around divisions that leave the carry flag alone: a 64-bit
/// division and remainder by constants (which LLVM turns into
/// multiplications) and a 32-bit division by a runtime value (one
/// `div.u32`), each between a carry-out and its carry-in, in the 128-bit
/// sum and difference of numbers held as two u64 words; the odd threads'
/// low words carry and borrow. (A 64-bit division by a runtime value, which
/// LLVM lowers with a branch, cannot sit inside a chain: the validator
/// rejects it.)
pub fn carry_around_division() -> Case {
    const COLS: usize = 9;
    let kernel = Kernel::builder("carry_around_division").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, 2)));
        let out = param_tensor::<u64>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        let (y, z) = (ys.load((t, 0)) * SPREAD as u32, ys.load((t, 1)) | 1u32);
        let low = add_cc(a0, b0);
        let quotient = a0 / 7u64;
        let remainder = b1 % 1_000_003u64;
        let narrow = (y / z).to::<u64>();
        let high = addc_cc(a1, b1);
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        let d0 = sub_cc(b0, a0);
        let signed = (a1.bitcast::<i64>() / -3i64).bitcast::<u64>();
        let d1 = subc(b1, a1);
        for (j, v) in [low, high, top, quotient, remainder, narrow, d0, d1, signed].into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>(), args[2].to_vec::<u64>());
        let (mut carries, mut borrows) = (0, 0);
        for t in 0..T {
            let w: [u64; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD));
            let odd = t % 2 == 1;
            let (a0, a1) = (if odd { u64::MAX } else { w[0] }, w[1]);
            let (b0, b1) = (if odd { w[2] | 1 } else { w[2] }, w[3]);
            let (y, z) = (ys[t * 2].wrapping_mul(SPREAD as u32), ys[t * 2 + 1] | 1);
            let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
            let (sum, carry) = a.overflowing_add(b);
            let difference = b.wrapping_sub(a);
            carries += a0.overflowing_add(b0).1 as usize;
            borrows += (b0 < a0) as usize;
            let want = [
                sum as u64,
                (sum >> 64) as u64,
                carry as u64,
                a0 / 7,
                b1 % 1_000_003,
                (y / z) as u64,
                difference as u64,
                (difference >> 64) as u64,
                (a1 as i64 / -3) as u64,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}: a = {a:#x}, b = {b:#x}");
        }
        assert!(carries >= T / 2 && borrows >= T / 2, "the odd threads should carry and borrow: {carries}, {borrows}");
    }
    Case { name: "carry_around_division", kernel, cfg: launch(1), small_only: false, check }
}

/// Carry chains through the warp collectives, which keep the carry flag:
/// each is one convergent PTX instruction that neither branches nor touches
/// the flag (see `cute::ir::carry_flag_breaker`), so the validator accepts a
/// chain around them and the simulator keeps the flag. Between a carry-out
/// and its carry-in sit shuffles with a runtime segment mask and clamp
/// (groups of 8 lanes below lane 16, the whole warp above), a 64-bit
/// butterfly shuffle (two `shfl.sync`), `redux.sync`, `match.any` on 32-
/// and 64-bit values and `match.all`, a ballot, a butterfly reduction over
/// groups of 8 and `sync_warp_with` per-lane half-warp masks: in the
/// 128-bit sum and difference of numbers held as two u64 words and in a
/// 96-bit sum of u32 words, whose odd threads carry (and borrow) out of
/// every word below the top one.
pub fn carry_around_collectives() -> Case {
    const N64: usize = 6;
    const N32: usize = 3 + 10;
    let kernel = Kernel::builder("carry_around_collectives").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, 6)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD);
        let y: [Uint32; 6] = std::array::from_fn(|i| ys.load((t, i)) * SPREAD as u32);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        // A runtime mask_and_clamp: groups of 8 lanes below lane 16, the
        // whole warp above.
        let clamp = select(lane.lt(16), 0x1807i32, 31i32);
        // The 128-bit sum, with collectives between its words.
        let low = add_cc(a0, b0);
        let shuffled = shuffle_sync_op(y[0], lane + 1, FULL_MASK, clamp, ShuffleKind::Idx);
        let swapped = shuffle_sync_bfly(w[1], 3);
        let total = warp_redux_sync(y[1], ReduxKind::Add);
        let high = addc_cc(a1, b1);
        let peers = match_sync(FULL_MASK, y[2] & 3u32, MatchKind::Any);
        let wide_peers = match_sync(FULL_MASK, w[0] & 7u64, MatchKind::Any);
        let uniform = match_sync(FULL_MASK, warp_idx(), MatchKind::All);
        let ballot = vote_ballot_sync(odd);
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        // The difference, around a butterfly reduction and a warp barrier.
        let d0 = sub_cc(b0, a0);
        let group_sum = warp_reduction_sum(y[3], 8);
        sync_warp_with(select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32));
        let d1 = subc(b1, a1);
        // A 96-bit sum of u32 words, around shuffles with the runtime clamp
        // and a redux.
        let x = [select(odd, u32::MAX, y[3]), select(odd, u32::MAX, y[4]), y[5]];
        let z = [select(odd, y[0] | 1u32, y[0]), select(odd, 0u32, y[1]), y[2]];
        let s0 = add_cc(x[0], z[0]);
        let rotated = shuffle_sync_op(y[4], lane + 3, FULL_MASK, clamp, ShuffleKind::Idx);
        let s1 = addc_cc(x[1], z[1]);
        let maximum = warp_redux_sync(y[5], ReduxKind::Max);
        let down = shuffle_sync_op(y[5], 2, FULL_MASK, clamp, ShuffleKind::Down);
        let s2 = addc(x[2], z[2]);
        for (j, v) in [low, high, top, d0, d1, swapped].into_iter().enumerate() {
            out64.store((t, j), v);
        }
        let narrow =
            [s0, s1, s2, shuffled, total, peers, wide_peers, uniform, ballot, group_sum, rotated, maximum, down];
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>());
        let (out64, out32) = (args[2].to_vec::<u64>(), args[3].to_vec::<u32>());
        let w = |t: usize, i: usize| xs[t * 4 + i].wrapping_mul(SPREAD);
        let y = |t: usize, i: usize| ys[t * 6 + i].wrapping_mul(SPREAD as u32);
        let (mut carries, mut borrows) = (0, 0);
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, (t % 32) as u32);
            let odd = t % 2 == 1;
            let (a0, a1) = (if odd { u64::MAX } else { w(t, 0) }, w(t, 1));
            let (b0, b1) = (if odd { w(t, 2) | 1 } else { w(t, 2) }, w(t, 3));
            let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
            let (sum, carry) = a.overflowing_add(b);
            let difference = b.wrapping_sub(a);
            carries += a0.overflowing_add(b0).1 as usize;
            borrows += (b0 < a0) as usize;
            let swapped = w(warp + (lane ^ 3) as usize, 1);
            let want =
                [sum as u64, (sum >> 64) as u64, carry as u64, difference as u64, (difference >> 64) as u64, swapped];
            assert_eq!(out64[t * N64..(t + 1) * N64], want, "thread {t}: a = {a:#x}, b = {b:#x}");
            let clamp = if lane < 16 { 0x1807 } else { 31 };
            let read = |i: usize, source: Option<u32>| y(source.map_or(t, |s| warp + s as usize), i);
            let mask_of = |f: &dyn Fn(usize) -> bool| (0..32).filter(|&l| f(warp + l)).fold(0u32, |m, l| m | 1 << l);
            let x = [if odd { u32::MAX } else { y(t, 3) }, if odd { u32::MAX } else { y(t, 4) }, y(t, 5)];
            let z = [if odd { y(t, 0) | 1 } else { y(t, 0) }, if odd { 0 } else { y(t, 1) }, y(t, 2)];
            let words = |v: [u32; 3]| v[0] as u128 | (v[1] as u128) << 32 | (v[2] as u128) << 64;
            let s = words(x) + words(z);
            let group = warp + (lane as usize & !7);
            let want = [
                s as u32,
                (s >> 32) as u32,
                (s >> 64) as u32,
                read(0, shfl_ref(ShuffleKind::Idx, lane, lane + 1, clamp)),
                (warp..warp + 32).fold(0u32, |acc, l| acc.wrapping_add(y(l, 1))),
                mask_of(&|l| y(l, 2) & 3 == y(t, 2) & 3),
                mask_of(&|l| w(l, 0) & 7 == w(t, 0) & 7),
                u32::MAX,
                0xaaaa_aaaa,
                (group..group + 8).fold(0u32, |acc, l| acc.wrapping_add(y(l, 3))),
                read(4, shfl_ref(ShuffleKind::Idx, lane, lane + 3, clamp)),
                (warp..warp + 32).map(|l| y(l, 5)).max().unwrap(),
                read(5, shfl_ref(ShuffleKind::Down, lane, 2, clamp)),
            ];
            assert_eq!(out32[t * N32..(t + 1) * N32], want, "thread {t}");
            if odd {
                assert_eq!(s >> 32 & 0xffff_ffff, 0, "thread {t}: the middle word should carry");
            }
        }
        assert!(carries >= T / 2 && borrows >= T / 2, "the odd threads should carry and borrow: {carries}, {borrows}");
    }
    Case { name: "carry_around_collectives", kernel, cfg: launch(1), small_only: false, check }
}

// Hand-written PTX and MLIR.

/// CuTe DSL's `inline_ptx` with named operands: two write-only outputs; a
/// read-write operand beside a write-only output; a predicate that holds in
/// the odd lanes only, guarding a read-write operand (kept elsewhere) and a
/// write-only output (0 elsewhere); constant Float32 operands (`f`
/// registers, never an immediate); 16-bit and Boolean operands; integer
/// constants as immediates (`lop3`'s table, which must be one, and
/// constants LLVM prints as negative numbers: a Uint32 and a Uint64 with
/// the top bit set); a constant Boolean in a predicate register; a `$` that
/// is PTX's own; and a pointer operand that the simulation ignores.
pub fn inline_ptx_named() -> Case {
    const COLS: usize = 14;
    let kernel = Kernel::builder("inline_ptx_named").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let fs = param_tensor::<f32>("fs", Layout::col_major(T));
        let hs = param_tensor::<u16>("hs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let (x, y, f, h) = (xs.load(t) * SPREAD, ys.load(t) * SPREAD as u32, fs.load(t), hs.load(t));
        let (lo, hi) = InlinePtx::new("mov.b64 {{$w0}, {$w1}}, {$r0};")
            .write::<u32>()
            .write::<u32>()
            .read(x)
            .pure()
            .sim(|r, _| vec![r[0] & 0xffff_ffff, r[0] >> 32])
            .emit();
        let (acc, rotated) =
            InlinePtx::new("add.u32 {$rw0}, {$rw0}, {$r0};\n\tshf.l.wrap.b32 {$w0}, {$r1}, {$r1}, {$r0};")
                .read_write(y)
                .write::<u32>()
                .read(lo)
                .read(hi)
                .sim(|r, rw| {
                    rw[0] = (rw[0] + r[0]) & 0xffff_ffff;
                    vec![(r[1] as u32).rotate_left(r[0] as u32 % 32) as u64]
                })
                .emit();
        let odd = (y & 1u32).eq(1u32);
        let kept: Uint32 = InlinePtx::new("max.u32 {$rw0}, {$rw0}, {$r0};")
            .read_write(lo)
            .read(hi)
            .predicate(odd)
            .sim(|r, rw| {
                rw[0] = rw[0].max(r[0]);
                vec![]
            })
            .emit();
        let guarded: Uint32 = InlinePtx::new("xor.b32 {$w0}, {$r0}, {$r1};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .predicate(odd)
            .sim(|r, _| vec![r[0] ^ r[1]])
            .emit();
        let fused: Float32 = InlinePtx::new("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};")
            .write::<f32>()
            .read(f)
            .read(1.5f32)
            .read(-0.25f32)
            .pure()
            .sim(|r, _| vec![f32::from_bits(r[0] as u32).mul_add(1.5, -0.25).to_bits() as u64])
            .emit();
        let narrow: Uint16 = InlinePtx::new("{ .reg .b16 t$h; add.u16 t$h, {$r0}, {$r1}; mov.b16 {$w0}, t$h; }")
            .write::<u16>()
            .read(h)
            .read(7u16)
            .sim(|r, _| vec![(r[0] + r[1]) & 0xffff])
            .emit();
        let chosen: Uint32 = InlinePtx::new("selp.b32 {$w0}, {$r0}, {$r1}, {$r2};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .read(odd)
            .sim(|r, _| vec![if r[2] != 0 { r[0] } else { r[1] }])
            .emit();
        let table: Uint32 = InlinePtx::new("lop3.b32 {$w0}, {$r0}, {$r1}, {$r2}, {$r3};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .read(acc)
            .read(0x96u32)
            .pure()
            .sim(|r, _| vec![lop3_ref(r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u8) as u64])
            .emit();
        let lowered: Uint32 = InlinePtx::new("add.u32 {$w0}, {$r0}, {$r1};")
            .write::<u32>()
            .read(lo)
            .read(0xffff_fff0u32)
            .pure()
            .sim(|r, _| vec![(r[0] + r[1]) & 0xffff_ffff])
            .emit();
        let flipped: Uint64 = InlinePtx::new("xor.b64 {$w0}, {$r0}, {$r1};")
            .write::<u64>()
            .read(x)
            .read(0x8000_0000_0000_0001u64)
            .pure()
            .sim(|r, _| vec![r[0] ^ r[1]])
            .emit();
        let never: Uint32 = InlinePtx::new("selp.b32 {$w0}, {$r0}, {$r1}, {$r2};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .read(false)
            .pure()
            .sim(|r, _| vec![if r[2] != 0 { r[0] } else { r[1] }])
            .emit();
        InlinePtx::new("prefetch.global.L2 [{$r0}];").read_ptr(&xs.ptr().offset(t)).sim(|_, _| vec![]).emit();
        let cols = [
            lo,
            hi,
            acc,
            rotated,
            kept,
            guarded,
            fused.bitcast(),
            narrow.to(),
            chosen,
            table,
            lowered,
            flipped.to(),
            (flipped >> 32u64).to(),
            never,
        ];
        for (j, v) in cols.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>());
        let (fs, hs, out) = (args[2].to_vec::<f32>(), args[3].to_vec::<u16>(), args[4].to_vec::<u32>());
        let odd_lanes = (0..T).filter(|&t| ys[t] & 1 == 1).count();
        assert!(odd_lanes > 0 && odd_lanes < T, "the predicate should hold in some lanes only, not {odd_lanes}");
        for t in 0..T {
            let (x, y) = (xs[t].wrapping_mul(SPREAD), ys[t].wrapping_mul(SPREAD as u32));
            let (lo, hi) = (x as u32, (x >> 32) as u32);
            let odd = y & 1 == 1;
            let want = [
                lo,
                hi,
                y.wrapping_add(lo),
                hi.rotate_left(lo % 32),
                if odd { lo.max(hi) } else { lo },
                if odd { lo ^ hi } else { 0 },
                fs[t].mul_add(1.5, -0.25).to_bits(),
                hs[t].wrapping_add(7) as u32,
                if odd { lo } else { hi },
                lo ^ hi ^ y.wrapping_add(lo),
                lo.wrapping_add(0xffff_fff0),
                (x ^ 0x8000_0000_0000_0001) as u32,
                ((x ^ 0x8000_0000_0000_0001) >> 32) as u32,
                hi,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}: x = {x:#x}, y = {y:#x}");
        }
    }
    Case { name: "inline_ptx_named", kernel, cfg: launch(1), small_only: false, check }
}

/// Raw inline assembly with constraint strings: a `+r` read-write operand
/// beside an `=r` output, an input tied to an output by number (in a string
/// written with spaces, which are dropped), an immediate (`n`) operand,
/// Float64 (`d`) operands, and a memory clobber with side effects.
pub fn inline_asm_raw() -> Case {
    const N32: usize = 3;
    const N64: usize = 2;
    let kernel = Kernel::builder("inline_asm_raw").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let ds = param_tensor::<f64>("ds", Layout::col_major(T));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let t = thread_idx().x;
        let (x, l, d) = (xs.load(t) * SPREAD as u32, ls.load(t) * SPREAD, ds.load(t));
        let y = t.bitcast::<u32>() * 0x0101_0101u32;
        let (sum, product) = InlineAsm::new("add.u32 $0, $0, $2;\n\tmul.lo.u32 $1, $2, $2;", "+r,=r,r")
            .output::<u32>()
            .output::<u32>()
            .arg(x)
            .arg(y)
            .sim(|r, rw| {
                rw[0] = (rw[0] + r[0]) & 0xffff_ffff;
                vec![(r[0] * r[0]) & 0xffff_ffff]
            })
            .emit();
        let difference: Uint32 = InlineAsm::new("sub.u32 $0, $0, $1;", "=r, r, 0")
            .output::<u32>()
            .arg(y)
            .arg(x)
            .sim(|r, rw| {
                rw[0] = rw[0].wrapping_sub(r[0]) & 0xffff_ffff;
                vec![]
            })
            .emit();
        let shifted: Uint64 = InlineAsm::new("shl.b64 $0, $1, $2;", "=l,l,n")
            .output::<u64>()
            .arg(l)
            .arg(7u32)
            .sim(|r, _| vec![r[0] << 7])
            .emit();
        let fused: Float64 = InlineAsm::new("fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d")
            .output::<f64>()
            .arg(d)
            .arg(d)
            .arg(2.0f64)
            .side_effects()
            .memory_clobber()
            .sim(|r, _| vec![f64::from_bits(r[0]).mul_add(f64::from_bits(r[1]), 2.0).to_bits()])
            .emit();
        for (j, v) in [sum, product, difference].into_iter().enumerate() {
            out32.store((t, j), v);
        }
        for (j, v) in [shifted, fused.bitcast()].into_iter().enumerate() {
            out64.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ls, ds) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>(), args[2].to_vec::<f64>());
        let (out32, out64) = (args[3].to_vec::<u32>(), args[4].to_vec::<u64>());
        for t in 0..T {
            let (x, l, d) = (xs[t].wrapping_mul(SPREAD as u32), ls[t].wrapping_mul(SPREAD), ds[t]);
            let y = (t as u32).wrapping_mul(0x0101_0101);
            let want = [x.wrapping_add(y), y.wrapping_mul(y), x.wrapping_sub(y)];
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "thread {t}");
            let want = [l << 7, d.mul_add(d, 2.0).to_bits()];
            assert_eq!(out64[t * N64..(t + 1) * N64], want[..], "thread {t}");
        }
    }
    Case { name: "inline_asm_raw", kernel, cfg: launch(1), small_only: false, check }
}

/// Hand-written MLIR operations: LLVM's byte swap and funnel shift and
/// `arith.mului_extended` (two results), each simulated by a closure.
pub fn mlir_ops() -> Case {
    const COLS: usize = 4;
    let kernel = Kernel::builder("mlir_ops").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let (x, y) = (xs.load(t) * SPREAD as u32, ys.load(t) * SPREAD as u32);
        let swapped: Uint32 = MlirOp::new("llvm.intr.bswap(%0) : (i32) -> i32")
            .operand(x)
            .result::<u32>()
            .pure()
            .sim(|r, _| vec![(r[0] as u32).swap_bytes() as u64])
            .emit();
        let (low, high) = MlirOp::new("arith.mului_extended %0, %1 : i32")
            .operand(x)
            .operand(y)
            .result::<u32>()
            .result::<u32>()
            .pure()
            .sim(|r, _| vec![(r[0] * r[1]) & 0xffff_ffff, (r[0] * r[1]) >> 32])
            .emit();
        let funnel: Uint32 = MlirOp::new("llvm.intr.fshl(%1, %0, %2) : (i32, i32, i32) -> i32")
            .operand(x)
            .operand(y)
            .operand(t.bitcast::<u32>())
            .result::<u32>()
            .sim(|r, _| vec![(((r[1] << 32 | r[0]) << (r[2] % 32)) >> 32) & 0xffff_ffff])
            .emit();
        for (j, v) in [swapped, low, high, funnel].into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let (x, y) = (xs[t].wrapping_mul(SPREAD as u32), ys[t].wrapping_mul(SPREAD as u32));
            let product = x as u64 * y as u64;
            let funnel = (((y as u64) << 32 | x as u64) << (t % 32) >> 32) as u32;
            let want = [x.swap_bytes(), product as u32, (product >> 32) as u32, funnel];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}");
        }
    }
    Case { name: "mlir_ops", kernel, cfg: launch(1), small_only: false, check }
}

/// Inline PTX with pointer operands, which the simulator cannot follow, so
/// the GPU is held to plain Rust instead: CuTe DSL's example global load
/// (`ld.global.nc`), a load from another thread's shared memory through a
/// shared-state-space address, a global store guarded by a predicate (odd
/// lanes only: the even lanes keep the sentinel an ordinary store put
/// there first), raw assembly with a memory clobber that writes shared
/// memory between an ordinary store and an ordinary load of it, which must
/// see its write, a generic load from a per-thread array, the distance
/// between two shared addresses computed in 32 bits (which `ptxas` accepts
/// only if they are 32-bit registers, as in CuTe DSL), a global load at an
/// immediate offset (`[{$r0}+{$r1}]` with a constant), and raw assembly
/// whose memory clobber is written in its constraint string.
pub fn inline_ptx_pointers() -> Case {
    const COLS: usize = 8;
    const SENTINEL: u64 = 0x5e17_1ee1_0dd0_c0de;
    let kernel = Kernel::builder("inline_ptx_pointers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let wide = param_tensor::<u64>("wide", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let loaded: Uint32 =
            InlinePtx::new("ld.global.nc.u32 {$w0}, [{$r0}];").write::<u32>().read_ptr(&xs.ptr().offset(t)).emit();
        let smem = alloc_smem::<u64>(T, 16);
        make_tensor(&smem, Layout::col_major(T)).store(t, ls.load(t));
        sync_threads();
        let neighbour: Uint64 = InlinePtx::new("ld.shared.u64 {$w0}, [{$r0}];")
            .write::<u64>()
            .read_ptr(&smem.offset((t + 5) % T as i32))
            .emit();
        wide.store(t, SENTINEL);
        InlinePtx::new("st.global.u64 [{$r0}], {$r1};")
            .read_ptr(&wide.ptr().offset(t))
            .read(neighbour + 1u64)
            .predicate((t & 1).eq(1))
            .emit();
        let words = alloc_smem::<u32>(T, 16);
        let tile = make_tensor(&words, Layout::col_major(T));
        tile.store(t, loaded);
        InlineAsm::new("st.shared.u32 [$0], $1;", "l,r")
            .arg_ptr(&words.offset(t))
            .arg(loaded * 3u32)
            .memory_clobber()
            .emit();
        let reread = tile.load(t);
        let table = make_rmem_tensor::<u32>(Layout::col_major(4));
        for_range(0, 4, 1, |i| table.store(i, loaded + i.to::<u32>() * 100u32));
        let picked: Uint32 =
            InlinePtx::new("ld.u32 {$w0}, [{$r0}];").write::<u32>().read_ptr(&table.ptr().offset(t % 4)).emit();
        let step: Uint32 = InlinePtx::new("sub.u32 {$w0}, {$r1}, {$r0};")
            .write::<u32>()
            .read_ptr(&smem.offset(t))
            .read_ptr(&smem.offset(t ^ 1))
            .emit();
        let next: Uint32 = InlinePtx::new("ld.global.nc.u32 {$w0}, [{$r0}+{$r1}];")
            .write::<u32>()
            .read_ptr(&xs.ptr().offset(t - (t & 1)))
            .read(4i32)
            .emit();
        let marks = alloc_smem::<u32>(T, 16);
        let marked = make_tensor(&marks, Layout::col_major(T));
        marked.store(t, loaded);
        InlineAsm::new("st.shared.u32 [$0], $1;", "r,r,~{memory}")
            .arg_ptr(&marks.offset(t))
            .arg(loaded ^ 0x55u32)
            .emit();
        let remarked = marked.load(t);
        let cols =
            [loaded, neighbour.to::<u32>(), (neighbour >> 32u64).to::<u32>(), reread, picked, step, next, remarked];
        for (j, v) in cols.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ls, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>(), args[3].to_vec::<u32>());
        let wide = args[2].to_vec::<u64>();
        for t in 0..T {
            let n = ls[(t + 5) % T];
            let step = ((t ^ 1) as i32 - t as i32) * 8;
            let want = [
                xs[t],
                n as u32,
                (n >> 32) as u32,
                xs[t].wrapping_mul(3),
                xs[t].wrapping_add(t as u32 % 4 * 100),
                step as u32,
                xs[t / 2 * 2 + 1],
                xs[t] ^ 0x55,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t}");
            let stored = if t % 2 == 1 { n.wrapping_add(1) } else { SENTINEL };
            assert_eq!(wide[t], stored, "thread {t}: the predicated store");
        }
    }
    Case { name: "inline_ptx_pointers", kernel, cfg: launch(1), small_only: false, check }
}

/// CuTe DSL's example of raw MLIR, `nvvm.read.ptx.sreg.nsmid`, beside
/// cute-rs's own `nsmid()`: the GPU must give the same positive number for
/// both. The simulator cannot run the hand-written operation.
pub fn mlir_nsmid() -> Case {
    let kernel = Kernel::builder("mlir_nsmid").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::row_major((T, 2)));
        let t = thread_idx().x;
        let sms: Int32 = MlirOp::new("nvvm.read.ptx.sreg.nsmid : i32").result::<i32>().emit();
        out.store((t, 0), sms);
        out.store((t, 1), nsmid());
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<i32>();
        for t in 0..T {
            assert!(
                out[2 * t] > 0 && out[2 * t] == out[2 * t + 1],
                "thread {t}: {} and {}",
                out[2 * t],
                out[2 * t + 1]
            );
        }
    }
    Case { name: "mlir_nsmid", kernel, cfg: launch(1), small_only: false, check }
}

/// Hand-written MLIR operations on pointers, which the simulator cannot
/// follow, so the GPU is held to plain Rust: a global load through an
/// `!llvm.ptr<1>`, a 4-byte `nvvm.cp.async` from global into shared memory
/// (`!llvm.ptr<3>` and `!llvm.ptr<1>`) with its commit and wait written as
/// MLIR too, read back after a barrier by an ordinary load of another
/// thread's element, a global store through a pointer, and a load from a
/// per-thread array through a generic `!llvm.ptr`.
pub fn mlir_pointers() -> Case {
    const COLS: usize = 4;
    let kernel = Kernel::builder("mlir_pointers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let loaded: Uint32 =
            MlirOp::new("llvm.load %0 : !llvm.ptr<1> -> i32").operand_ptr(&xs.ptr().offset(t)).result::<u32>().emit();
        let smem = alloc_smem::<u32>(T, 16);
        MlirOp::new("nvvm.cp.async.shared.global %0, %1, 4, cache = ca : !llvm.ptr<3>, !llvm.ptr<1>")
            .operand_ptr(&smem.offset(t))
            .operand_ptr(&ys.ptr().offset(t))
            .emit();
        MlirOp::new("nvvm.cp.async.commit.group").emit();
        MlirOp::new("nvvm.cp.async.wait.group 0").emit();
        sync_threads();
        let neighbour = make_tensor(&smem, Layout::col_major(T)).load((t + 5) % T as i32);
        MlirOp::new("llvm.store %1, %0 : i32, !llvm.ptr<1>")
            .operand_ptr(&out.ptr().offset(t * COLS as i32 + 2))
            .operand(loaded ^ 0x5a5a_5a5au32)
            .emit();
        let table = make_rmem_tensor::<u32>(Layout::col_major(4));
        for_range(0, 4, 1, |i| table.store(i, loaded + i.to::<u32>() * 100u32));
        let picked: Uint32 = MlirOp::new("llvm.load %0 : !llvm.ptr -> i32")
            .operand_ptr(&table.ptr().offset(t % 4))
            .result::<u32>()
            .emit();
        out.store((t, 0), loaded);
        out.store((t, 1), neighbour);
        out.store((t, 3), picked);
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let want = [xs[t], ys[(t + 5) % T], xs[t] ^ 0x5a5a_5a5a, xs[t].wrapping_add(t as u32 % 4 * 100)];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t}");
        }
    }
    Case { name: "mlir_pointers", kernel, cfg: launch(1), small_only: false, check }
}

/// What the GPU does with a carry flag that one kind of instruction sets
/// and the other reads, which the validator rejects for the carry
/// instructions and which inline PTX holding the whole chain can still
/// write. The PTX ISA describes one flag for carries and borrows, which
/// would make `subc` after `add.cc` subtract the carry and `addc` after
/// `sub.cc` add the borrow; the GPU (an SM89 with `ptxas` 12.8) keeps a
/// borrow as the inverted carry instead, so they subtract one minus the
/// carry and add one minus the borrow. This holds the GPU to that, so that a
/// GPU or `ptxas` that changes it shows here.
pub fn mixed_carry_flag() -> Case {
    let kernel = Kernel::builder("mixed_carry_flag").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 2)));
        let t = thread_idx().x;
        let (x, y, flag) = (xs.load(t) * SPREAD as u32, ys.load(t) * SPREAD as u32, (t & 1).bitcast::<u32>());
        // add.cc carries exactly when flag is 1; sub.cc borrows exactly then.
        let after_add: Uint32 =
            InlinePtx::new("{ .reg .u32 t; add.cc.u32 t, {$r0}, 0xffffffff; subc.u32 {$w0}, {$r1}, {$r2}; }")
                .write::<u32>()
                .read(flag)
                .read(x)
                .read(y)
                .emit();
        let after_sub: Uint32 =
            InlinePtx::new("{ .reg .u32 t; sub.cc.u32 t, 0, {$r0}; addc.u32 {$w0}, {$r1}, {$r2}; }")
                .write::<u32>()
                .read(flag)
                .read(x)
                .read(y)
                .emit();
        out.store((t, 0), after_add);
        out.store((t, 1), after_sub);
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let (x, y, flag) = (xs[t].wrapping_mul(SPREAD as u32), ys[t].wrapping_mul(SPREAD as u32), t as u32 & 1);
            let want = [x.wrapping_sub(y).wrapping_sub(1 - flag), x.wrapping_add(y).wrapping_add(1 - flag)];
            assert_eq!(out[2 * t..2 * t + 2], want, "thread {t}: the flag read across kinds (x = {x:#x}, y = {y:#x})");
        }
    }
    Case { name: "mixed_carry_flag", kernel, cfg: launch(1), small_only: false, check }
}

// Carry chains through the special-register reads and cp.async.mbarrier.arrive.

/// Output rows of [`carry_around_smem_kernel`]: one per thread of its two blocks.
const SMEM_CARRY_ROWS: usize = 2 * T;

/// Words of each of [`carry_around_smem_kernel`]'s copies: 16 bytes per thread.
const SMEM_CARRY_WORDS: usize = 4 * T;

/// The bytes [`carry_around_smem_kernel`]'s launches give its dynamic region: one copy's.
const SMEM_CARRY_REGION_BYTES: u32 = 4 * SMEM_CARRY_WORDS as u32;

/// Where [`carry_around_smem_kernel`]'s dynamic region starts: after its two mbarriers and its
/// buffer sized at build time (16 + 1024 bytes).
pub const SMEM_CARRY_REGION_START: u32 = 16 + 4 * SMEM_CARRY_WORDS as u32;

/// Words per row of [`carry_around_smem_kernel`]'s `out32`: the three size reads, four words of
/// each copy and, with `cluster`, the block's rank, the cluster's size and `aggr_smem_size`.
fn smem_carry_columns(cluster: bool) -> usize {
    3 + 4 + 4 + if cluster { 3 } else { 0 }
}

/// Carry chains through the special-register reads and `cp.async.mbarrier.arrive`, which keep
/// the carry flag like the warp collectives (see `cute::ir::carry_flag_breaker`), so the
/// validator accepts a chain around them and the simulator keeps the flag. Two mbarriers and a
/// 1 KB buffer sized at build time come first in shared memory, and the dynamic region follows
/// them (at [`SMEM_CARRY_REGION_START`]). Each thread copies 16 bytes into the buffer with
/// `cp.async`, and then 16 bytes into the region. Between a carry-out and its carry-in sit
/// `dynamic_smem_size` (a `mov` and a subtraction of the region's start), `get_dyn_smem_size`
/// (the same read, which the builder merges with it), `total_smem_size` and
/// `cp_async_mbarrier_arrive_noinc`, which makes the first mbarrier track the first copy, in the
/// 128-bit sum of numbers held as two u64 words; and `cp_async_mbarrier_arrive`, which makes the
/// second track both copies, in their difference, with `block_idx_in_cluster`, `cluster_size` and
/// `aggr_smem_size` (sm_90) too when `cluster` is set. The odd threads carry and borrow out of the
/// low word. After the chains each thread arrives on the second mbarrier, waits on both (giving
/// up after [`SPIN_LIMIT`] polls, counted in `gave_up`) and reads the other warp's copies.
pub fn carry_around_smem_kernel(cluster: bool) -> Kernel {
    const W: usize = SMEM_CARRY_WORDS;
    let name = if cluster { "carry_around_smem_registers_sm90" } else { "carry_around_smem_registers" };
    Kernel::builder(name).block_dim([T as u32, 1, 1]).build(move || {
        let xs = param_tensor::<u64>("xs", Layout::row_major((SMEM_CARRY_ROWS, 4)));
        let src = param_tensor::<u32>("src", Layout::col_major(2 * W));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((SMEM_CARRY_ROWS, 5)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((SMEM_CARRY_ROWS, smem_carry_columns(cluster))));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(SMEM_CARRY_ROWS));
        let t = thread_idx().x;
        let row = block_idx().x * T as i32 + t;
        let bars = alloc_smem::<u64>(2, 8);
        let (full, tracked) = (bars.clone(), bars.offset(1usize));
        let stage = make_tensor(&alloc_smem::<u32>(W, 16), Layout::col_major(W));
        let region = make_tensor(&get_dyn_smem::<u32>(None), Layout::col_major(W));
        if_(t.eq(0), || {
            mbarrier_init(&full, T as i32);
            mbarrier_init(&tracked, T as i32);
        });
        sync_threads();
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((row, i)) * SPREAD);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(T), &Layout::col_major(4));
        let thr = tiled.get_slice(t);
        let source = |half: usize| make_tensor(&src.ptr().offset(half * W), Layout::col_major(W));
        copy(&tiled, &thr.partition_s(&source(0)), &thr.partition_d(&stage));
        // The sum, around the size registers and the first copy's arrival.
        let low = add_cc(a0, b0);
        let dynamic = dynamic_smem_size();
        cp_async_mbarrier_arrive_noinc(&full);
        let region_bytes = get_dyn_smem_size();
        let high = addc_cc(a1, b1);
        let total = total_smem_size();
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        copy(&tiled, &thr.partition_s(&source(1)), &thr.partition_d(&region));
        // The difference, around the arrival that tracks both copies (and the cluster registers).
        let d0 = sub_cc(b0, a0);
        cp_async_mbarrier_arrive(&tracked);
        let cluster_registers = cluster.then(|| [block_idx_in_cluster(), cluster_size(), aggr_smem_size()]);
        let d1 = subc(b1, a1);
        mbarrier_arrive(&tracked);
        let misses = Var::new(Uint32::from(0u32));
        for bar in [&full, &tracked] {
            let missed = bounded_spin(bar, Int32::from(0));
            misses.update(|m| m + select(missed, 1u32, 0u32));
        }
        gave_up.store(row, misses.get());
        for (j, v) in [low, high, top, d0, d1].into_iter().enumerate() {
            out64.store((row, j), v);
        }
        // The other warp's copies.
        let o = (t + 32) % 64;
        let mut words = vec![dynamic.bitcast::<u32>(), region_bytes.bitcast::<u32>(), total.bitcast::<u32>()];
        words.extend((0..4).map(|k| stage.load(o * 4 + k)));
        words.extend((0..4).map(|k| region.load(o * 4 + k)));
        words.extend(cluster_registers.into_iter().flatten().map(|v| v.bitcast::<u32>()));
        for (j, v) in words.into_iter().enumerate() {
            out32.store((row, j), v);
        }
    })
}

/// The plain-Rust result of [`carry_around_smem_kernel`] (in clusters of 2 with `cluster`): the
/// sum and difference in u128 arithmetic, the region's bytes, the block's shared memory
/// rounded up to 128 bytes (and 1 KB more for `aggr_smem_size`, as in
/// [`check_cluster_ranks`]), the copied words and, with `cluster`, the block's rank and the
/// cluster's size.
fn check_carry_around_smem(args: &[KernelArg], cluster: bool) {
    let (xs, src) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>());
    let (out64, out32) = (args[2].to_vec::<u64>(), args[3].to_vec::<u32>());
    check_no_spin_gave_up(&args[4].to_vec::<u32>());
    let n32 = smem_carry_columns(cluster);
    let total = (SMEM_CARRY_REGION_START + SMEM_CARRY_REGION_BYTES).next_multiple_of(128);
    let (mut carries, mut borrows, mut top_carries) = (0, 0, 0);
    for r in 0..SMEM_CARRY_ROWS {
        let (block, t) = (r / T, r % T);
        let w = |i: usize| xs[r * 4 + i].wrapping_mul(SPREAD);
        let odd = t % 2 == 1;
        let (a0, a1) = (if odd { u64::MAX } else { w(0) }, w(1));
        let (b0, b1) = (if odd { w(2) | 1 } else { w(2) }, w(3));
        let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
        let (sum, carry) = a.overflowing_add(b);
        let difference = b.wrapping_sub(a);
        carries += a0.overflowing_add(b0).1 as usize;
        borrows += (b0 < a0) as usize;
        top_carries += carry as usize;
        let want = [sum as u64, (sum >> 64) as u64, carry as u64, difference as u64, (difference >> 64) as u64];
        assert_eq!(out64[r * 5..r * 5 + 5], want, "row {r}: a = {a:#x}, b = {b:#x}");
        let o = (t + 32) % T;
        let mut want = vec![SMEM_CARRY_REGION_BYTES, SMEM_CARRY_REGION_BYTES, total];
        want.extend(&src[o * 4..o * 4 + 4]);
        want.extend(&src[SMEM_CARRY_WORDS + o * 4..SMEM_CARRY_WORDS + o * 4 + 4]);
        if cluster {
            // `%aggr_smem_size` is not rounded to the allocation unit (a B200 gives this).
            want.extend([block as u32 % 2, 2, SMEM_CARRY_REGION_START + SMEM_CARRY_REGION_BYTES + 1024]);
        }
        assert_eq!(out32[r * n32..(r + 1) * n32], want[..], "row {r}");
    }
    assert!(
        carries >= SMEM_CARRY_ROWS / 2 && borrows >= SMEM_CARRY_ROWS / 2,
        "the odd threads should carry and borrow: {carries}, {borrows}"
    );
    // The carry out of the high word is the one that crosses total_smem_size: keep it exercised.
    assert!(top_carries > 0, "no row carries out of the high word, so the chain through total_smem_size goes untested");
}

/// [`carry_around_smem_kernel`] on two blocks with a region of [`SMEM_CARRY_REGION_BYTES`].
pub fn carry_around_smem_registers() -> Case {
    let cfg = launch(2).with_dynamic_smem(SMEM_CARRY_REGION_BYTES);
    let check = |args: &[KernelArg]| check_carry_around_smem(args, false);
    Case { name: "carry_around_smem_registers", kernel: carry_around_smem_kernel(false), cfg, small_only: false, check }
}

/// [`carry_around_smem_kernel`] with the cluster registers and `aggr_smem_size` (sm_90), on two
/// blocks in a cluster of 2.
pub fn carry_around_smem_registers_sm90() -> Case {
    let cfg = launch(2).with_cluster([2, 1, 1]).with_dynamic_smem(SMEM_CARRY_REGION_BYTES);
    let check = |args: &[KernelArg]| check_carry_around_smem(args, true);
    let kernel = carry_around_smem_kernel(true);
    Case { name: "carry_around_smem_registers_sm90", kernel, cfg, small_only: false, check }
}

// ---------------------------------------------------------------------------
// Carry chains through loads and stores with memory attributes, and prefetches.
// ---------------------------------------------------------------------------

/// Words per thread that [`carry_around_memory_accesses`] loads (`ys`) and stores (`out32`).
const MEM_CARRY_WORDS: usize = 8;

/// Carry chains through loads and stores with memory attributes and through prefetches, which
/// keep the carry flag like the warp collectives (see `cute::ir::carry_flag_breaker`), so the
/// validator accepts a chain around them and the simulator keeps the flag. Between a carry-out
/// and its carry-in sit, in the 128-bit sum of numbers held as two u64 words: an acquire load
/// and a release store of global memory and a relaxed load and a release store of the thread's
/// own word of shared memory (LLVM atomics with a sync scope), a seq_cst load (`fence.sc` and an
/// acquire load), an L2 prefetch with an eviction priority, a load with an L1 eviction priority
/// and an L2 cache policy (inline PTX), a volatile load (an LLVM volatile load), a uniform
/// prefetch and a plain load and store; and in their difference: an acquire vector of eight
/// bytes and a relaxed one (each two instructions of four), a non-coherent load, a predicated
/// prefetch, a streaming store and a volatile one. The odd threads carry and borrow out of the
/// low word.
pub fn carry_around_memory_accesses() -> Case {
    const W: usize = MEM_CARRY_WORDS;
    let kernel = Kernel::builder("carry_around_memory_accesses").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, W)));
        let bs = param_tensor::<u8>("bs", Layout::col_major(T * 8));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, 5)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, W)));
        let ob = param_tensor::<u8>("ob", Layout::col_major(T * 8));
        let t = thread_idx().x;
        let own = make_smem_tensor::<u32>(Layout::col_major(T)).ptr().offset(t);
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| vec![0]);
        own.store(ys.load((t, 6)));
        let (l, s, p) = (LoadOptions::default(), StoreOptions::default(), PrefetchOptions::default());
        let (gpu, cta) = (Some(MemoryScope::Gpu), Some(MemoryScope::Cta));
        let y = |j: usize| ys.ptr().offset(IndexExpr::from(t) * W + j);
        let o = |j: usize| out32.ptr().offset(IndexExpr::from(t) * W + j);
        let byte_row = |b: &Tensor<u8>| b.ptr().offset(IndexExpr::from(t) * 8);
        // The sum.
        let low = add_cc(a0, b0);
        let acquired = load(&y(0), LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l });
        store(&o(0), acquired ^ 0x0101_0101u32, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let relaxed = load(&own, LoadOptions { sem: MemoryOrder::Relaxed, scope: cta, ..l });
        store(&own, relaxed + 1u32, StoreOptions { sem: MemoryOrder::Release, scope: cta, ..s });
        let sequential = load(&y(1), LoadOptions { sem: MemoryOrder::Sc, scope: gpu, ..l });
        let evict_last = Some(CacheEvictionPriority::EvictLast);
        prefetch(&y(2), PrefetchOptions { cache_level: Some(PrefetchCacheLevel::L2), evict_priority: evict_last, ..p });
        let high = addc_cc(a1, b1);
        let hinted = load(&y(2), LoadOptions { level1_eviction_priority: evict_last, cache_policy: Some(policy), ..l });
        let volatile = load(&y(3), LoadOptions { sem: MemoryOrder::Volatile, ..l });
        prefetch(&y(4), PrefetchOptions { uniform: true, ..p });
        let plain = ys.load((t, 4));
        out32.store((t, 4), plain);
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        // The difference.
        let d0 = sub_cc(b0, a0);
        let bytes = load_vector::<u8, 8>(&byte_row(&bs), LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l });
        store_vector(&byte_row(&ob), bytes, StoreOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..s });
        let non_coherent = load(&y(5), LoadOptions { sem: MemoryOrder::Constant, ..l });
        prefetch(&y(7), PrefetchOptions { predicate: Some(odd), ..p });
        store(&o(1), sequential, StoreOptions { cop: Some(CacheOperator::Cs), ..s });
        store(&o(2), hinted, StoreOptions { sem: MemoryOrder::Volatile, ..s });
        let d1 = subc(b1, a1);
        for (j, v) in [low, high, top, d0, d1].into_iter().enumerate() {
            out64.store((t, j), v);
        }
        out32.store((t, 3), volatile);
        out32.store((t, 5), non_coherent);
        out32.store((t, 6), own.load());
        out32.store((t, 7), ys.load((t, 7)));
    });
    fn check(args: &[KernelArg]) {
        const W: usize = MEM_CARRY_WORDS;
        let (xs, ys, bs) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>(), args[2].to_vec::<u8>());
        let (out64, out32, ob) = (args[3].to_vec::<u64>(), args[4].to_vec::<u32>(), args[5].to_vec::<u8>());
        let (mut carries, mut borrows, mut top_carries) = (0, 0, 0);
        for t in 0..T {
            let w = |i: usize| xs[t * 4 + i].wrapping_mul(SPREAD);
            let odd = t % 2 == 1;
            let (a0, a1) = (if odd { u64::MAX } else { w(0) }, w(1));
            let (b0, b1) = (if odd { w(2) | 1 } else { w(2) }, w(3));
            let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
            let (sum, carry) = a.overflowing_add(b);
            let difference = b.wrapping_sub(a);
            carries += a0.overflowing_add(b0).1 as usize;
            borrows += (b0 < a0) as usize;
            top_carries += carry as usize;
            let want = [sum as u64, (sum >> 64) as u64, carry as u64, difference as u64, (difference >> 64) as u64];
            assert_eq!(out64[t * 5..t * 5 + 5], want, "thread {t}: a = {a:#x}, b = {b:#x}");
            let y = |j: usize| ys[t * W + j];
            let want = [y(0) ^ 0x0101_0101, y(1), y(2), y(3), y(4), y(5), y(6).wrapping_add(1), y(7)];
            assert_eq!(out32[t * W..(t + 1) * W], want, "thread {t}");
            assert_eq!(ob[t * 8..t * 8 + 8], bs[t * 8..t * 8 + 8], "thread {t}");
        }
        assert!(carries >= T / 2 && borrows >= T / 2, "the odd threads should carry and borrow: {carries}, {borrows}");
        // The carry out of the high word crosses the hinted and volatile loads: keep it exercised.
        assert!(top_carries > 0, "no thread carries out of the high word");
    }
    Case { name: "carry_around_memory_accesses", kernel, cfg: launch(1), small_only: false, check }
}

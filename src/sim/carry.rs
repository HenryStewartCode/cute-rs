//! Semantics of [`Op::CarryInst`](crate::ir::Op::CarryInst): PTX's
//! extended-precision instructions, written from the PTX ISA's definitions
//! (section 9.7.2), not from the code the backend emits.
//!
//! The ISA's pseudo-code, for `w`-bit operands:
//!
//! ```text
//! add.cc    d = a + b;                   carry-out written to CC.CF
//! addc{.cc} d = a + b + CC.CF;           if .cc, carry-out written to CC.CF
//! sub.cc    d = a - b;                   borrow-out written to CC.CF
//! subc{.cc} d = a - (b + CC.CF);         if .cc, borrow-out written to CC.CF
//! mad.lo.cc t = a * b; d = t<w-1..0> + c;          carry-out from the addition
//! madc.lo   t = a * b; d = t<w-1..0> + c + CC.CF;  if .cc, carry-out from the addition
//! ```
//!
//! "Behavior is the same for unsigned and signed integers" for the adds and
//! subtracts; for the multiply-adds, the low half of the product is the same
//! for either signedness, and the carry is that of the addition. The carry
//! out of a sum is the bit above the top bit of the exact sum; the borrow
//! out of `a - (b + cin)` is whether the exact difference is negative, both
//! taken over the operands as unsigned numbers.
//!
//! Operands and results are raw bits, zero-extended to 64 like every value
//! in the simulator; `ty` is the operands' type.

use crate::ir::{CarryFn, CarryOp, DType};

/// The result of `func` on `args` (of type `ty`) with the carry flag at
/// `carry_in` (ignored unless `func` reads it), and the carry out (the new
/// flag if `func` writes it).
pub(super) fn carry_inst(func: CarryFn, ty: DType, args: &[u64], carry_in: bool) -> (u64, bool) {
    let w = ty.bits() as u32;
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let cin = u128::from(func.carry_in && carry_in);
    let operand = |k: usize| u128::from(args[k] & mask);
    match func.op {
        CarryOp::Add => {
            let exact = operand(0) + operand(1) + cin;
            ((exact as u64) & mask, exact >> w != 0)
        }
        CarryOp::Sub => {
            let (a, subtrahend) = (operand(0), operand(1) + cin);
            (a.wrapping_sub(subtrahend) as u64 & mask, a < subtrahend)
        }
        CarryOp::MadLo => {
            // t<w-1..0>: the product's low half, the same for signed and
            // unsigned operands.
            let low = operand(0).wrapping_mul(operand(1)) & u128::from(mask);
            let exact = low + operand(2) + cin;
            ((exact as u64) & mask, exact >> w != 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMS: [(bool, bool); 3] = [(false, true), (true, false), (true, true)];

    fn func(op: CarryOp, (carry_in, carry_out): (bool, bool)) -> CarryFn {
        CarryFn { op, carry_in, carry_out }
    }

    /// Operands at the edges of the carry: zero, one, the top bit, all ones.
    fn patterns(w: u32) -> Vec<u64> {
        let top = 1u64 << (w - 1);
        let ones = if w == 64 { u64::MAX } else { (1 << w) - 1 };
        let mut v = vec![0, 1, 2, top - 1, top, top + 1, ones - 1, ones, 0x1234_5678_9abc_def0 & ones];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..40 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            v.push(state & ones);
        }
        v
    }

    /// Adds and subtracts against i128 arithmetic: a multi-word sum built
    /// from them, word by word, is the sum of the whole numbers.
    #[test]
    fn adds_and_subtracts_carry_like_wide_arithmetic() {
        for (ty, w) in [(DType::U32, 32u32), (DType::I32, 32), (DType::U64, 64), (DType::I64, 64)] {
            let base = 1i128 << w;
            for &a in &patterns(w) {
                for &b in &patterns(w) {
                    for form in FORMS {
                        for cin in [false, true] {
                            let c = i128::from(form.0 && cin);
                            let (s, carry) = carry_inst(func(CarryOp::Add, form), ty, &[a, b], cin);
                            let exact = a as i128 + b as i128 + c;
                            assert_eq!((s as i128, carry), (exact % base, exact >= base), "{a:#x} + {b:#x} + {c}");
                            let (d, borrow) = carry_inst(func(CarryOp::Sub, form), ty, &[a, b], cin);
                            let exact = a as i128 - b as i128 - c;
                            assert_eq!(
                                (d as i128, borrow),
                                (exact.rem_euclid(base), exact < 0),
                                "{a:#x} - {b:#x} - {c}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The multiply-adds: the low half of the product (the same for either
    /// signedness) plus the addend and the carry-in, and its carry.
    #[test]
    fn multiply_adds_carry_out_of_the_addition() {
        for (ty, w) in [(DType::U32, 32u32), (DType::I32, 32), (DType::U64, 64), (DType::I64, 64)] {
            let base = 1u128 << w;
            let ps = patterns(w);
            for &a in &ps {
                for &b in ps.iter().step_by(3) {
                    for &c in ps.iter().step_by(5) {
                        for form in FORMS {
                            for cin in [false, true] {
                                let low = (a as u128 * b as u128) % base;
                                let exact = low + c as u128 + u128::from(form.0 && cin);
                                let got = carry_inst(func(CarryOp::MadLo, form), ty, &[a, b, c], cin);
                                assert_eq!(got, ((exact % base) as u64, exact >= base), "{a:#x} * {b:#x} + {c:#x}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// A carry-out-only instruction ignores the flag it is given.
    #[test]
    fn carry_out_forms_ignore_the_incoming_flag() {
        for op in [CarryOp::Add, CarryOp::Sub, CarryOp::MadLo] {
            let f = func(op, (false, true));
            assert_eq!(carry_inst(f, DType::U32, &[7, 9, 11], true), carry_inst(f, DType::U32, &[7, 9, 11], false));
        }
        // 0 - (0xffffffff + 1) borrows, and wraps to 0.
        let subc = func(CarryOp::Sub, (true, true));
        assert_eq!(carry_inst(subc, DType::U32, &[0, 0xffff_ffff], true), (0, true));
        // 0xffffffff + 0xffffffff + 1 carries, leaving 0xffffffff.
        let addc = func(CarryOp::Add, (true, true));
        assert_eq!(carry_inst(addc, DType::U32, &[0xffff_ffff, 0xffff_ffff], true), (0xffff_ffff, true));
    }
}

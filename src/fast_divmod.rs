// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cutlass/fast_math.h,
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The ported portions are under the BSD-3-Clause licence.
// THIRD-PARTY-NOTICES.md has the licence text and lists what each part ports.

//! Integer division by a runtime divisor without a divide instruction.
//!
//! [`FastDivmod`] precomputes, for a divisor `d`, a 32-bit multiplier and a
//! shift such that `n / d` is the high half of `n * multiplier` shifted
//! right, for every `0 <= n < 2^31` (CUTLASS's `FastDivmod`). A quotient
//! then costs a `mul.hi` and a shift instead of the dozens of instructions
//! a 32-bit division takes on the GPU. CuTe DSL's
//! `cute.fast_divmod_create_divisor_v2` makes the same object.
//!
//! The precomputation needs a 64-bit division, so it is best done once: on
//! the host ([`FastDivmod::host`], passed with [`param_fast_divmod`] and
//! [`KernelArg::fast_divmod`]), or in the kernel ([`FastDivmod::new`]) where
//! the divisor is only known there.

use std::ops::{Div, Rem};

use crate::sim::KernelArg;
use crate::value::{Int32, IntoValue, Uint32, Uint64, select};

/// `ceil(log2(x))` for `x > 0` (CUTLASS's `find_log2`).
fn find_log2(x: u32) -> u32 {
    (31 - x.leading_zeros()) + u32::from(!x.is_power_of_two())
}

/// A divisor prepared for fast division of non-negative `Int32`s below
/// `2^31` (CUTLASS's `FastDivmod`, CuTe DSL's `FastDivmodDivisorV2`): `n /
/// d`, `n % d` and [`divmod`](FastDivmod::divmod) on an [`Int32`] use it.
///
/// For `d > 1`, with `p = 31 + ceil(log2(d))`, the multiplier is
/// `ceil(2^p / d)` and the shift `p - 32`, and `n / d` is
/// `mul_hi(n, multiplier) >> shift` (unsigned). For `d == 1` the quotient is
/// `n` itself. The divisor must be positive.
#[derive(Clone, Copy, Debug)]
pub struct FastDivmod {
    /// The divisor, readable in the kernel.
    pub divisor: Int32,
    /// `ceil(2^p / divisor)` (0 for a divisor of 1).
    pub multiplier: Uint32,
    /// `p - 32` (0 for a divisor of 1).
    pub shift: Uint32,
}

/// What [`FastDivmod`] holds, computed on the host ([`FastDivmod::host`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostFastDivmod {
    pub divisor: i32,
    pub multiplier: u32,
    pub shift: u32,
}

impl HostFastDivmod {
    /// `(n / divisor, n % divisor)` by the multiply-and-shift formula, as the
    /// kernel computes it, for `0 <= n < 2^31`.
    pub fn divmod(&self, n: i32) -> (i32, i32) {
        let q = if self.divisor == 1 {
            n
        } else {
            ((((n as u32 as u64) * self.multiplier as u64) >> 32) as u32 >> self.shift) as i32
        };
        (q, n.wrapping_sub(q.wrapping_mul(self.divisor)))
    }

    /// The three scalar arguments [`param_fast_divmod`] declares, in order.
    pub fn kernel_args(&self) -> [KernelArg; 3] {
        [KernelArg::scalar(self.divisor), KernelArg::scalar(self.multiplier), KernelArg::scalar(self.shift)]
    }
}

impl FastDivmod {
    /// Prepare `divisor`, which must be positive, on the host.
    pub fn host(divisor: i32) -> HostFastDivmod {
        assert!(divisor > 0, "cute: FastDivmod needs a positive divisor, not {divisor}");
        if divisor == 1 {
            return HostFastDivmod { divisor, multiplier: 0, shift: 0 };
        }
        let d = divisor as u32;
        let p = 31 + find_log2(d);
        let multiplier = (1u64 << p).div_ceil(d as u64);
        HostFastDivmod { divisor, multiplier: multiplier as u32, shift: p - 32 }
    }

    /// Prepare a runtime `divisor`, which must be positive, in the kernel
    /// (CuTe DSL's `cute.fast_divmod_create_divisor_v2`). This takes one
    /// 64-bit unsigned division; [`FastDivmod::host`] with
    /// [`param_fast_divmod`] moves it to the host.
    #[track_caller]
    pub fn new(divisor: impl IntoValue<i32>) -> FastDivmod {
        let divisor = divisor.into_value();
        let d = divisor.bitcast::<u32>();
        // ceil(log2(d)): floor(log2(d)), plus one unless d is a power of two.
        let floor_log2 = Uint32::from(31u32) - d.clz();
        let not_pow2 = select((d & (d - 1u32)).ne(0u32), 1u32, 0u32);
        let p = floor_log2 + not_pow2 + 31u32;
        let d64 = d.to::<u64>();
        let multiplier = ((Uint64::from(1u64) << p.to::<u64>()) + d64 - 1u64) / d64;
        let is_one = d.eq(1u32);
        FastDivmod {
            divisor,
            multiplier: select(is_one, 0u32, multiplier.to::<u32>()),
            shift: select(is_one, 0u32, p - 32u32),
        }
    }

    /// `(n / divisor, n % divisor)` for `0 <= n < 2^31`.
    #[track_caller]
    pub fn divmod(&self, n: impl IntoValue<i32>) -> (Int32, Int32) {
        let n = n.into_value();
        let hi = n.bitcast::<u32>().mul_hi(self.multiplier) >> self.shift;
        let q = select(self.divisor.eq(1), n, hi.bitcast::<i32>());
        (q, n - q * self.divisor)
    }
}

/// Declare the three scalar parameters of a [`FastDivmod`] prepared on the
/// host: `{name}_divisor` (`i32`), `{name}_multiplier` and `{name}_shift`
/// (`u32`). Pass them with [`KernelArg::fast_divmod`] or
/// [`HostFastDivmod::kernel_args`].
#[track_caller]
pub fn param_fast_divmod(name: &str) -> FastDivmod {
    FastDivmod {
        divisor: crate::param_scalar::<i32>(&format!("{name}_divisor")),
        multiplier: crate::param_scalar::<u32>(&format!("{name}_multiplier")),
        shift: crate::param_scalar::<u32>(&format!("{name}_shift")),
    }
}

impl KernelArg {
    /// The three arguments of a [`param_fast_divmod`] for `divisor`, prepared
    /// on the host.
    pub fn fast_divmod(divisor: i32) -> [KernelArg; 3] {
        FastDivmod::host(divisor).kernel_args()
    }
}

impl Div<FastDivmod> for Int32 {
    type Output = Int32;
    #[track_caller]
    fn div(self, d: FastDivmod) -> Int32 {
        d.divmod(self).0
    }
}

impl Rem<FastDivmod> for Int32 {
    type Output = Int32;
    #[track_caller]
    fn rem(self, d: FastDivmod) -> Int32 {
        d.divmod(self).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(d: i32, n: i32) {
        let f = FastDivmod::host(d);
        assert_eq!(f.divmod(n), (n / d, n % d), "{n} divmod {d} ({f:?})");
    }

    /// Every divisor up to 2^16 with dividends around its multiples, and
    /// random divisors up to 2^31 - 1: the multiplier has no off-by-one.
    #[test]
    fn formula_matches_division() {
        for d in 1..=(1i32 << 16) {
            check(d, 0);
            check(d, i32::MAX);
            let top = i32::MAX / d;
            for k in [1, 2, 3, 7, top / 2, top - 1, top].into_iter().filter(|&k| k >= 1) {
                let m = k as i64 * d as i64;
                for n in [m - 1, m, m + 1] {
                    if (0..=i32::MAX as i64).contains(&n) {
                        check(d, n as i32);
                    }
                }
            }
        }
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20000 {
            let d = (next() % i32::MAX as u64) as i32 + 1;
            check(d, 0);
            check(d, i32::MAX);
            check(d, d - 1);
            check(d, d);
            if let Some(n) = d.checked_add(1) {
                check(d, n);
            }
            let k = (next() % (i32::MAX / d) as u64 + 1) as i64;
            let m = k * d as i64;
            for n in [m - 1, m, m + 1] {
                if (0..=i32::MAX as i64).contains(&n) {
                    check(d, n as i32);
                }
            }
            check(d, (next() % (i32::MAX as u64 + 1)) as i32);
        }
    }
}

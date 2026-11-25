//! Symbolic index expressions.
//!
//! The extents and strides of a layout, and the coordinates fed into it, can
//! be static integers or runtime values (thread indices, loop counters,
//! problem sizes known at launch). [`IndexExpr`] lets layout arithmetic run
//! at kernel-build time on a mix of the two, folding and simplifying as it
//! goes. Only what cannot be resolved statically becomes IR, when the
//! expression is materialised.
//!
//! The values of layout arithmetic are non-negative, which is what makes
//! rewrites such as `(x * 16 + y) / 16 == x + y / 16` valid: a runtime value
//! ([`IndexExpr::Dynamic`], what an [`Int32`](crate::Int32) converts to) is
//! assumed to be non-negative, and a difference `a - b` of two such values is
//! only formed where `a >= b` (an extent minus one, say, extents being at
//! least 1).
//!
//! # Signed terms
//!
//! Two CuTe idioms need values that may be negative: shifting a tensor by
//! the K residue `K - bK * k_tiles`, which is zero or negative
//! ([`domain_offset`](crate::domain_offset)), and comparing a coordinate
//! with `-1` ([`elem_less`](crate::elem_less)). So an expression may also
//! hold negative terms: a negation ([`IndexExpr::Neg`], which a negative
//! `i32` or `i64`, or a negative constant [`Int32`](crate::Int32), converts
//! to), and a runtime value of either sign ([`IndexExpr::signed`]).
//! [`IndexExpr::is_non_negative`] says whether an expression is provably
//! non-negative. Sums, differences and products of such terms fold as
//! integers do, and are computed with wrapping (two's complement) adds,
//! subtracts and multiplies, which give the signed result; comparisons of
//! them are signed. The rewrites that need non-negative operands (splitting
//! the terms of a division or a remainder, and the folds that use an upper
//! bound) only ever see such operands: dividing, taking the remainder of,
//! masking or shifting an expression that may be negative is refused when
//! the kernel is built, and no CuTe idiom does it (a negative shift is only
//! added to what the layout arithmetic gives). An exclusive upper bound
//! survives adding a term that is not positive, and divisibility does not
//! depend on the sign, so both facts stay usable.
//!
//! `Display` prints a value that is not static as `?`, as CuTe DSL prints a
//! dynamic integer, with what is known about it in braces: `?{div=8}` for a
//! multiple of 8, `?{i64}` for a 64-bit value. `Debug` prints the
//! expression, with runtime values as `%n`.
//!
//! An expression is computed in 32 bits unless a runtime `i64` value (an
//! [`Int64`](crate::Int64) leaf: a 64-bit stride or extent) or a static value
//! beyond `i32` enters it; then it is computed in 64 bits
//! ([`IndexExpr::width`]). Only the parts that meet a 64-bit leaf are
//! widened, so a thread's position in a tile stays 32-bit arithmetic.

use std::fmt;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Shr, Sub};

use crate::ir::ValueId;

/// An integer expression over static values and runtime `i32` and `i64`
/// values: non-negative, except where it holds a signed term (see the
/// module documentation).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum IndexExpr {
    Static(usize),
    /// A runtime `i32` or `i64` value, assumed non-negative.
    Dynamic(ValueId),
    /// A runtime `i32` or `i64` value that may be negative
    /// ([`IndexExpr::signed`]).
    Signed(ValueId),
    /// `-a`: with a static `a`, a negative constant.
    Neg(Box<IndexExpr>),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    /// `a - b`, formed only where `a >= b`.
    Sub(Box<IndexExpr>, Box<IndexExpr>),
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    Div(Box<IndexExpr>, Box<IndexExpr>),
    Rem(Box<IndexExpr>, Box<IndexExpr>),
    /// The smaller of the two values.
    Min(Box<IndexExpr>, Box<IndexExpr>),
    And(Box<IndexExpr>, Box<IndexExpr>),
    Xor(Box<IndexExpr>, Box<IndexExpr>),
    Shr(Box<IndexExpr>, Box<IndexExpr>),
}

use IndexExpr::Static as S;

fn bx(e: IndexExpr) -> Box<IndexExpr> {
    Box::new(e)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl IndexExpr {
    /// The value, if it is known statically.
    pub fn as_static(&self) -> Option<usize> {
        match self {
            S(v) => Some(*v),
            _ => None,
        }
    }

    /// Whether the value is known statically.
    pub fn is_static(&self) -> bool {
        self.as_static().is_some()
    }

    /// The value, if it is a static integer of either sign: `Some(-3)` for
    /// the negative constant `-3`.
    pub fn as_static_signed(&self) -> Option<i64> {
        match self {
            S(v) => i64::try_from(*v).ok(),
            IndexExpr::Neg(e) => e.as_static().and_then(|v| i64::try_from(v).ok()).map(|v| -v),
            _ => None,
        }
    }

    /// The static integer `v`, of either sign.
    pub fn from_signed(v: i64) -> IndexExpr {
        if v >= 0 { S(v as usize) } else { IndexExpr::Neg(bx(S(v.unsigned_abs() as usize))) }
    }

    /// Whether every value of the expression is provably non-negative: it
    /// holds no negation or signed runtime value, or they cancel.
    pub fn is_non_negative(&self) -> bool {
        match self {
            S(_) | IndexExpr::Dynamic(_) => true,
            IndexExpr::Signed(_) | IndexExpr::Neg(_) => false,
            IndexExpr::Add(a, b) | IndexExpr::Mul(a, b) | IndexExpr::Min(a, b) => {
                a.is_non_negative() && b.is_non_negative()
            }
            // Formed only where the result is non-negative, or from
            // operands checked to be.
            IndexExpr::Sub(..)
            | IndexExpr::Div(..)
            | IndexExpr::Rem(..)
            | IndexExpr::And(..)
            | IndexExpr::Xor(..)
            | IndexExpr::Shr(..) => true,
        }
    }

    /// Panic, naming `op`, unless both operands are provably non-negative:
    /// for the operations whose rewrites need them to be.
    #[track_caller]
    fn require_non_negative(op: &str, a: &IndexExpr, b: &IndexExpr) {
        if let Some(e) = [a, b].into_iter().find(|e| !e.is_non_negative()) {
            panic!(
                "cute: index expression {a:?} {op} {b:?}: the operand {e:?} may be negative (it holds a negative or \
                 signed term), and index arithmetic only divides, takes remainders of, masks and shifts values that \
                 are provably non-negative; add a signed shift after the layout arithmetic (as domain_offset does) \
                 instead"
            );
        }
    }

    /// The expression with its runtime values taken to be of either sign:
    /// each [`IndexExpr::Dynamic`] reached through sums, differences,
    /// products and negations becomes [`IndexExpr::Signed`]. For a shift
    /// that may be negative even though it was converted from an `Int32`
    /// (the K residue `K - bK * k_tiles`, passed to
    /// [`domain_offset`](crate::domain_offset) as CuTe DSL passes it), so
    /// that comparisons of what it is added to stay signed and the folds
    /// that assume non-negative values do not apply. Quotients, remainders,
    /// masks, shifts and minima are kept: they are formed only from
    /// operands that are non-negative.
    pub(crate) fn with_signed_runtime_values(&self) -> IndexExpr {
        match self {
            IndexExpr::Dynamic(v) => IndexExpr::Signed(*v),
            IndexExpr::Neg(a) => Self::negation(a.with_signed_runtime_values()),
            IndexExpr::Add(a, b) => Self::sum(a.with_signed_runtime_values(), b.with_signed_runtime_values()),
            IndexExpr::Sub(a, b) => {
                Self::sum(a.with_signed_runtime_values(), Self::negation(b.with_signed_runtime_values()))
            }
            IndexExpr::Mul(a, b) => Self::product(a.with_signed_runtime_values(), b.with_signed_runtime_values()),
            e => e.clone(),
        }
    }

    /// `-a`, simplified (also available as the unary `-` operator).
    pub fn negation(a: IndexExpr) -> IndexExpr {
        match a {
            S(v) => Self::from_signed(-(v as i64)),
            IndexExpr::Neg(e) => *e,
            IndexExpr::Add(x, y) => Self::sum(Self::negation(*x), Self::negation(*y)),
            e => IndexExpr::Neg(bx(e)),
        }
    }

    /// `a + b`, simplified (also available as the `+` operator).
    pub fn sum(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        let constant = |e: &IndexExpr| e.as_static_signed().is_some();
        if let (Some(x), Some(y)) = (a.as_static_signed(), b.as_static_signed()) {
            return Self::from_signed(x + y);
        }
        match (a, b) {
            (S(0), e) | (e, S(0)) => e,
            (c, e) if constant(&c) => Self::sum(e, c),
            (IndexExpr::Add(e, c), y) if constant(&c) && constant(&y) => {
                Self::sum(*e, Self::from_signed(c.as_static_signed().unwrap() + y.as_static_signed().unwrap()))
            }
            (IndexExpr::Sub(e, c), S(y)) if c.is_static() => {
                let c = c.as_static().unwrap();
                if y >= c { Self::sum(*e, S(y - c)) } else { Self::difference(*e, S(c - y)) }
            }
            // (x - y) + y == x, in either order.
            (IndexExpr::Sub(x, y), b) | (b, IndexExpr::Sub(x, y)) if *y == b => *x,
            // x + -x == 0, in either order.
            (IndexExpr::Neg(x), b) | (b, IndexExpr::Neg(x)) if *x == b => S(0),
            (IndexExpr::Add(e1, c), e2) if constant(&c) => Self::sum(Self::sum(*e1, e2), *c),
            (e1, IndexExpr::Add(e2, c)) if constant(&c) => Self::sum(Self::sum(e1, *e2), *c),
            (a, b) => IndexExpr::Add(bx(a), bx(b)),
        }
    }

    /// `a - b`, simplified (also available as the `-` operator). For two
    /// non-negative operands the caller guarantees `a >= b`, and a static
    /// difference that would be negative panics; an operand that may be
    /// negative makes this `a + -b`.
    pub fn difference(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        if !(a.is_non_negative() && b.is_non_negative()) {
            return Self::sum(a, Self::negation(b));
        }
        match (a, b) {
            (S(x), S(y)) => {
                S(x.checked_sub(y).unwrap_or_else(|| panic!("index expression {x} - {y} would be negative")))
            }
            (e, S(0)) => e,
            (a, b) if a == b => S(0),
            (IndexExpr::Add(e, c), S(k)) if c.is_static() => {
                let c = c.as_static().unwrap();
                if c >= k { Self::sum(*e, S(c - k)) } else { Self::difference(*e, S(k - c)) }
            }
            (IndexExpr::Sub(e, c), S(k)) if c.is_static() => Self::difference(*e, S(c.as_static().unwrap() + k)),
            // (x + y) - y == x and (x + y) - x == y.
            (IndexExpr::Add(x, y), b) if *y == b => *x,
            (IndexExpr::Add(x, y), b) if *x == b => *y,
            (a, b) => IndexExpr::Sub(bx(a), bx(b)),
        }
    }

    /// `a * b`, simplified (also available as the `*` operator).
    pub fn product(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        match (a, b) {
            (S(x), S(y)) => S(x * y),
            (S(0), _) | (_, S(0)) => S(0),
            (S(1), e) | (e, S(1)) => e,
            (IndexExpr::Neg(a), b) | (b, IndexExpr::Neg(a)) => Self::negation(Self::product(*a, b)),
            (S(x), e) => Self::product(e, S(x)),
            (IndexExpr::Mul(e, c), S(y)) if c.is_static() => Self::product(*e, S(c.as_static().unwrap() * y)),
            (IndexExpr::Add(e1, e2), S(c)) => Self::sum(Self::product(*e1, S(c)), Self::product(*e2, S(c))),
            // (x - y) * e == x * e - y * e, which lets terms cancel in sums.
            (IndexExpr::Sub(x, y), e) | (e, IndexExpr::Sub(x, y)) => {
                Self::difference(Self::product(*x, e.clone()), Self::product(*y, e))
            }
            (IndexExpr::Mul(e1, c), e2) if c.is_static() => Self::product(Self::product(*e1, e2), *c),
            (e1, IndexExpr::Mul(e2, c)) if c.is_static() => Self::product(Self::product(e1, *e2), *c),
            (a, b) => IndexExpr::Mul(bx(a), bx(b)),
        }
    }

    /// `a / b` (floor division), simplified (also the `/` operator).
    #[track_caller]
    pub fn quotient(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative("/", &a, &b);
        let c = match b {
            S(c) => c,
            b => {
                // A runtime divisor, assumed non-zero.
                if a == S(0) {
                    return S(0);
                }
                return match a.exact_quotient(&b) {
                    Some(q) => q,
                    None => IndexExpr::Div(bx(a), bx(b)),
                };
            }
        };
        assert!(c != 0, "index expression divided by zero");
        if c == 1 {
            return a;
        }
        if let S(x) = a {
            return S(x / c);
        }
        if a.bound().is_some_and(|b| b <= c) {
            return S(0);
        }
        match a {
            IndexExpr::Mul(e, k) if k.is_static() => {
                let k = k.as_static().unwrap();
                if k % c == 0 {
                    Self::product(*e, S(k / c))
                } else if c % k == 0 {
                    Self::quotient(*e, S(c / k))
                } else {
                    IndexExpr::Div(bx(IndexExpr::Mul(e, bx(S(k)))), bx(S(c)))
                }
            }
            IndexExpr::Div(e, k) if k.is_static() => Self::quotient(*e, S(k.as_static().unwrap() * c)),
            a @ IndexExpr::Add(..) => {
                // (q*c + r) / c == q + r / c for non-negative terms.
                let (multiples, rest) = a.split_terms(c);
                if multiples.is_empty() {
                    return IndexExpr::Div(bx(a), bx(S(c)));
                }
                let q = multiples.into_iter().fold(S(0), |acc, t| Self::sum(acc, Self::quotient(t, S(c))));
                let r = rest.into_iter().fold(S(0), Self::sum);
                Self::sum(q, Self::quotient(r, S(c)))
            }
            a => IndexExpr::Div(bx(a), bx(S(c))),
        }
    }

    /// Flatten a sum into its terms.
    fn terms(&self) -> Vec<IndexExpr> {
        match self {
            IndexExpr::Add(a, b) => {
                let mut t = a.terms();
                t.extend(b.terms());
                t
            }
            e => vec![e.clone()],
        }
    }

    /// Split a sum into terms that are multiples of `c` and the rest
    /// (constants are split into their multiple-of-`c` part and remainder).
    fn split_terms(&self, c: usize) -> (Vec<IndexExpr>, Vec<IndexExpr>) {
        let mut multiples = Vec::new();
        let mut rest = Vec::new();
        for t in self.terms() {
            match t {
                S(k) => {
                    if k - k % c > 0 {
                        multiples.push(S(k - k % c));
                    }
                    if k % c > 0 {
                        rest.push(S(k % c));
                    }
                }
                t if t.is_multiple_of(c) => multiples.push(t),
                t => rest.push(t),
            }
        }
        (multiples, rest)
    }

    /// `a % b`, simplified (also the `%` operator).
    #[track_caller]
    pub fn remainder(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative("%", &a, &b);
        let c = match b {
            S(c) => c,
            b => {
                // A runtime divisor, assumed non-zero.
                if a == S(0) || a.exact_quotient(&b).is_some() {
                    return S(0);
                }
                return IndexExpr::Rem(bx(a), bx(b));
            }
        };
        assert!(c != 0, "index expression remainder by zero");
        if c == 1 {
            return S(0);
        }
        if let S(x) = a {
            return S(x % c);
        }
        if a.bound().is_some_and(|b| b <= c) {
            return a;
        }
        if a.is_multiple_of(c) {
            return S(0);
        }
        match a {
            IndexExpr::Rem(e, k) if k.as_static().is_some_and(|k| k % c == 0) => Self::remainder(*e, S(c)),
            a @ IndexExpr::Add(..) => {
                // (q*c + r) % c == r % c.
                let (multiples, rest) = a.split_terms(c);
                if multiples.is_empty() {
                    return IndexExpr::Rem(bx(a), bx(S(c)));
                }
                Self::remainder(rest.into_iter().fold(S(0), Self::sum), S(c))
            }
            IndexExpr::Mul(e, k) if k.as_static().is_some_and(|k| c % k == 0) => {
                let k = k.as_static().unwrap();
                Self::product(Self::remainder(*e, S(c / k)), S(k))
            }
            a => IndexExpr::Rem(bx(a), bx(S(c))),
        }
    }

    /// `a / b` rounded up, simplified. `b` must be non-zero (assumed so when
    /// it is a runtime value). An exact quotient folds to `a / b`; otherwise
    /// this is `(a + b - 1) / b`.
    #[track_caller]
    pub fn ceil_div(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative("ceil_div", &a, &b);
        match (&a, &b) {
            (S(x), S(y)) => {
                assert!(*y != 0, "index expression ceil_div by zero");
                S(x.div_ceil(*y))
            }
            (_, S(0)) => panic!("index expression ceil_div by zero"),
            (_, S(1)) => a,
            (_, S(c)) if a.is_multiple_of(*c) => Self::quotient(a, b),
            (_, S(c)) => {
                let c = *c;
                Self::quotient(Self::sum(a, S(c - 1)), b)
            }
            _ => match a.exact_quotient(&b) {
                Some(q) => q,
                None if a == S(0) => S(0),
                None => Self::quotient(Self::sum(a, Self::difference(b.clone(), S(1))), b),
            },
        }
    }

    /// `min(a, b)`, simplified. Composition takes the smaller of two extents
    /// this way when it cannot tell which one it is while the kernel is
    /// built, as C++ CuTe's `cute::min` does on dynamic integers.
    #[track_caller]
    pub fn min(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative("min", &a, &b);
        match (a, b) {
            (S(x), S(y)) => S(x.min(y)),
            (S(0), _) | (_, S(0)) => S(0),
            (a, b) if a == b => a,
            // A value bounded by the static one is the smaller.
            (e, S(c)) | (S(c), e) if e.bound().is_some_and(|b| b.saturating_sub(1) <= c) => e,
            (e, S(c)) | (S(c), e) => IndexExpr::Min(bx(e), bx(S(c))),
            (a, b) => IndexExpr::Min(bx(a), bx(b)),
        }
    }

    /// The largest static number known to divide every value of the
    /// expression: its value when it is static, otherwise what its static
    /// factors and the facts about its runtime values (a
    /// [`param_extent`](crate::param_extent)'s `multiple_of`, say) give, so
    /// 8 for `8 * m`.
    pub(crate) fn known_divisor(&self) -> usize {
        match self {
            S(v) => *v,
            e => e.divisor().unwrap_or(1),
        }
    }

    /// The static coefficient and the other factors of a product.
    fn factors(&self) -> (usize, Vec<&IndexExpr>) {
        match self {
            S(c) => (*c, Vec::new()),
            IndexExpr::Mul(a, b) => {
                let (ca, mut fa) = a.factors();
                let (cb, fb) = b.factors();
                fa.extend(fb);
                (ca * cb, fa)
            }
            e => (1, vec![e]),
        }
    }

    /// `self / d` when `d`'s factors provably divide `self`'s: `self` is a
    /// product that contains every factor of `d` (compared structurally) and
    /// a multiple of its coefficient, so `(m * n * 8) / (n * 4)` is `m * 2`.
    pub(crate) fn exact_quotient(&self, d: &IndexExpr) -> Option<IndexExpr> {
        if self == d {
            return Some(S(1));
        }
        let (cn, mut fn_) = self.factors();
        let (cd, fd) = d.factors();
        if cd == 0 || cn % cd != 0 {
            return None;
        }
        for f in fd {
            let at = fn_.iter().position(|g| *g == f)?;
            fn_.remove(at);
        }
        Some(fn_.into_iter().cloned().fold(S(cn / cd), Self::product))
    }

    /// Whether `self` and `other` provably have the same value: they are the
    /// same expression, or products of the same factors in another order
    /// (`m * n` and `n * m`). `false` means unknown, not different: `k * 2`
    /// and `k + k` are equal but not recognised as such.
    pub fn same_value(&self, other: &IndexExpr) -> bool {
        self == other || self.exact_quotient(other) == Some(S(1))
    }

    /// `a & b`, simplified (also the `&` operator).
    #[track_caller]
    pub fn and(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative("&", &a, &b);
        match (a, b) {
            (S(x), S(y)) => S(x & y),
            (S(0), _) | (_, S(0)) => S(0),
            (S(x), e) => Self::and(e, S(x)),
            (e, S(m)) if e.bound().is_some_and(|b| b.is_power_of_two() && (b - 1) & m == b - 1) => e,
            (a, b) => IndexExpr::And(bx(a), bx(b)),
        }
    }

    /// `a ^ b`, simplified (also the `^` operator).
    #[track_caller]
    pub fn xor(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative("^", &a, &b);
        match (a, b) {
            (S(x), S(y)) => S(x ^ y),
            (S(0), e) | (e, S(0)) => e,
            (a, b) => IndexExpr::Xor(bx(a), bx(b)),
        }
    }

    /// `a >> b`, simplified (also the `>>` operator).
    #[track_caller]
    pub fn shift_right(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        Self::require_non_negative(">>", &a, &b);
        match (a, b) {
            (S(x), S(s)) => S(x >> s),
            (e, S(0)) => e,
            (e, S(s)) if e.bound().is_some_and(|b| (b - 1) >> s == 0) => S(0),
            (a, b) => IndexExpr::Shr(bx(a), bx(b)),
        }
    }

    /// Largest known divisor of every value of the expression; `None` means
    /// the expression is identically zero.
    fn divisor(&self) -> Option<usize> {
        fn tz(d: Option<usize>) -> u32 {
            d.map_or(usize::BITS - 1, usize::trailing_zeros)
        }
        match self {
            S(0) => None,
            S(v) => Some(*v),
            IndexExpr::Dynamic(v) | IndexExpr::Signed(v) => crate::builder::value_divisor(*v),
            IndexExpr::Neg(a) => a.divisor(),
            IndexExpr::Add(a, b) | IndexExpr::Sub(a, b) => match (a.divisor(), b.divisor()) {
                (None, d) | (d, None) => d,
                (Some(x), Some(y)) => Some(gcd(x, y)),
            },
            // Not a saturating product: `usize::MAX` would claim odd
            // divisors the product does not have.
            IndexExpr::Mul(a, b) => match (a.divisor(), b.divisor()) {
                (None, _) | (_, None) => None,
                (Some(x), Some(y)) => Some(crate::builder::mul_divisors(x, y)),
            },
            IndexExpr::Rem(a, b) => match (a.divisor(), b.as_static()) {
                (Some(x), Some(c)) => Some(gcd(x, c)),
                (None, _) => None,
                _ => Some(1),
            },
            // The minimum is one of the two values (and 0 if either is).
            IndexExpr::Min(a, b) => match (a.divisor(), b.divisor()) {
                (None, _) | (_, None) => None,
                (Some(x), Some(y)) => Some(gcd(x, y)),
            },
            IndexExpr::And(a, b) => Some(1usize << tz(a.divisor()).max(tz(b.divisor())).min(usize::BITS - 2)),
            IndexExpr::Xor(a, b) => Some(1usize << tz(a.divisor()).min(tz(b.divisor())).min(usize::BITS - 2)),
            IndexExpr::Shr(a, b) => match b.as_static() {
                Some(s) => Some(1usize << tz(a.divisor()).saturating_sub(s as u32).min(usize::BITS - 2)),
                None => Some(1),
            },
            // A floor division by a constant keeps what is left of the
            // dividend's divisor when the constant divides it exactly.
            IndexExpr::Div(a, b) => match (a.divisor(), b.as_static()) {
                (None, _) => None,
                (Some(x), Some(c)) if c != 0 && x.is_multiple_of(c) => Some(x / c),
                _ => Some(1),
            },
        }
    }

    /// The width, 32 or 64 bits, the expression is computed in, with the
    /// types of its runtime values given by `ty` (`None` for a value that is
    /// not known, which counts as 32-bit): 64 when an `i64` value or a static
    /// value beyond `i32` appears in it.
    pub(crate) fn width_with(&self, ty: &dyn Fn(ValueId) -> Option<crate::ir::DType>) -> u32 {
        match self {
            S(v) => {
                if i32::try_from(*v).is_ok() {
                    32
                } else {
                    64
                }
            }
            IndexExpr::Neg(a) => a.width_with(ty),
            IndexExpr::Dynamic(v) | IndexExpr::Signed(v) => {
                if ty(*v) == Some(crate::ir::DType::I64) {
                    64
                } else {
                    32
                }
            }
            IndexExpr::Add(a, b)
            | IndexExpr::Sub(a, b)
            | IndexExpr::Mul(a, b)
            | IndexExpr::Div(a, b)
            | IndexExpr::Rem(a, b)
            | IndexExpr::Min(a, b)
            | IndexExpr::And(a, b)
            | IndexExpr::Xor(a, b)
            | IndexExpr::Shr(a, b) => a.width_with(ty).max(b.width_with(ty)),
        }
    }

    /// The width, 32 or 64 bits, the expression is computed in when it
    /// becomes a value in the kernel being built: 64 when one of its runtime
    /// values is an [`Int64`](crate::Int64) or one of its static values does
    /// not fit in `i32`, 32 otherwise.
    pub fn width(&self) -> u32 {
        self.width_with(&crate::builder::value_dtype)
    }

    /// Whether every value of the expression is a multiple of `n`.
    pub fn is_multiple_of(&self, n: usize) -> bool {
        self.divisor().is_none_or(|d| d % n == 0)
    }

    /// Exclusive upper bound on the expression's value, when known.
    pub fn bound(&self) -> Option<usize> {
        match self {
            S(v) => Some(v + 1),
            IndexExpr::Dynamic(v) => crate::builder::value_facts(*v).bound,
            // What is known about a value of either sign is not kept.
            IndexExpr::Signed(_) => None,
            // A non-positive value.
            IndexExpr::Neg(a) => a.is_non_negative().then_some(1),
            IndexExpr::Add(a, b) => match (a.as_static_signed(), b.as_static_signed()) {
                // Adding a negative constant lowers the bound, which stays at
                // least 1.
                (_, Some(c)) if c < 0 => Some(a.bound()?.saturating_sub(c.unsigned_abs() as usize).max(1)),
                (Some(c), _) if c < 0 => Some(b.bound()?.saturating_sub(c.unsigned_abs() as usize).max(1)),
                _ => Some(a.bound()? + b.bound()? - 1),
            },
            IndexExpr::Sub(a, _) => a.bound(),
            IndexExpr::Mul(a, b) if a.is_non_negative() && b.is_non_negative() => {
                Some((a.bound()? - 1).checked_mul(b.bound()? - 1)? + 1)
            }
            IndexExpr::Mul(..) => None,
            IndexExpr::Div(a, b) => match b.as_static() {
                Some(c) => Some((a.bound()? - 1) / c + 1),
                None => a.bound(),
            },
            IndexExpr::Rem(a, b) => match (a.bound(), b.as_static()) {
                (Some(x), Some(c)) => Some(x.min(c)),
                (None, Some(c)) => Some(c),
                (x, None) => x,
            },
            IndexExpr::Min(a, b) | IndexExpr::And(a, b) => match (a.bound(), b.bound()) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, y) => x.or(y),
            },
            IndexExpr::Xor(a, b) => Some(a.bound()?.max(b.bound()?).next_power_of_two()),
            IndexExpr::Shr(a, b) => Some(((a.bound()? - 1) >> b.as_static()?) + 1),
        }
    }

    /// Evaluate with runtime values supplied by `env`.
    pub fn eval(&self, env: &dyn Fn(ValueId) -> usize) -> usize {
        match self {
            S(v) => *v,
            IndexExpr::Dynamic(v) | IndexExpr::Signed(v) => env(*v),
            IndexExpr::Neg(a) => match a.eval(env) {
                0 => 0,
                x => panic!("index expression {self:?} is negative (-{x}); use eval_signed"),
            },
            IndexExpr::Add(a, b) => a.eval(env) + b.eval(env),
            IndexExpr::Sub(a, b) => {
                let (x, y) = (a.eval(env), b.eval(env));
                x.checked_sub(y).unwrap_or_else(|| panic!("index expression {self:?} is negative ({x} - {y})"))
            }
            IndexExpr::Mul(a, b) => a.eval(env) * b.eval(env),
            IndexExpr::Div(a, b) => a.eval(env) / b.eval(env),
            IndexExpr::Rem(a, b) => a.eval(env) % b.eval(env),
            IndexExpr::Min(a, b) => a.eval(env).min(b.eval(env)),
            IndexExpr::And(a, b) => a.eval(env) & b.eval(env),
            IndexExpr::Xor(a, b) => a.eval(env) ^ b.eval(env),
            IndexExpr::Shr(a, b) => a.eval(env) >> b.eval(env),
        }
    }

    /// Evaluate in `i64`, as the kernel computes it, with runtime values of
    /// either sign supplied by `env`: sums, differences and products wrap,
    /// so a signed expression gives its signed value.
    pub fn eval_signed(&self, env: &dyn Fn(ValueId) -> i64) -> i64 {
        match self {
            S(v) => *v as i64,
            IndexExpr::Dynamic(v) | IndexExpr::Signed(v) => env(*v),
            IndexExpr::Neg(a) => a.eval_signed(env).wrapping_neg(),
            IndexExpr::Add(a, b) => a.eval_signed(env).wrapping_add(b.eval_signed(env)),
            IndexExpr::Sub(a, b) => a.eval_signed(env).wrapping_sub(b.eval_signed(env)),
            IndexExpr::Mul(a, b) => a.eval_signed(env).wrapping_mul(b.eval_signed(env)),
            IndexExpr::Div(a, b) => a.eval_signed(env) / b.eval_signed(env),
            IndexExpr::Rem(a, b) => a.eval_signed(env) % b.eval_signed(env),
            IndexExpr::Min(a, b) => a.eval_signed(env).min(b.eval_signed(env)),
            IndexExpr::And(a, b) => a.eval_signed(env) & b.eval_signed(env),
            IndexExpr::Xor(a, b) => a.eval_signed(env) ^ b.eval_signed(env),
            IndexExpr::Shr(a, b) => a.eval_signed(env) >> b.eval_signed(env),
        }
    }

    /// Runtime values the expression depends on.
    pub fn dependencies(&self, out: &mut Vec<ValueId>) {
        match self {
            S(_) => {}
            IndexExpr::Dynamic(v) | IndexExpr::Signed(v) => out.push(*v),
            IndexExpr::Neg(a) => a.dependencies(out),
            IndexExpr::Add(a, b)
            | IndexExpr::Sub(a, b)
            | IndexExpr::Mul(a, b)
            | IndexExpr::Div(a, b)
            | IndexExpr::Rem(a, b)
            | IndexExpr::Min(a, b)
            | IndexExpr::And(a, b)
            | IndexExpr::Xor(a, b)
            | IndexExpr::Shr(a, b) => {
                a.dependencies(out);
                b.dependencies(out);
            }
        }
    }
}

/// A static value, or `?` for a value only known at run time (CuTe DSL's
/// notation for a dynamic integer), followed by what is known about it: its
/// largest known divisor when that is more than 1 and `i64` when it is
/// 64-bit, as in `?{div=8}`, `?{i64}` and `?{i64 div=4}`. `{:?}` prints the
/// expression instead.
impl fmt::Display for IndexExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(v) = self.as_static_signed() {
            return write!(f, "{v}");
        }
        let div = self.known_divisor();
        match (self.width() == 64, div) {
            (false, 1) => write!(f, "?"),
            (false, d) => write!(f, "?{{div={d}}}"),
            (true, 1) => write!(f, "?{{i64}}"),
            (true, d) => write!(f, "?{{i64 div={d}}}"),
        }
    }
}

/// The expression, with runtime values as `%n`: `((%3 * 4) + 1)`.
impl fmt::Debug for IndexExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bin = |f: &mut fmt::Formatter<'_>, a: &IndexExpr, op: &str, b: &IndexExpr| write!(f, "({a:?} {op} {b:?})");
        match self {
            S(v) => write!(f, "{v}"),
            IndexExpr::Dynamic(v) => write!(f, "{v}"),
            IndexExpr::Signed(v) => write!(f, "{v}s"),
            IndexExpr::Neg(a) => write!(f, "-{a:?}"),
            IndexExpr::Add(a, b) => bin(f, a, "+", b),
            IndexExpr::Sub(a, b) => bin(f, a, "-", b),
            IndexExpr::Mul(a, b) => bin(f, a, "*", b),
            IndexExpr::Div(a, b) => bin(f, a, "/", b),
            IndexExpr::Rem(a, b) => bin(f, a, "%", b),
            IndexExpr::Min(a, b) => write!(f, "min({a:?}, {b:?})"),
            IndexExpr::And(a, b) => bin(f, a, "&", b),
            IndexExpr::Xor(a, b) => bin(f, a, "^", b),
            IndexExpr::Shr(a, b) => bin(f, a, ">>", b),
        }
    }
}

macro_rules! index_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for IndexExpr {
            fn from(v: $t) -> Self {
                S(usize::try_from(v).unwrap_or_else(|_| panic!("index {v} does not fit in usize")))
            }
        }
    )*};
}
index_from_int!(usize, u8, u16, u32, u64);

/// A static integer of either sign: a negative one is a negative constant
/// ([`IndexExpr::Neg`]), as a signed shift takes it.
impl From<i32> for IndexExpr {
    fn from(v: i32) -> Self {
        IndexExpr::from_signed(v.into())
    }
}

/// A static integer of either sign, as for `i32`.
impl From<i64> for IndexExpr {
    fn from(v: i64) -> Self {
        IndexExpr::from_signed(v)
    }
}

impl std::ops::Neg for IndexExpr {
    type Output = IndexExpr;
    fn neg(self) -> IndexExpr {
        IndexExpr::negation(self)
    }
}

impl std::ops::Neg for &IndexExpr {
    type Output = IndexExpr;
    fn neg(self) -> IndexExpr {
        IndexExpr::negation(self.clone())
    }
}

impl From<&IndexExpr> for IndexExpr {
    fn from(e: &IndexExpr) -> Self {
        e.clone()
    }
}

impl super::tree::StaticLeaf for IndexExpr {
    fn static_value(&self) -> Option<usize> {
        self.as_static()
    }
}

macro_rules! index_binop {
    ($trait:ident, $method:ident, $ctor:ident) => {
        impl<R: Into<IndexExpr>> $trait<R> for IndexExpr {
            type Output = IndexExpr;
            fn $method(self, rhs: R) -> IndexExpr {
                IndexExpr::$ctor(self, rhs.into())
            }
        }
        impl<R: Into<IndexExpr>> $trait<R> for &IndexExpr {
            type Output = IndexExpr;
            fn $method(self, rhs: R) -> IndexExpr {
                IndexExpr::$ctor(self.clone(), rhs.into())
            }
        }
    };
}
index_binop!(Add, add, sum);
index_binop!(Sub, sub, difference);
index_binop!(Mul, mul, product);
index_binop!(Div, div, quotient);
index_binop!(Rem, rem, remainder);
index_binop!(BitAnd, bitand, and);
index_binop!(BitXor, bitxor, xor);
index_binop!(Shr, shr, shift_right);

#[cfg(test)]
mod tests {
    use super::*;

    fn d(i: u32) -> IndexExpr {
        IndexExpr::Dynamic(ValueId(i))
    }

    #[test]
    fn folding() {
        assert_eq!(S(3) + 4usize, S(7));
        assert_eq!((d(0) * 8usize) / 16usize, d(0) / 2usize);
        assert_eq!((d(0) * 16usize) % 16usize, S(0));
        assert_eq!((d(0) * 16usize + d(1) * 4usize) / 4usize, d(0) * 4usize + d(1));
        assert_eq!(((d(0) % 8usize) % 16usize), d(0) % 8usize);
        assert_eq!((d(0) % 8usize) / 8usize, S(0));
        assert_eq!((d(0) * 2usize) % 8usize, (d(0) % 4usize) * 2usize);
        assert!((d(0) * 8usize + 16usize).is_multiple_of(8));
        assert!(!(d(0) * 8usize + 4usize).is_multiple_of(8));
        // min folds static operands, equal operands and a bounded operand.
        assert_eq!(IndexExpr::min(S(3), S(5)), S(3));
        assert_eq!(IndexExpr::min(d(0), d(0)), d(0));
        assert_eq!(IndexExpr::min(S(0), d(0)), S(0));
        assert_eq!(IndexExpr::min(d(0) % 8usize, S(8)), d(0) % 8usize);
        assert_eq!(IndexExpr::min(S(4), d(0)), IndexExpr::Min(bx(d(0)), bx(S(4))));
        assert_eq!(format!("{:?}", IndexExpr::min(d(0), d(1))), "min(%0, %1)");
        assert_eq!(IndexExpr::min(d(0) * 8usize, d(1) * 12usize).known_divisor(), 4);
        assert_eq!((d(0) * 8usize).known_divisor(), 8);
        assert_eq!(S(6).known_divisor(), 6);
    }

    #[test]
    fn rewrites_preserve_values() {
        // The simplified forms must agree with the arithmetic they encode.
        for a in 0..64usize {
            for b in 0..64usize {
                let env = |v: ValueId| if v.0 == 0 { a } else { b };
                assert_eq!((((d(0) * 3usize + 5usize) * 4usize) / 6usize).eval(&env), ((a * 3 + 5) * 4) / 6);
                assert_eq!((((d(0) * 12usize + d(1)) % 8usize) + d(0) / 3usize).eval(&env), ((a * 12 + b) % 8) + a / 3);
                assert_eq!(
                    (((d(0) % 16usize) * 8usize + (d(1) / 4usize) * 128usize) % 64usize).eval(&env),
                    ((a % 16) * 8 + (b / 4) * 128) % 64
                );
                assert_eq!(((d(0) * 16usize + d(1) % 16usize) / 16usize).eval(&env), (a * 16 + b % 16) / 16);
                assert_eq!(((d(0) * 2usize) % 8usize).eval(&env), (a * 2) % 8);
                let m = IndexExpr::min(d(0) * 3usize, d(1) + 2usize);
                assert_eq!(m.eval(&env), (a * 3).min(b + 2));
                assert_eq!(IndexExpr::min(d(0), S(17)).eval(&env), a.min(17));
                assert!(m.bound().is_none_or(|bound| (a * 3).min(b + 2) < bound));
            }
        }
    }
}

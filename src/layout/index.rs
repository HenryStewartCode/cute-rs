//! Symbolic index expressions.
//!
//! Layouts are static, but the coordinates fed into them often are not
//! (thread indices, block indices, loop counters). [`IndexExpr`] lets layout
//! arithmetic run at kernel-build time on a mix of static integers and
//! runtime values, folding and simplifying as it goes. Only what cannot be
//! resolved statically becomes IR, when the expression is materialised.
//!
//! Every value an `IndexExpr` describes is assumed to be non-negative, which
//! is what makes rewrites such as `(x * 16 + y) / 16 == x + y / 16` valid.

use std::fmt;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Shr};

use crate::ir::ValueId;

/// A non-negative integer expression over static values and runtime `i32`s.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum IndexExpr {
    Static(usize),
    /// A runtime `i32` value, assumed non-negative.
    Dynamic(ValueId),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    Div(Box<IndexExpr>, Box<IndexExpr>),
    Rem(Box<IndexExpr>, Box<IndexExpr>),
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

    /// `a + b`, simplified (also available as the `+` operator).
    pub fn sum(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        match (a, b) {
            (S(x), S(y)) => S(x + y),
            (S(0), e) | (e, S(0)) => e,
            (S(x), e) => Self::sum(e, S(x)),
            (IndexExpr::Add(e, c), S(y)) if c.is_static() => Self::sum(*e, S(c.as_static().unwrap() + y)),
            (IndexExpr::Add(e1, c), e2) if c.is_static() => Self::sum(Self::sum(*e1, e2), *c),
            (e1, IndexExpr::Add(e2, c)) if c.is_static() => Self::sum(Self::sum(e1, *e2), *c),
            (a, b) => IndexExpr::Add(bx(a), bx(b)),
        }
    }

    /// `a * b`, simplified (also available as the `*` operator).
    pub fn product(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        match (a, b) {
            (S(x), S(y)) => S(x * y),
            (S(0), _) | (_, S(0)) => S(0),
            (S(1), e) | (e, S(1)) => e,
            (S(x), e) => Self::product(e, S(x)),
            (IndexExpr::Mul(e, c), S(y)) if c.is_static() => Self::product(*e, S(c.as_static().unwrap() * y)),
            (IndexExpr::Add(e1, e2), S(c)) => Self::sum(Self::product(*e1, S(c)), Self::product(*e2, S(c))),
            (IndexExpr::Mul(e1, c), e2) if c.is_static() => Self::product(Self::product(*e1, e2), *c),
            (e1, IndexExpr::Mul(e2, c)) if c.is_static() => Self::product(Self::product(e1, *e2), *c),
            (a, b) => IndexExpr::Mul(bx(a), bx(b)),
        }
    }

    /// `a / b` (floor division), simplified (also the `/` operator).
    pub fn quotient(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        let c = match b {
            S(c) => c,
            b => return IndexExpr::Div(bx(a), bx(b)),
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
    pub fn remainder(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        let c = match b {
            S(c) => c,
            b => return IndexExpr::Rem(bx(a), bx(b)),
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

    /// `a & b`, simplified (also the `&` operator).
    pub fn and(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        match (a, b) {
            (S(x), S(y)) => S(x & y),
            (S(0), _) | (_, S(0)) => S(0),
            (S(x), e) => Self::and(e, S(x)),
            (e, S(m)) if e.bound().is_some_and(|b| b.is_power_of_two() && (b - 1) & m == b - 1) => e,
            (a, b) => IndexExpr::And(bx(a), bx(b)),
        }
    }

    /// `a ^ b`, simplified (also the `^` operator).
    pub fn xor(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        match (a, b) {
            (S(x), S(y)) => S(x ^ y),
            (S(0), e) | (e, S(0)) => e,
            (a, b) => IndexExpr::Xor(bx(a), bx(b)),
        }
    }

    /// `a >> b`, simplified (also the `>>` operator).
    pub fn shift_right(a: IndexExpr, b: IndexExpr) -> IndexExpr {
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
            IndexExpr::Dynamic(v) => Some(crate::builder::value_facts(*v).divisor),
            IndexExpr::Add(a, b) => match (a.divisor(), b.divisor()) {
                (None, d) | (d, None) => d,
                (Some(x), Some(y)) => Some(gcd(x, y)),
            },
            IndexExpr::Mul(a, b) => match (a.divisor(), b.divisor()) {
                (None, _) | (_, None) => None,
                (Some(x), Some(y)) => Some(x.saturating_mul(y)),
            },
            IndexExpr::Rem(a, b) => match (a.divisor(), b.as_static()) {
                (Some(x), Some(c)) => Some(gcd(x, c)),
                (None, _) => None,
                _ => Some(1),
            },
            IndexExpr::And(a, b) => Some(1usize << tz(a.divisor()).max(tz(b.divisor())).min(usize::BITS - 2)),
            IndexExpr::Xor(a, b) => Some(1usize << tz(a.divisor()).min(tz(b.divisor())).min(usize::BITS - 2)),
            IndexExpr::Shr(a, b) => match b.as_static() {
                Some(s) => Some(1usize << tz(a.divisor()).saturating_sub(s as u32).min(usize::BITS - 2)),
                None => Some(1),
            },
            IndexExpr::Div(..) => Some(1),
        }
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
            IndexExpr::Add(a, b) => Some(a.bound()? + b.bound()? - 1),
            IndexExpr::Mul(a, b) => Some((a.bound()? - 1).checked_mul(b.bound()? - 1)? + 1),
            IndexExpr::Div(a, b) => match b.as_static() {
                Some(c) => Some((a.bound()? - 1) / c + 1),
                None => a.bound(),
            },
            IndexExpr::Rem(a, b) => match (a.bound(), b.as_static()) {
                (Some(x), Some(c)) => Some(x.min(c)),
                (None, Some(c)) => Some(c),
                (x, None) => x,
            },
            IndexExpr::And(a, b) => match (a.bound(), b.bound()) {
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
            IndexExpr::Dynamic(v) => env(*v),
            IndexExpr::Add(a, b) => a.eval(env) + b.eval(env),
            IndexExpr::Mul(a, b) => a.eval(env) * b.eval(env),
            IndexExpr::Div(a, b) => a.eval(env) / b.eval(env),
            IndexExpr::Rem(a, b) => a.eval(env) % b.eval(env),
            IndexExpr::And(a, b) => a.eval(env) & b.eval(env),
            IndexExpr::Xor(a, b) => a.eval(env) ^ b.eval(env),
            IndexExpr::Shr(a, b) => a.eval(env) >> b.eval(env),
        }
    }

    /// Runtime values the expression depends on.
    pub fn dependencies(&self, out: &mut Vec<ValueId>) {
        match self {
            S(_) => {}
            IndexExpr::Dynamic(v) => out.push(*v),
            IndexExpr::Add(a, b)
            | IndexExpr::Mul(a, b)
            | IndexExpr::Div(a, b)
            | IndexExpr::Rem(a, b)
            | IndexExpr::And(a, b)
            | IndexExpr::Xor(a, b)
            | IndexExpr::Shr(a, b) => {
                a.dependencies(out);
                b.dependencies(out);
            }
        }
    }
}

impl fmt::Display for IndexExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bin = |f: &mut fmt::Formatter<'_>, a: &IndexExpr, op: &str, b: &IndexExpr| write!(f, "({a} {op} {b})");
        match self {
            S(v) => write!(f, "{v}"),
            IndexExpr::Dynamic(v) => write!(f, "{v}"),
            IndexExpr::Add(a, b) => bin(f, a, "+", b),
            IndexExpr::Mul(a, b) => bin(f, a, "*", b),
            IndexExpr::Div(a, b) => bin(f, a, "/", b),
            IndexExpr::Rem(a, b) => bin(f, a, "%", b),
            IndexExpr::And(a, b) => bin(f, a, "&", b),
            IndexExpr::Xor(a, b) => bin(f, a, "^", b),
            IndexExpr::Shr(a, b) => bin(f, a, ">>", b),
        }
    }
}

impl fmt::Debug for IndexExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

macro_rules! index_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for IndexExpr {
            fn from(v: $t) -> Self {
                S(usize::try_from(v).unwrap_or_else(|_| panic!("negative index {v}")))
            }
        }
    )*};
}
index_from_int!(usize, u32, u64, i32, i64);

impl From<&IndexExpr> for IndexExpr {
    fn from(e: &IndexExpr) -> Self {
        e.clone()
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
            }
        }
    }
}

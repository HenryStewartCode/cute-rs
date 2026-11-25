//! Parsing CuTe's printed form of tuples and layouts.
//!
//! `"(2,(3,4))".parse::<IntTuple>()` and
//! `"(2,(3,4)):(1,(2,6))".parse::<Layout>()` read what `Display` writes, so
//! layouts printed by CuTe (or its documentation) can be used directly.
//! Whitespace is ignored, `(3)` and `(3,)` are 1-tuples, `()` is the empty
//! tuple, and an integer may carry C++ CuTe's `_` prefix for static values
//! (`_8`).
//!
//! In a [`Layout`] or a [`Shape`] (`"(?,8)".parse::<Shape>()`), `?` is a
//! dynamic leaf, as CuTe DSL prints one: each `?` is a fresh runtime value,
//! different from every other, and prints back as `?`. It is a placeholder
//! for exercising the algebra on dynamic layouts, such as
//! `"(?,?):(?,1)"`; it stands for no value of a kernel, so an offset that
//! depends on it cannot be computed in one. An [`IntTuple`] is static and
//! does not take `?`.
//!
//! A stride leaf may be a scaled basis element, as CuTe prints one: a scale
//! (an integer, `?`, or a ratio `n/d`), then `@` and the mode path,
//! innermost first: `"(2,3):(1@0,1@1)"`, `"4:2@1@0"` (mode 1 of mode 0),
//! `"8:1/4@0"`. [`Stride`] parses on its own too.
//!
//! Swizzles parse as CuTe prints them, `"S<3,4,3>".parse::<Swizzle>()`,
//! and composed layouts as `inner o offset o outer`: `"S<3,5,4> o 2 o
//! (2,3):(1,2)".parse::<ComposedLayout>()`, with a layout inner too.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};

use super::basis::{BasisScale, Ratio, ScaledBasis, StrideLeaf};
use super::index::IndexExpr;
use super::layout::{Layout, Shape, Stride};
use super::swizzle::{ComposedLayout, Inner, Swizzle};
use super::tree::{IntTuple, Tree};
use crate::ir::ValueId;

/// Why a tuple or layout string did not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseLayoutError {
    input: String,
    position: usize,
    message: String,
}

impl ParseLayoutError {
    /// The byte offset in the input where parsing failed.
    pub fn position(&self) -> usize {
        self.position
    }
}

impl fmt::Display for ParseLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot parse {:?} at byte {}: {}", self.input, self.position, self.message)
    }
}

impl std::error::Error for ParseLayoutError {}

/// A fresh runtime value for a `?`. Placeholders are numbered down from the
/// top of the value range, so they never name a value of a kernel.
fn placeholder() -> IndexExpr {
    static NEXT: AtomicU32 = AtomicU32::new(u32::MAX);
    IndexExpr::Dynamic(ValueId(NEXT.fetch_sub(1, Ordering::Relaxed)))
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input, pos: 0 }
    }

    fn error(&self, message: impl Into<String>) -> ParseLayoutError {
        ParseLayoutError { input: self.input.to_string(), position: self.pos, message: message.into() }
    }

    /// The next byte that is not whitespace, without consuming it.
    fn peek(&mut self) -> Option<u8> {
        let bytes = self.input.as_bytes();
        while bytes.get(self.pos).is_some_and(|b| b.is_ascii_whitespace()) {
            self.pos += 1;
        }
        bytes.get(self.pos).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), ParseLayoutError> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected `{}`", c as char)))
        }
    }

    fn finish(&mut self) -> Result<(), ParseLayoutError> {
        match self.peek() {
            None => Ok(()),
            Some(_) => Err(self.error("unexpected text after the end")),
        }
    }

    /// A tuple whose leaves `leaf` reads; `dynamic` says whether `?` is one.
    fn tree<T>(
        &mut self,
        dynamic: bool,
        leaf: &mut impl FnMut(&mut Self) -> Result<T, ParseLayoutError>,
    ) -> Result<Tree<T>, ParseLayoutError> {
        let expected = if dynamic { "a non-negative integer, `?` or `(`" } else { "a non-negative integer or `(`" };
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let mut children = Vec::new();
                if self.peek() == Some(b')') {
                    self.pos += 1;
                    return Ok(Tree::Branch(children));
                }
                loop {
                    children.push(self.tree(dynamic, leaf)?);
                    match self.peek() {
                        Some(b',') => {
                            self.pos += 1;
                            // A trailing comma, as Python writes 1-tuples.
                            if self.peek() == Some(b')') {
                                self.pos += 1;
                                break;
                            }
                        }
                        Some(b')') => {
                            self.pos += 1;
                            break;
                        }
                        _ => return Err(self.error("expected `,` or `)`")),
                    }
                }
                Ok(Tree::Branch(children))
            }
            Some(b'_' | b'0'..=b'9' | b'?') => leaf(self).map(Tree::Leaf),
            Some(_) => Err(self.error(format!("expected {expected}"))),
            None => Err(self.error(format!("unexpected end; expected {expected}"))),
        }
    }

    fn int_tuple(&mut self) -> Result<IntTuple, ParseLayoutError> {
        self.tree(false, &mut |p| {
            if p.peek() == Some(b'?') {
                return Err(p.error(
                    "`?` (a runtime value) cannot be a leaf of an IntTuple, which is static; parse a Shape or a Layout",
                ));
            }
            p.integer()
        })
    }

    fn shape(&mut self) -> Result<Shape, ParseLayoutError> {
        self.tree(true, &mut |p| {
            if p.peek() == Some(b'?') {
                p.pos += 1;
                return Ok(placeholder());
            }
            p.integer().map(IndexExpr::Static)
        })
    }

    /// A stride: integers, `?`, and scaled basis elements `scale@m@...`.
    fn stride(&mut self) -> Result<Stride, ParseLayoutError> {
        self.tree(true, &mut |p| {
            let scale = if p.peek() == Some(b'?') {
                p.pos += 1;
                BasisScale::Int(placeholder())
            } else {
                let n = p.integer()?;
                if p.peek() == Some(b'/') {
                    p.pos += 1;
                    if !p.peek().is_some_and(|b| b.is_ascii_digit()) {
                        return Err(p.error("expected the denominator of a ratio"));
                    }
                    let d = p.integer()?;
                    let (Ok(n), Ok(d)) = (i64::try_from(n), i64::try_from(d)) else {
                        return Err(p.error("ratio too large"));
                    };
                    if d == 0 {
                        return Err(p.error("the denominator of a ratio must not be 0"));
                    }
                    if p.peek() != Some(b'@') {
                        return Err(p.error("a ratio is only a stride as the scale of a basis element: expected `@`"));
                    }
                    BasisScale::Ratio(Ratio::new(n, d))
                } else {
                    BasisScale::Int(IndexExpr::Static(n))
                }
            };
            if p.peek() != Some(b'@') {
                let BasisScale::Int(e) = scale else { unreachable!("a ratio is followed by `@`") };
                return Ok(StrideLeaf::Int(e));
            }
            let mut mode = Vec::new();
            while p.peek() == Some(b'@') {
                p.pos += 1;
                if !p.peek().is_some_and(|b| b.is_ascii_digit()) {
                    return Err(p.error("expected a mode index after `@`"));
                }
                mode.push(p.integer()?);
            }
            mode.reverse();
            Ok(StrideLeaf::Basis(ScaledBasis::new(scale, mode)))
        })
    }

    fn integer(&mut self) -> Result<usize, ParseLayoutError> {
        let bytes = self.input.as_bytes();
        if bytes[self.pos] == b'_' {
            self.pos += 1;
        }
        let start = self.pos;
        let mut value: usize = 0;
        while let Some(&b) = bytes.get(self.pos).filter(|b| b.is_ascii_digit()) {
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add(usize::from(b - b'0')))
                .ok_or_else(|| self.error("integer too large"))?;
            self.pos += 1;
        }
        if self.pos == start {
            return Err(self.error("expected digits"));
        }
        Ok(value)
    }
}

impl FromStr for IntTuple {
    type Err = ParseLayoutError;

    /// Parse a static tuple as CuTe prints it, such as `(2,(3,4))` or `8`.
    fn from_str(s: &str) -> Result<IntTuple, ParseLayoutError> {
        let mut p = Parser::new(s);
        let t = p.int_tuple()?;
        p.finish()?;
        Ok(t)
    }
}

impl FromStr for Shape {
    type Err = ParseLayoutError;

    /// Parse a shape (or a stride, or a coordinate) as CuTe prints it, such
    /// as `(?,(3,4))`: each `?` is a fresh runtime value.
    fn from_str(s: &str) -> Result<Shape, ParseLayoutError> {
        let mut p = Parser::new(s);
        let t = p.shape()?;
        p.finish()?;
        Ok(t)
    }
}

impl FromStr for Stride {
    type Err = ParseLayoutError;

    /// Parse a stride as CuTe prints it, such as `(1@0,(1@0@1,1@1@1))` or
    /// `(?,1)`: each `?` is a fresh runtime value.
    fn from_str(s: &str) -> Result<Stride, ParseLayoutError> {
        let mut p = Parser::new(s);
        let t = p.stride()?;
        p.finish()?;
        Ok(t)
    }
}

impl<'a> Parser<'a> {
    /// `shape:stride`, the two congruent.
    fn layout(&mut self) -> Result<Layout, ParseLayoutError> {
        let shape = self.shape()?;
        self.expect(b':')?;
        let stride_at = self.pos;
        let stride = self.stride()?;
        if !shape.congruent(&stride) {
            self.pos = stride_at;
            return Err(self.error(format!("shape {shape} and stride {stride} are not congruent")));
        }
        Ok(Layout::new(shape, stride))
    }

    /// `S<b,m,s>`, the shift of either sign.
    fn swizzle(&mut self) -> Result<Swizzle, ParseLayoutError> {
        self.expect(b'S')?;
        self.expect(b'<')?;
        let bits = self.integer_after_space()?;
        self.expect(b',')?;
        let base = self.integer_after_space()?;
        self.expect(b',')?;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.pos += 1;
        }
        let shift = self.integer_after_space()?;
        self.expect(b'>')?;
        let to = |v: usize| u32::try_from(v).ok().filter(|&v| v < 64);
        let (Some(bits), Some(base), Some(shift)) = (to(bits), to(base), to(shift)) else {
            return Err(self.error("swizzle parameters out of range"));
        };
        let shift = if negative { -(shift as i32) } else { shift as i32 };
        if shift.unsigned_abs() < bits || bits + base + shift.unsigned_abs() >= usize::BITS {
            return Err(self.error(format!("S<{bits},{base},{shift}> is not a swizzle: |shift| must be at least bits")));
        }
        Ok(Swizzle::new(bits, base, shift))
    }

    fn integer_after_space(&mut self) -> Result<usize, ParseLayoutError> {
        if self.peek().is_none() {
            return Err(self.error("expected digits"));
        }
        self.integer()
    }
}

impl FromStr for Layout {
    type Err = ParseLayoutError;

    /// Parse a layout as CuTe prints it, `shape:stride`, such as
    /// `(2,(3,4)):(1,(2,6))`, `(?,?):(?,1)` or `(4,8):(1@0,1@1)`. The two
    /// halves need the same nesting; each `?` is a fresh runtime value.
    fn from_str(s: &str) -> Result<Layout, ParseLayoutError> {
        let mut p = Parser::new(s);
        let layout = p.layout()?;
        p.finish()?;
        Ok(layout)
    }
}

impl FromStr for Swizzle {
    type Err = ParseLayoutError;

    /// Parse a swizzle as CuTe prints it, `S<3,4,3>` (or `S<2,4,-3>`).
    fn from_str(s: &str) -> Result<Swizzle, ParseLayoutError> {
        let mut p = Parser::new(s);
        let sw = p.swizzle()?;
        p.finish()?;
        Ok(sw)
    }
}

impl FromStr for ComposedLayout {
    type Err = ParseLayoutError;

    /// Parse a composed layout as CuTe prints it, `inner o offset o outer`:
    /// `S<3,5,4> o 2 o (2,3):(1,2)`, `(4,5):(1,4) o (0,0) o
    /// (2,3):(1@0,1@1)`. The offset is a tuple that may hold `?`.
    fn from_str(s: &str) -> Result<ComposedLayout, ParseLayoutError> {
        let mut p = Parser::new(s);
        let inner = if p.peek() == Some(b'S') { Inner::Swizzle(p.swizzle()?) } else { Inner::Layout(p.layout()?) };
        p.expect(b'o')?;
        let offset = p.shape()?;
        p.expect(b'o')?;
        let outer_at = p.pos;
        let outer = p.layout()?;
        p.finish()?;
        if let Inner::Swizzle(sw) = &inner
            && (outer.has_scaled_basis() || offset.as_leaf().is_none())
        {
            p.pos = outer_at;
            return Err(p.error(format!("invalid composition {sw} o {offset} o {outer}")));
        }
        Ok(ComposedLayout { inner, offset, outer })
    }
}

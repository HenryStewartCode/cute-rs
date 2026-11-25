//! Device printing: [`printf`], [`printf_end`] and [`printf_values`]
//! (CuTe DSL's `cute.printf`), with `{}` placeholders for numbers,
//! pointers, layouts, tuples and tensors next to C's `%` conversions.
//!
//! Everything is resolved when the kernel is built. A `{}` becomes the C
//! conversion of its operand's type (`%d`, `%u`, `%lld`, `%llu`, `%f`, `%p`),
//! and a layout, tuple or tensor becomes text: its static parts as they
//! print on the host, each runtime leaf a `%d` (`%lld` when it is 64 bits wide)
//! whose value joins the operands. The result is one `vprintf` call (more
//! when it has over [`MAX_PRINTF_ARGS`] operands, or for a tensor's data).
//!
//! `vprintf` reads its operands as C variadic arguments, so they are widened
//! as C promotes them: Boolean and the 8- and 16-bit integers to 32 bits
//! (sign- or zero-extended, by the type's signedness) and the narrow floats
//! (Float16, BFloat16, the 8-, 6- and 4-bit formats, TFloat32) to Float64, in
//! the IR, so the simulator sees the same values. Float32 operands are left
//! to `gpu.printf`'s lowering, which extends them to `double` itself.

use std::panic::Location;

use crate::builder;
use crate::ir::{DType, Loc, Op, ValueId};
use crate::layout::{
    BasisScale, ComposedLayout, FnComposedLayout, IndexExpr, Inner, IntTuple, Layout, ScaledBasis, StrideLeaf, Tree,
};
use crate::tensor::{Ptr, Tensor};
use crate::tensor_ssa::{PrintTensor, TensorSsa};
use crate::value::{AnyValue, Numeric, Value};

/// The most operands one `vprintf` call takes (CUDA's limit).
pub const MAX_PRINTF_ARGS: usize = 32;

/// An operand of [`printf`]: a number, a pointer, or a structure printed as
/// text (a layout, a composed layout, an integer tuple, a coordinate with
/// `_`, a stride with scaled-basis leaves, a tile, a tensor). Convert with
/// `.into()`: `printf("{} at {}\n", &[x.into(), (&layout).into()])`.
///
/// Rust literals convert as their types do: `bool` to Boolean, `i32` (an
/// unsuffixed integer) to Int32, `f32` to Float32 and `f64` (an unsuffixed
/// float) to Float64; `usize` to Int32, as elsewhere in cute-rs.
#[derive(Clone)]
pub struct PrintArg(Repr);

#[derive(Clone)]
enum Repr {
    /// A number.
    Value(ValueId),
    /// A pointer: its address, `%p` through a `{}`.
    Pointer { address: ValueId },
    /// Text: C format pieces (any `%` of static text doubled) and their
    /// operands, with what the operand is, for messages.
    Text { what: &'static str, pieces: Vec<(String, Vec<AnyValue>)> },
}

impl<T: Numeric> From<Value<T>> for PrintArg {
    #[track_caller]
    fn from(v: Value<T>) -> PrintArg {
        PrintArg(Repr::Value(v.checked()))
    }
}

impl From<AnyValue> for PrintArg {
    fn from(v: AnyValue) -> PrintArg {
        PrintArg(Repr::Value(v.0))
    }
}

macro_rules! from_literal {
    ($($t:ty => $v:ty),* $(,)?) => {$(
        impl From<$t> for PrintArg {
            #[track_caller]
            fn from(x: $t) -> PrintArg {
                PrintArg::from(<$v>::from(x))
            }
        }
    )*};
}
from_literal!(
    bool => crate::Boolean, i32 => crate::Int32, i64 => crate::Int64, u32 => crate::Uint32, u64 => crate::Uint64,
    f32 => crate::Float32, f64 => crate::Float64, usize => crate::Int32,
);

impl<T: Numeric> From<&Ptr<T>> for PrintArg {
    #[track_caller]
    fn from(p: &Ptr<T>) -> PrintArg {
        PrintArg(Repr::Pointer { address: p.toint().0 })
    }
}

impl<T: Numeric> From<Ptr<T>> for PrintArg {
    #[track_caller]
    fn from(p: Ptr<T>) -> PrintArg {
        PrintArg::from(&p)
    }
}

/// Text built from static pieces and runtime leaves.
#[derive(Default)]
struct Render {
    pieces: Vec<(String, Vec<AnyValue>)>,
}

impl Render {
    /// Static text (a `%` in it is doubled).
    fn text(&mut self, s: &str) {
        let s = s.replace('%', "%%");
        match self.pieces.last_mut() {
            Some(last) => last.0.push_str(&s),
            None => self.pieces.push((s, vec![])),
        }
    }

    /// A conversion and its operand.
    fn conv(&mut self, spec: &str, v: ValueId) {
        match self.pieces.last_mut() {
            Some(last) => {
                last.0.push_str(spec);
                last.1.push(AnyValue(v));
            }
            None => self.pieces.push((spec.to_string(), vec![AnyValue(v)])),
        }
    }

    /// An integer: static as its digits, runtime as `%d` or `%lld`.
    #[track_caller]
    fn int(&mut self, e: &IndexExpr) {
        if let Some(v) = e.as_static_signed() {
            return self.text(&v.to_string());
        }
        let id = crate::tensor::index_operand(e.clone());
        let spec = if builder::with(|b| b.dtype(id)) == DType::I64 { "%lld" } else { "%d" };
        self.conv(spec, id);
    }

    #[track_caller]
    fn tree<T>(&mut self, t: &Tree<T>, leaf: &mut impl FnMut(&mut Render, &T)) {
        match t {
            Tree::Leaf(v) => leaf(self, v),
            Tree::Branch(cs) => {
                self.text("(");
                for (i, c) in cs.iter().enumerate() {
                    if i > 0 {
                        self.text(",");
                    }
                    self.tree(c, leaf);
                }
                self.text(")");
            }
        }
    }

    #[track_caller]
    fn basis(&mut self, b: &ScaledBasis) {
        match b.value() {
            BasisScale::Int(e) => self.int(e),
            BasisScale::Ratio(r) => self.text(&r.to_string()),
        }
        for m in b.mode().iter().rev() {
            self.text(&format!("@{m}"));
        }
    }

    #[track_caller]
    fn stride_leaf(&mut self, s: &StrideLeaf) {
        match s {
            StrideLeaf::Int(e) => self.int(e),
            StrideLeaf::Basis(b) => self.basis(b),
        }
    }

    #[track_caller]
    fn layout(&mut self, l: &Layout) {
        self.tree(l.shape(), &mut |r, e| r.int(e));
        self.text(":");
        self.tree(l.stride(), &mut |r, s| r.stride_leaf(s));
    }

    #[track_caller]
    fn composed(&mut self, c: &ComposedLayout) {
        match &c.inner {
            Inner::Swizzle(sw) => self.text(&sw.to_string()),
            Inner::Layout(l) => self.layout(l),
        }
        self.text(" o ");
        self.tree(&c.offset, &mut |r, e| r.int(e));
        self.text(" o ");
        self.layout(&c.outer);
    }

    fn arg(self, what: &'static str) -> PrintArg {
        PrintArg(Repr::Text { what, pieces: self.pieces })
    }
}

macro_rules! from_text {
    ($t:ty, $what:literal, |$r:ident, $x:ident| $body:expr) => {
        impl From<&$t> for PrintArg {
            #[track_caller]
            fn from($x: &$t) -> PrintArg {
                let mut $r = Render::default();
                $body;
                $r.arg($what)
            }
        }
        impl From<$t> for PrintArg {
            #[track_caller]
            fn from(x: $t) -> PrintArg {
                PrintArg::from(&x)
            }
        }
    };
}

from_text!(Layout, "layout", |r, l| r.layout(l));
from_text!(ComposedLayout, "composed layout", |r, c| r.composed(c));
from_text!(IntTuple, "integer tuple", |r, t| r.text(&t.to_string()));
// A shape or a coordinate (the same type).
from_text!(Tree<IndexExpr>, "integer tuple", |r, t| r.tree(t, &mut |r, e| r.int(e)));
from_text!(IndexExpr, "integer", |r, e| r.int(e));
from_text!(Tree<StrideLeaf>, "stride", |r, t| r.tree(t, &mut |r, s| r.stride_leaf(s)));
from_text!(ScaledBasis, "scaled basis", |r, b| r.basis(b));
// A coordinate with `_` (CuTe's `None`) for the modes it leaves free.
from_text!(Tree<Option<IndexExpr>>, "coordinate", |r, t| r.tree(t, &mut |r, e| match e {
    Some(e) => r.int(e),
    None => r.text("_"),
}));
// A tile: a layout per mode, `_` for a mode it leaves alone.
from_text!(Tree<Option<Layout>>, "tile", |r, t| r.tree(t, &mut |r, l| match l {
    Some(l) => r.layout(l),
    None => r.text("_"),
}));

/// `(name o offset o outer)`, as the host prints it, with the offset's and
/// the outer layout's runtime leaves.
impl<F> From<&FnComposedLayout<F>> for PrintArg {
    #[track_caller]
    fn from(c: &FnComposedLayout<F>) -> PrintArg {
        let mut r = Render::default();
        r.text(&format!("({} o ", c.name()));
        r.tree(c.offset(), &mut |r, e| r.int(e));
        r.text(" o ");
        r.layout(c.outer());
        r.text(")");
        r.arg("composed layout")
    }
}

/// A tensor prints as [`print_tensor`](crate::print_tensor) prints it: the
/// header (element type, where it lives, layout) and the data. CuTe DSL
/// prints a tensor operand with its data too. A tensor whose layout carries
/// a swizzle (made with a swizzled composed layout) is refused, as in CuTe
/// DSL; one made from a swizzled pointer prints.
#[track_caller]
fn tensor_arg(t: &impl PrintTensor, swizzled_layout: bool) -> PrintArg {
    if swizzled_layout {
        builder::user_panic(
            Location::caller(),
            "printf: a tensor whose layout is swizzled (a swizzled composed layout, PISL) cannot be printed; build \
             it from a swizzled pointer (PDSL, recast_ptr) and print that tensor",
        );
    }
    let _at = builder::LocationScope::new();
    let mut pieces = crate::tensor_ssa::print_pieces(t, false);
    // The text ends with a newline, which the format's own text replaces.
    if let Some(last) = pieces.last_mut()
        && last.0.ends_with('\n')
    {
        last.0.pop();
    }
    PrintArg(Repr::Text { what: "tensor", pieces })
}

impl<T: Numeric> From<&Tensor<T>> for PrintArg {
    #[track_caller]
    fn from(t: &Tensor<T>) -> PrintArg {
        tensor_arg(t, t.swizzle_in_layout())
    }
}

impl From<&Tensor<crate::tensor::Ident>> for PrintArg {
    #[track_caller]
    fn from(t: &Tensor<crate::tensor::Ident>) -> PrintArg {
        tensor_arg(t, false)
    }
}

impl<T: Numeric> From<&TensorSsa<T>> for PrintArg {
    #[track_caller]
    fn from(t: &TensorSsa<T>) -> PrintArg {
        tensor_arg(t, false)
    }
}

/// One piece of a format string.
#[derive(Debug, PartialEq)]
pub(crate) enum FormatPiece<'a> {
    Text(&'a str),
    /// `{}`.
    Placeholder,
    /// A `%` conversion (its whole text).
    Conversion(Conversion<'a>),
}

/// A parsed `%` conversion.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct Conversion<'a> {
    pub text: &'a str,
    /// The conversion letter (`%` for `%%`).
    pub conv: char,
    /// `l`, `ll`, `j`, `z`, `t` or `L`: a 64-bit operand.
    pub long: bool,
}

impl Conversion<'_> {
    /// The operand types the conversion reads, for messages.
    fn wants(&self) -> &'static str {
        match self.conv {
            'd' | 'i' | 'c' if self.long => "a signed 64-bit integer (Int64)",
            'd' | 'i' | 'c' => "a 32-bit integer or narrower (Int32, Int16, Int8, Boolean)",
            'u' | 'x' | 'X' | 'o' if self.long => "a 64-bit integer (Uint64, Int64)",
            'u' | 'x' | 'X' | 'o' => "a 32-bit integer or narrower (Uint32, Int32, ...)",
            'p' => "a pointer or a 64-bit address",
            _ => "a float",
        }
    }

    /// Whether an operand of type `t` (after widening) fits.
    pub(crate) fn accepts(&self, t: DType) -> bool {
        match self.conv {
            'd' | 'i' | 'u' | 'x' | 'X' | 'o' | 'c' => {
                (t.is_int() || t == DType::Bool) && if self.long { t.bits() == 64 } else { t.bits() <= 32 }
            }
            'p' => matches!(t, DType::I64 | DType::U64),
            _ => t.is_float(),
        }
    }

    /// Why an operand of type `t` does not fit.
    pub(crate) fn mismatch(&self, t: DType) -> String {
        format!(
            "the conversion `{}` reads {}, but its operand is {}{}",
            self.text,
            self.wants(),
            t.cute_name(),
            match (self.conv, t.bits()) {
                ('d' | 'i' | 'u' | 'x' | 'X' | 'o', 64) => " (use %lld or %llu)",
                ('d' | 'i' | 'u' | 'x' | 'X' | 'o', _) if t.is_int() || t == DType::Bool => " (use %d or %u)",
                _ if t.is_float() => " (use %f, %e or %g)",
                _ => "",
            }
        )
    }
}

/// Split a format into text, `{}` placeholders (when `braces`) and `%`
/// conversions; an unsupported conversion is an error.
pub(crate) fn parse_format(format: &str, braces: bool) -> Result<Vec<FormatPiece<'_>>, String> {
    let bytes = format.as_bytes();
    let mut out = Vec::new();
    let (mut i, mut start) = (0, 0);
    while i < bytes.len() {
        if braces && bytes[i] == b'{' && bytes.get(i + 1) == Some(&b'}') {
            if start < i {
                out.push(FormatPiece::Text(&format[start..i]));
            }
            out.push(FormatPiece::Placeholder);
            i += 2;
            start = i;
            continue;
        }
        if bytes[i] != b'%' {
            i += 1;
            continue;
        }
        if start < i {
            out.push(FormatPiece::Text(&format[start..i]));
        }
        let begin = i;
        i += 1;
        while i < bytes.len() && b"-+ 0#".contains(&bytes[i]) {
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        let mut long = false;
        while i < bytes.len() && b"hlLjzt".contains(&bytes[i]) {
            long |= bytes[i] != b'h';
            i += 1;
        }
        let Some(&c) = bytes.get(i) else {
            return Err(format!(
                "the format ends inside the conversion `{}` (write %% for a literal %)",
                &format[begin..]
            ));
        };
        if !b"diuoxXceEfFgGp%".contains(&c) {
            let next = format[i..].chars().next().map_or(0, char::len_utf8);
            return Err(format!(
                "`{}` is not a conversion printf takes (it takes d i u o x X c e E f F g G p and %%; write %% for a \
                 literal %)",
                &format[begin..i + next]
            ));
        }
        i += 1;
        let text = &format[begin..i];
        if c == b'%' && text != "%%" {
            return Err(format!("`{text}`: a literal % is written %%"));
        }
        out.push(FormatPiece::Conversion(Conversion { text, conv: c as char, long }));
        start = i;
    }
    if start < bytes.len() {
        out.push(FormatPiece::Text(&format[start..]));
    }
    Ok(out)
}

/// The operands a format's conversions read (`%%` reads none).
pub(crate) fn operand_conversions<'a>(pieces: &[FormatPiece<'a>]) -> Vec<Conversion<'a>> {
    pieces
        .iter()
        .filter_map(|p| match p {
            FormatPiece::Conversion(c) if c.conv != '%' => Some(*c),
            _ => None,
        })
        .collect()
}

/// `x` widened as C's variadic promotion widens it: Boolean and integers
/// narrower than 32 bits to Int32 or Uint32, floats other than Float32
/// and Float64 to Float64.
#[track_caller]
fn promote(x: ValueId, loc: Loc) -> ValueId {
    let t = builder::with(|b| b.dtype(x));
    let to = match t {
        // Refused by the validator, naming the line.
        t if t.bits() == 128 || t.is_word_packed() => return x,
        DType::Bool | DType::I4 | DType::I8 | DType::I16 => DType::I32,
        DType::U8 | DType::U16 => DType::U32,
        DType::F32 | DType::F64 => return x,
        t if t.is_float() => DType::F64,
        _ => return x,
    };
    builder::with(|b| crate::convert::convert(b, x, to, loc))
}

/// The conversion a `{}` takes for an operand of type `t` (after
/// promotion).
fn placeholder_spec(t: DType) -> &'static str {
    match t {
        DType::I64 => "%lld",
        DType::U64 => "%llu",
        t if t.is_float() => "%f",
        t if t.is_signed() => "%d",
        _ => "%u",
    }
}

/// A 64-bit address for `%p`.
#[track_caller]
fn address64(a: ValueId, loc: Loc) -> ValueId {
    builder::with(|b| match b.dtype(a) {
        DType::I64 => a,
        _ => {
            let u = b.bitcast(a, DType::U32, loc);
            b.cast(u, DType::I64, loc)
        }
    })
}

/// Push one `Op::Printf` (the format already C text), its operands
/// promoted.
pub(crate) fn push_printf(format: &str, args: &[AnyValue], loc: Loc) {
    let args = args.iter().map(|a| promote(a.0, loc)).collect();
    builder::with(|b| b.push(Op::Printf { format: format.to_string(), args }, loc));
}

/// A piece with more than `per_call` operands (a structure with that many
/// runtime leaves) cut after each of its conversions, so that it can be
/// spread over several calls; other pieces are kept whole.
fn split_oversized(pieces: Vec<(String, Vec<AnyValue>)>, per_call: usize) -> Vec<(String, Vec<AnyValue>)> {
    let mut out = Vec::with_capacity(pieces.len());
    for (text, args) in pieces {
        let parsed = match parse_format(&text, false) {
            Ok(p) if args.len() > per_call && operand_conversions(&p).len() == args.len() => p,
            _ => {
                out.push((text, args));
                continue;
            }
        };
        let mut args = args.into_iter();
        let mut cur = String::new();
        for p in parsed {
            match p {
                FormatPiece::Text(t) => cur.push_str(t),
                FormatPiece::Conversion(c) if c.conv == '%' => cur.push_str(c.text),
                FormatPiece::Conversion(c) => {
                    cur.push_str(c.text);
                    out.push((std::mem::take(&mut cur), vec![args.next().expect("counted")]));
                }
                FormatPiece::Placeholder => unreachable!("no placeholders without braces"),
            }
        }
        if !cur.is_empty() {
            out.push((cur, vec![]));
        }
    }
    out
}

/// Emit `pieces` as `vprintf` calls of at most `per_call` operands, cut
/// between pieces (and inside a piece that alone has more than `per_call`).
pub(crate) fn emit_pieces(pieces: Vec<(String, Vec<AnyValue>)>, per_call: usize, loc: Loc) {
    let mut format = String::new();
    let mut args: Vec<AnyValue> = Vec::new();
    for (text, a) in split_oversized(pieces, per_call) {
        if !args.is_empty() && args.len() + a.len() > per_call {
            push_printf(&format, &args, loc);
            format.clear();
            args.clear();
        }
        format.push_str(&text);
        args.extend(a);
    }
    if !format.is_empty() {
        push_printf(&format, &args, loc);
    }
}

/// Resolve `format` with `args` into C format pieces.
#[track_caller]
fn resolve(format: &str, args: &[PrintArg], loc: Loc) -> Vec<(String, Vec<AnyValue>)> {
    let parsed = parse_format(format, true)
        .unwrap_or_else(|e| builder::user_panic(loc, format!("printf: {e}, in the format {format:?}")));
    let holes = parsed
        .iter()
        .filter(|p| !matches!(p, FormatPiece::Text(_) | FormatPiece::Conversion(Conversion { conv: '%', .. })))
        .count();
    if holes != args.len() {
        builder::user_panic(
            loc,
            format!(
                "printf: the format {format:?} has {holes} placeholder{} and conversion{}, but {} operand{} {} given",
                if holes == 1 { "" } else { "s" },
                if holes == 1 { "" } else { "s" },
                args.len(),
                if args.len() == 1 { "" } else { "s" },
                if args.len() == 1 { "is" } else { "are" },
            ),
        );
    }
    let mut out: Vec<(String, Vec<AnyValue>)> = vec![(String::new(), vec![])];
    let mut args = args.iter();
    for piece in parsed {
        let cur = out.last_mut().expect("a piece");
        match piece {
            FormatPiece::Text(t) => cur.0.push_str(t),
            FormatPiece::Conversion(c) if c.conv == '%' => cur.0.push_str("%%"),
            FormatPiece::Placeholder => match &args.next().expect("counted").0 {
                Repr::Value(v) => {
                    let v = promote(*v, loc);
                    cur.0.push_str(placeholder_spec(builder::with(|b| b.dtype(v))));
                    cur.1.push(AnyValue(v));
                    out.push((String::new(), vec![]));
                }
                Repr::Pointer { address } => {
                    cur.0.push_str("%p");
                    cur.1.push(AnyValue(address64(*address, loc)));
                    out.push((String::new(), vec![]));
                }
                Repr::Text { pieces, .. } => {
                    let mut pieces = pieces.iter();
                    if let Some((t, a)) = pieces.next() {
                        cur.0.push_str(t);
                        cur.1.extend(a.iter().copied());
                    }
                    out.extend(pieces.cloned());
                    out.push((String::new(), vec![]));
                }
            },
            FormatPiece::Conversion(c) => {
                let v = match &args.next().expect("counted").0 {
                    Repr::Value(v) => promote(*v, loc),
                    Repr::Pointer { address } if c.conv == 'p' => address64(*address, loc),
                    Repr::Pointer { address } => *address,
                    Repr::Text { what, .. } => builder::user_panic(
                        loc,
                        format!("printf: a {what} prints through a {{}} placeholder, not the conversion `{}`", c.text),
                    ),
                };
                let t = builder::with(|b| b.dtype(v));
                // 128-bit and packed operands are refused by the validator.
                let refused_later = t.bits() == 128 || t.is_word_packed();
                if !refused_later && !c.accepts(t) {
                    builder::user_panic(loc, format!("printf: {}", c.mismatch(t)));
                }
                cur.0.push_str(c.text);
                cur.1.push(AnyValue(v));
                out.push((String::new(), vec![]));
            }
        }
    }
    out
}

/// Print from the GPU (CuTe DSL's `cute.printf(format, *args, end="")`):
/// `format` is C's `printf` format, in which `{}` also stands for the next
/// operand, printed by its type: a number as `%d`, `%u`, `%lld`, `%llu` or
/// `%f`, a pointer as `%p`, and a layout, composed layout, tuple,
/// coordinate (`_` for a free mode), stride, tile or tensor as text with
/// its runtime leaves filled in. `{}` and `%` conversions mix: `printf("a={},
/// b=%.2f\n", &[layout.into(), x.into()])`.
///
/// Nothing is appended: this is one `vprintf` call, so write the newline in
/// the format. CuTe DSL's `cute.printf` appends `end="\n"` by default;
/// [`printf_end`] takes the `end`.
///
/// A `%` conversion must fit its operand (`%d` a 32-bit or narrower
/// integer, `%lld` an Int64 or Uint64, `%f` a float, `%p` a pointer); CuTe DSL
/// leaves a mismatch undefined, cute-rs refuses it, as it refuses a count
/// of conversions that differs from the operands. The simulator collects
/// the lines in [`SimReport::printed`](crate::SimReport), formatted as
/// `vprintf` formats them.
#[track_caller]
pub fn printf(format: &str, args: &[PrintArg]) {
    printf_end(format, args, "");
}

/// [`printf`] followed by `end` (literal text): CuTe DSL's
/// `cute.printf(format, *args, end=end)`, whose default `end="\n"` is
/// `printf_end(format, args, "\n")`.
#[track_caller]
pub fn printf_end(format: &str, args: &[PrintArg], end: &str) {
    let loc = Location::caller();
    let mut pieces = resolve(format, args, loc);
    pieces.last_mut().expect("a piece").0.push_str(&end.replace('%', "%%"));
    emit_pieces(pieces, MAX_PRINTF_ARGS, loc);
}

/// Print values without a format (CuTe DSL's `cute.printf(a, b, ...)`):
/// `{}, {}, ...` followed by a newline.
#[track_caller]
pub fn printf_values(args: &[PrintArg]) {
    assert!(!args.is_empty(), "printf_values: give at least one value to print");
    let format = vec!["{}"; args.len()].join(", ");
    printf_end(&format, args, "\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_parse() {
        let p = parse_format("a={}, b=%.2f %% %lld{x}", true).unwrap();
        assert_eq!(p.len(), 9, "{p:?}");
        assert_eq!(p[1], FormatPiece::Placeholder);
        assert_eq!(operand_conversions(&p).iter().map(|c| c.text).collect::<Vec<_>>(), ["%.2f", "%lld"]);
        assert!(parse_format("%s", true).unwrap_err().contains("`%s` is not a conversion"));
        assert!(parse_format("50%", true).unwrap_err().contains("ends inside"));
        assert!(parse_format("%5%", true).unwrap_err().contains("literal %"));
        assert_eq!(parse_format("{}", false).unwrap(), vec![FormatPiece::Text("{}")]);
    }
}

//! [`print_tensor`]: CuTe DSL's `cute.print_tensor`, as `printf` calls
//! emitted when the kernel is built.

use super::TensorSsa;
use crate::builder::LocationScope;
use crate::ir::DType;
use crate::layout::{IntTuple, Layout, Tree};
use crate::tensor::{Engine, Ident, Tensor};
use crate::value::{AnyValue, IntoValue, Numeric, Value};

/// Arguments per `printf` call: CUDA's `vprintf` takes at most 32.
const ARGS_PER_CALL: usize = 16;

/// What [`print_tensor`] prints: a [`Tensor`] in memory or registers, a
/// coordinate tensor, or a [`TensorSsa`].
pub trait PrintTensor {
    #[doc(hidden)]
    fn print_parts(&self) -> PrintParts;
}

/// The header, the shape, and every element as a `printf` piece and its
/// arguments, in colexicographic order.
#[doc(hidden)]
pub struct PrintParts {
    header: String,
    shape: IntTuple,
    elements: Vec<(String, Vec<AnyValue>)>,
}

/// One element's conversion and arguments: floats as doubles (`%7.4f`;
/// `vprintf` has no half type), signed integers as `%4d` (`%4lld` for
/// 64-bit), unsigned as `%4u` (`%4llu`), Booleans as `%d`, 128-bit integers
/// in hexadecimal.
#[track_caller]
fn element<T: Numeric>(v: Value<T>) -> (String, Vec<AnyValue>) {
    let t = T::DTYPE;
    if t.is_float() {
        return ("%7.4f".into(), vec![v.to::<f64>().into()]);
    }
    match t {
        DType::Bool => ("%d".into(), vec![v.to::<i32>().into()]),
        DType::I64 => ("%4lld".into(), vec![v.into()]),
        DType::U64 => ("%4llu".into(), vec![v.into()]),
        DType::I128 | DType::U128 => {
            let x = v.bitcast::<u128>();
            ("0x%016llx%016llx".into(), vec![(x >> 64u128).to::<u64>().into(), x.to::<u64>().into()])
        }
        t if t.is_signed() => ("%4d".into(), vec![v.to::<i32>().into()]),
        _ => ("%4u".into(), vec![v.to::<u32>().into()]),
    }
}

fn type_name(t: DType) -> &'static str {
    t.cute_name()
}

impl<T: Numeric> PrintTensor for TensorSsa<T> {
    #[track_caller]
    fn print_parts(&self) -> PrintParts {
        let layout = Layout::col_major(self.shape.clone());
        PrintParts {
            header: format!("tensor(rmem<{}> o {layout}, data=", type_name(T::DTYPE)),
            shape: self.shape.clone(),
            elements: self.values.iter().map(|&v| element(v)).collect(),
        }
    }
}

impl<T: Numeric> PrintTensor for Tensor<T> {
    #[track_caller]
    fn print_parts(&self) -> PrintParts {
        let values = self.load_ssa();
        let swizzle = self.swizzle().map(|s| format!("{s} o ")).unwrap_or_default();
        let header = match &self.engine {
            Engine::Register(_) => format!("tensor(rmem<{}> o {swizzle}{}, data=", type_name(T::DTYPE), self.layout()),
            _ => {
                let align = self.iterator_alignment(T::BYTES).unwrap_or(T::BYTES);
                format!(
                    "tensor(raw_ptr<{}, {}, align({align})> o {swizzle}{}, data=",
                    type_name(T::DTYPE),
                    self.memspace(),
                    self.layout()
                )
            }
        };
        PrintParts {
            header,
            shape: values.shape.clone(),
            elements: values.values.iter().map(|&v| element(v)).collect(),
        }
    }
}

impl PrintTensor for Tensor<Ident> {
    #[track_caller]
    fn print_parts(&self) -> PrintParts {
        let shape = super::SsaShape::ssa_shape(self);
        let elements = (0..shape.size())
            .map(|i| {
                let c = self.coord(i);
                let leaves = c.leaves();
                let text = match &c {
                    Tree::Leaf(_) => "%d".to_string(),
                    _ => fmt_coord_pattern(&c.map(&mut |_| "%d".to_string())),
                };
                let args =
                    leaves.into_iter().map(|e| AnyValue::from(IntoValue::<i32>::into_value(e.clone()))).collect();
                (text, args)
            })
            .collect();
        PrintParts { header: format!("tensor(coord<{:?}> o {}, data=", self.origin(), self.layout()), shape, elements }
    }
}

fn fmt_coord_pattern(t: &Tree<String>) -> String {
    match t {
        Tree::Leaf(s) => s.clone(),
        Tree::Branch(cs) => format!("({})", cs.iter().map(fmt_coord_pattern).collect::<Vec<_>>().join(",")),
    }
}

/// Emit `printf` calls for `text` pieces and their arguments, at most
/// [`ARGS_PER_CALL`] arguments a call.
#[track_caller]
fn emit(pieces: Vec<(String, Vec<AnyValue>)>) {
    crate::print::emit_pieces(pieces, ARGS_PER_CALL, std::panic::Location::caller());
}

/// `%` doubled, for static text inside a format.
fn literal(s: &str) -> String {
    s.replace('%', "%%")
}

/// Print a tensor from every thread that runs it (CuTe DSL's
/// `cute.print_tensor(t, verbose=)`), as `printf` calls built here: a
/// header naming the element type, where the tensor lives and its layout,
/// then the data, which is loaded for a tensor in memory (whose shape must
/// be static), read for a fragment or a [`TensorSsa`], and the coordinates
/// for a coordinate tensor.
///
/// ```text
/// tensor(raw_ptr<Float32, gmem, align(16)> o (2,3):(3,1), data=
///        [[ 0.5000, -1.2500,  2.0000],
///         [ 3.0000,  4.0000, -0.0000]])
/// ```
///
/// The data is in torch's bracket style: a rank-1 tensor is one row; a
/// rank-2 tensor prints mode 0 down and mode 1 across; above rank 2 the
/// modes from 2 on number rank-2 blocks (as C++ CuTe's `print_tensor`
/// iterates them), printed one after another in an outer bracket. A nested
/// mode counts as one mode, indexed in colexicographic order. Floats print
/// as `%7.4f` (Float16, BFloat16 and the narrow formats converted to double,
/// since `vprintf` has no half type), signed integers as `%4d` (`%4lld`
/// for 64 bits), unsigned ones as `%4u` (`%4llu`), Booleans as 0 and 1, and
/// 128-bit integers in hexadecimal. With `verbose`, every element is on its
/// own line after its coordinate: `  (1,2): -1.2500`.
///
/// Every thread prints, so wrap the call in `if_(thread_idx().x.eq(0), ...)`
/// for one copy. The simulator collects the text in
/// [`SimReport::printed`](crate::SimReport), formatted as `vprintf` formats
/// it.
#[track_caller]
pub fn print_tensor(tensor: &impl PrintTensor, verbose: bool) {
    let _at = LocationScope::new();
    emit(print_pieces(tensor, verbose));
}

/// [`print_tensor`]'s text, as `printf` pieces and their arguments.
#[track_caller]
pub(crate) fn print_pieces(tensor: &impl PrintTensor, verbose: bool) -> Vec<(String, Vec<AnyValue>)> {
    let parts = tensor.print_parts();
    let mut pieces: Vec<(String, Vec<AnyValue>)> = vec![(literal(&parts.header) + "\n", vec![])];
    let shape = &parts.shape;
    let n = parts.elements.len();
    if verbose {
        let layout = Layout::col_major(shape.clone());
        for (i, (text, args)) in parts.elements.into_iter().enumerate() {
            let crd = layout.idx2crd(i);
            pieces.push((format!("  {crd}: {text}\n"), args));
        }
        pieces.push((")\n".into(), vec![]));
        return pieces;
    }
    // Extents: rows (mode 0), columns (mode 1), blocks (the rest).
    let modes: Vec<usize> = match shape {
        Tree::Leaf(e) => vec![*e],
        Tree::Branch(ms) => ms.iter().map(|m| m.size()).collect(),
    };
    let rank = modes.len();
    let elems = parts.elements;
    let indent = "       ";
    if rank <= 1 {
        let row: Vec<_> = (0..n).collect();
        pieces.push((format!("{indent}["), vec![]));
        push_row(&mut pieces, &elems, &row);
        pieces.push(("])\n".into(), vec![]));
        return pieces;
    }
    let (rows, cols) = (modes[0], modes[1]);
    let blocks: usize = modes[2..].iter().product();
    let nested = rank > 2;
    pieces.push((format!("{indent}{}", if nested { "[" } else { "" }), vec![]));
    for b in 0..blocks {
        if b > 0 {
            pieces.push((format!(",\n\n{indent} "), vec![]));
        }
        for r in 0..rows {
            let lead = if r == 0 { "[[".to_string() } else { format!("{indent} {}[", if nested { " " } else { "" }) };
            pieces.push((lead, vec![]));
            let row: Vec<usize> = (0..cols).map(|c| r + rows * (c + cols * b)).collect();
            push_row(&mut pieces, &elems, &row);
            pieces.push((if r + 1 < rows { "],\n".into() } else { "]]".into() }, vec![]));
        }
    }
    pieces.push((format!("{})\n", if nested { "]" } else { "" }), vec![]));
    pieces
}

fn push_row(pieces: &mut Vec<(String, Vec<AnyValue>)>, elems: &[(String, Vec<AnyValue>)], row: &[usize]) {
    for (k, &i) in row.iter().enumerate() {
        let (text, args) = &elems[i];
        let sep = if k > 0 { ", " } else { "" };
        pieces.push((format!("{sep}{text}"), args.clone()));
    }
}

/// Print `values`, row-major over `shape`, in nested brackets after
/// `header` (`cutlass.print_nd_vector`).
#[track_caller]
pub(crate) fn print_nd<T: Numeric>(header: &str, shape: &[usize], values: &[Value<T>]) {
    fn nest(pieces: &mut Vec<(String, Vec<AnyValue>)>, elems: &[(String, Vec<AnyValue>)], shape: &[usize], at: usize) {
        pieces.push(("[".into(), vec![]));
        let inner: usize = shape[1..].iter().product();
        for i in 0..shape[0] {
            if i > 0 {
                pieces.push((", ".into(), vec![]));
            }
            if shape.len() == 1 {
                let (text, args) = &elems[at + i];
                pieces.push((text.clone(), args.clone()));
            } else {
                nest(pieces, elems, &shape[1..], at + i * inner);
            }
        }
        pieces.push(("]".into(), vec![]));
    }
    let elems: Vec<_> = values.iter().map(|&v| element(v)).collect();
    let mut pieces = vec![(literal(header), vec![])];
    nest(&mut pieces, &elems, shape, 0);
    pieces.push(("\n".into(), vec![]));
    emit(pieces);
}

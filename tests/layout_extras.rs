//! The layout extras: composed layouts with a closure inner,
//! `size_in_bytes`, `max_alignment` with divisor facts, LaTeX printing and
//! the general `make_tensor`.

use cute::prelude::*;
use cute::viz::{
    LatexOptions, LatexTvOptions, Palette, print_latex, print_latex_body, print_latex_tv, print_latex_tv_with,
    print_latex_with, tikz_color_rainbow, tikz_escape,
};

/// Run `f` while a kernel is built (what needs runtime values).
fn in_kernel(f: impl FnOnce()) {
    let mut f = Some(f);
    let _ = Kernel::builder("scratch").block_dim([32, 1, 1]).build(|| (f.take().unwrap())());
}

fn panic_text(f: impl FnOnce()) -> String {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

fn leaf(c: &Coord) -> usize {
    c.as_leaf().and_then(IndexExpr::as_static).expect("a static integer")
}

/// `composed_layout.ipynb`'s custom transformation: `(x, y) -> (x, y + 1)`
/// over `make_identity_layout((8, 4))` offset by `(1, 0)`, printed as CuTe
/// DSL prints it; slices and the algebra act on the outer layout.
#[test]
fn fn_composed_layout_notebook_transformation() {
    let inner = |c: Coord| (leaf(c.mode(0)), leaf(c.mode(1)) + 1);
    let layout = make_composed_layout_fn(inner, (1, 0), make_identity_layout((8, 4)));
    assert_eq!(layout.to_string(), "(fn o (1,0) o (8,4):(1@0,1@1))");
    assert_eq!(layout.named("shift").to_string(), "(shift o (1,0) o (8,4):(1@0,1@1))");
    assert_eq!(layout.call(0), (1, 1));
    assert_eq!(layout.call(9), (2, 2));
    assert_eq!(layout.call((7, 3)), (8, 4));
    // A slice keeps the inner and moves the fixed modes into the offset.
    let column = layout.slice((.., 2));
    assert_eq!(column.to_string(), "(fn o (1,2) o (8):(1@0))");
    assert_eq!(column.call(5), layout.call((5, 2)));
    // The algebra tiles the outer layout.
    let tiled = zipped_divide(&layout, (2, 2));
    assert_eq!(size(&tiled, &[]).as_static(), Some(32));
    assert_eq!(tiled.call(((1, 1), (2, 0))), layout.call((5, 1)));
    assert_eq!(coalesce(&layout).outer(), &coalesce(layout.outer()));
}

/// `size_in_bytes` of static, runtime, swizzled and layout-inner composed
/// layouts, sub-byte types, `None`, and the Hopper GEMM's slice of a staged
/// shared-memory layout (`size_in_bytes(a_dtype, slice_(staged, (None, None, 0)))`).
#[test]
fn size_in_bytes_of_layouts() {
    let l = Layout::col_major((16, 8));
    assert_eq!(size_in_bytes::<f32>(&l).as_static(), Some(512));
    assert_eq!(size_in_bytes::<F16>(make_layout((4, 4), (1, 8))).as_static(), Some(2 * 28));
    assert_eq!(size_in_bytes_dtype(DType::F64, &l).as_static(), Some(1024));
    // Sub-byte widths round up to a whole byte.
    assert_eq!(size_in_bytes::<I4>(Layout::col_major(3)).as_static(), Some(2));
    assert_eq!(size_in_bytes::<F4E2M1FN>(Layout::col_major(4)).as_static(), Some(2));
    assert_eq!(size_in_bytes::<f32>(None::<Layout>).as_static(), Some(0));
    // Swizzled: the outer layout's cosize; layout inner: the inner's.
    let atom = make_composed_layout(make_swizzle(3, 4, 3), 0, Layout::row_major((8, 64)));
    assert_eq!(size_in_bytes::<F16>(&atom).as_static(), Some(1024));
    let transposed = make_composed_layout(make_layout((8, 4), (1, 16)), 0, Layout::col_major((4, 8)));
    assert_eq!(size_in_bytes::<f32>(&transposed).as_static(), Some(4 * (7 + 3 * 16 + 1)));
    // The staged layout and one stage of it, as the Hopper GEMM sizes its
    // TMA transaction.
    let staged = tile_to_shape(&atom, (128, 64, 4));
    assert_eq!(size_in_bytes::<F16>(&staged).as_static(), Some(128 * 64 * 4 * 2));
    let stage = staged.slice((.., .., 0));
    assert_eq!(stage.offset().as_leaf().and_then(IndexExpr::as_static), Some(0));
    assert_eq!(size_in_bytes::<F16>(&stage).as_static(), Some(128 * 64 * 2));
    let last = staged.slice((.., .., 3));
    assert_eq!(size_in_bytes::<F16>(&last).as_static(), Some(128 * 64 * 2));
    // A runtime cosize gives a runtime size.
    in_kernel(|| {
        let m = param_extent("m", 8);
        let bytes = size_in_bytes::<F16>(Layout::col_major((m, 4)));
        assert_eq!(bytes.as_static(), None);
        assert!(bytes.is_multiple_of(16));
        let nibbles = size_in_bytes::<I4>(Layout::col_major(m));
        assert!(nibbles.is_multiple_of(4));
    });
}

/// `max_alignment` with runtime leaves counted by their divisors: CuTe
/// DSL's `sgemm.py` vectorizes when `mA.layout[0].max_alignment % 16 == 0`
/// on an extent marked with that divisibility.
#[test]
fn max_alignment_counts_divisor_facts() {
    in_kernel(|| {
        let m = param_extent("m", 16);
        let k = param_extent("k", 1);
        let a = Layout::col_major((m, k, 3));
        assert_eq!(a.mode(0).max_alignment(), 16);
        assert_eq!(a.mode(0).max_alignment() % 16, 0);
        // As for static extents, the alignment need not be a power of two.
        let six = param_extent("six", 6);
        assert_eq!(Layout::col_major(six).max_alignment(), 6);
        assert_eq!(make_layout((six, 2), (1, 8)).max_alignment(), 2);
        // A runtime stride bounds the result by its divisor.
        let s4 = param_extent("s4", 4);
        assert_eq!(make_layout((4, 2), (1, s4)).max_alignment(), 4);
        assert_eq!(make_layout((8, 2), (1, s4)).max_alignment(), 4);
        assert_eq!(make_layout((4, 2), (1, k)).max_alignment(), 1);
        // Only `divisor(m)` elements of a runtime extent are known to be
        // contiguous: with m = 32, (m,4):(1,16) overlaps itself and its
        // contiguous runs are 16 long, so it must not count as (16,4):(1,16)
        // (which coalesces to 64:1).
        assert_eq!(make_layout((m, 4), (1, 16)).max_alignment(), 16);
        assert_eq!(make_layout((m, 2), (2, 32)).max_alignment(), 1);
        assert_eq!(make_layout((2, m), (1, 2)).max_alignment(), 32);
        // Composed: the gcd with the swizzle's run.
        let sw = make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::col_major(m));
        assert_eq!(sw.max_alignment(), 4);
    });
}

/// CuTe's `print_latex` of a 2 x 2 layout, whole.
#[test]
fn print_latex_golden() {
    let want = "% Layout: (2,2):(2,1)
\\documentclass[convert]{standalone}
\\usepackage{tikz}

\\begin{document}
\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]

\\node[fill=black!00] at (0,0) {0};
\\node[fill=black!40] at (0,1) {1};
\\node[fill=black!20] at (1,0) {2};
\\node[fill=black!60] at (1,1) {3};
\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (2,2);

\\node at (0,-1) {\\Large{\\texttt{0}}};
\\node at (1,-1) {\\Large{\\texttt{1}}};
\\node at (-1,0) {\\Large{\\texttt{0}}};
\\node at (-1,1) {\\Large{\\texttt{1}}};
\\end{tikzpicture}
\\end{document}
";
    assert_eq!(print_latex(&Layout::row_major((2, 2))), want);
    // The body alone, for embedding.
    let body = print_latex_body(&Layout::row_major((2, 2)));
    assert!(body.starts_with("\\begin{tikzpicture}") && body.ends_with("\\end{tikzpicture}\n"), "{body}");
    // A rank-1 layout is one column; a colour function picks the fills.
    let white = |_: usize| "white".to_string();
    let col = print_latex_with(&Layout::col_major(3), &LatexOptions { color: Some(&white), body_only: true });
    assert!(col.contains("\\node[fill=white] at (2,0) {2};"), "{col}");
    assert!(col.contains("grid (3,1);"), "{col}");
    // A swizzled layout prints its swizzled offsets.
    let sw = make_composed_layout(make_swizzle(1, 0, 1), 0, Layout::row_major((2, 2)));
    let doc = print_latex(&sw);
    assert!(doc.starts_with("% Layout: S<1,0,1> o 0 o (2,2):(2,1)\n"), "{doc}");
    assert!(doc.contains("at (1,0) {3};") && doc.contains("at (1,1) {2};"), "{doc}");
}

/// `print_latex_tv` of the tutorial's TV layout, a small golden, and the
/// palettes, title and axis labels.
#[test]
fn print_latex_tv_golden() {
    let want = "% Layout TV: (2,2):(2,1)
\\documentclass[convert]{standalone}
\\usepackage{tikz}

\\begin{document}
\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]

\\node[fill={rgb,255:red,175;green,175;blue,255}] at (0,0) {\\shortstack{T0 \\\\ V0}};
\\node[fill={rgb,255:red,175;green,175;blue,255}] at (1,0) {\\shortstack{T0 \\\\ V1}};
\\node[fill={rgb,255:red,175;green,255;blue,175}] at (0,1) {\\shortstack{T1 \\\\ V0}};
\\node[fill={rgb,255:red,175;green,255;blue,175}] at (1,1) {\\shortstack{T1 \\\\ V1}};
\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (2,2);

\\node at (0,-1) {\\Large{\\texttt{0}}};
\\node at (1,-1) {\\Large{\\texttt{1}}};
\\node at (-1,0) {\\Large{\\texttt{0}}};
\\node at (-1,1) {\\Large{\\texttt{1}}};
\\end{tikzpicture}
\\end{document}
";
    assert_eq!(print_latex_tv(&Layout::row_major((2, 2)), (2, 2)), want);
    // The tutorial: 128 threads of 4 values over a 16 x 32 tile.
    let (tile, tv) = make_layout_tv(&make_ordered_layout((4, 32), &[1, 0]), &make_ordered_layout((4, 1), &[1, 0]));
    let doc = print_latex_tv(&tv, tile);
    assert_eq!(doc.matches("\\shortstack").count(), 16 * 32);
    assert!(doc.contains("grid (16,32);"), "{doc}");
    // Palettes: a band per palette strip, under the label.
    let opts = LatexTvOptions { palette: Palette::from_name("rainbow_dual"), body_only: true, ..Default::default() };
    let dual = print_latex_tv_with(&Layout::row_major((2, 2)), (2, 2), &opts);
    assert!(dual.contains("\\fill[{rgb,255:red,255;green,0;blue,0}] (-0.5,-0.5) rectangle (0,0.5);"), "{dual}");
    assert!(dual.contains("\\node at (0,0) {\\shortstack{T0 \\\\ V0}};"), "{dual}");
    assert!(Palette::from_name("sepia").is_none());
    let opts = LatexTvOptions {
        palette: Some(Palette::Bw),
        title: Some("tiled_copy\nA_tv"),
        axis_labels: true,
        ..Default::default()
    };
    let titled = print_latex_tv_with(&Layout::row_major((2, 2)), (2, 2), &opts);
    assert!(titled.contains("\\texttt{tiled\\_copy} \\\\ \\texttt{A\\_tv}"), "{titled}");
    assert!(titled.contains("\\rightarrow"), "{titled}");
    // The colour primitives.
    assert_eq!(tikz_color_rainbow(0, 3, 1.0), "{rgb,255:red,255;green,0;blue,0}");
    assert_eq!(tikz_color_rainbow(1, 3, 1.0), "{rgb,255:red,0;green,255;blue,0}");
    assert_eq!(tikz_color_rainbow(2, 4, 0.5), "{rgb,255:red,128;green,255;blue,255}");
    assert_eq!(tikz_escape("a_b%{c}"), "a\\_b\\%\\{c\\}");
}

/// What `print_latex` refuses: a runtime layout and a rank-3 layout.
#[test]
fn print_latex_refusals() {
    in_kernel(|| {
        let m = param_extent("m", 1);
        let e = panic_text(|| {
            print_latex(&Layout::col_major((m, 4)));
        });
        assert!(e.contains("runtime extents or strides"), "{e}");
    });
    let e = panic_text(|| {
        print_latex(&Layout::col_major((2, 2, 2)));
    });
    assert!(e.contains("rank 3"), "{e}");
}

/// `make_tensor` refuses a composed layout whose inner is a function.
#[test]
fn make_tensor_refuses_a_function_inner() {
    in_kernel(|| {
        let data = param_ptr::<i32>("data");
        let gather = make_composed_layout_fn(|c: Coord| c, 0, make_layout(16, 1));
        let e = panic_text(|| {
            make_tensor(&data, &gather);
        });
        assert!(e.contains("(fn o 0 o 16:1) has a function as its inner"), "{e}");
    });
}

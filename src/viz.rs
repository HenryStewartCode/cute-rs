// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/util/print_latex.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Layouts drawn as LaTeX TikZ pictures (CuTe DSL's `cute.viz`).
//!
//! [`print_latex`] draws a static layout of rank at most 2 as a grid whose
//! cell `(m, n)` holds the offset the layout maps it to; [`print_latex_tv`]
//! draws a thread-value layout over an `M x N` tile, each cell labelled
//! with the first thread and value that own it. Both return a standalone
//! LaTeX document, which `pdflatex` turns into a picture; the
//! `*_body` forms return only the `tikzpicture`, to embed in another
//! document (CuTe DSL's `render_func=`), and [`write_latex`] and
//! [`write_latex_tv`] write either into any [`fmt::Write`].
//!
//! ```
//! use cute::prelude::*;
//! use cute::viz::{print_latex, print_latex_tv};
//!
//! let doc = print_latex(&make_layout((2, 2), (1, 2)));
//! assert!(doc.contains("\\node[fill=black!00] at (0,0) {0};"));
//! let (tile, tv) = make_layout_tv(&make_ordered_layout((4, 32), &[1, 0]), &make_ordered_layout((4, 1), &[1, 0]));
//! assert!(print_latex_tv(&tv, tile).contains("{\\shortstack{T0 \\\\ V0}}"));
//! ```

use std::fmt::{self, Write};

use crate::layout::{ComposedLayout, Coord, IndexExpr, Layout, Shape, Tree};

/// The eight grays CuTe's `TikzColor_BWx8` cycles through.
const BWX8: [u32; 8] = [0, 40, 20, 60, 10, 50, 30, 70];

/// The eight pastels CuTe's `TikzColor_TV` cycles through, by thread.
const TV_COLORS: [&str; 8] = [
    "{rgb,255:red,175;green,175;blue,255}",
    "{rgb,255:red,175;green,255;blue,175}",
    "{rgb,255:red,255;green,255;blue,175}",
    "{rgb,255:red,255;green,175;blue,175}",
    "{rgb,255:red,210;green,210;blue,255}",
    "{rgb,255:red,210;green,255;blue,210}",
    "{rgb,255:red,255;green,255;blue,210}",
    "{rgb,255:red,255;green,210;blue,210}",
];

/// CuTe's `TikzColor_BWx8`: one of eight grays, by offset (the default
/// colouring of [`print_latex`]).
pub fn tikz_color_bwx8(idx: usize) -> String {
    format!("black!{:02}", BWX8[idx % 8])
}

/// CuTe's `TikzColor_White`: every cell white.
pub fn tikz_color_white(_idx: usize) -> String {
    "white".to_string()
}

/// CuTe's `TikzColor_TV`: one of eight pastels, by thread (the default
/// colouring of [`print_latex_tv`]).
pub fn tikz_color_tv(tid: usize, _vid: usize) -> String {
    TV_COLORS[tid % 8].to_string()
}

/// A colour of the hue wheel: hue `index / count` (`count` taken as at
/// least 1), at full value and the given saturation (1 for a vivid colour,
/// less to wash it toward white), as a TikZ `{rgb,255:...}` colour.
pub fn tikz_color_rainbow(index: usize, count: usize, saturation: f64) -> String {
    let count = count.max(1);
    let h = (index % count) as f64 / count as f64 * 6.0;
    let s = saturation.clamp(0.0, 1.0);
    let sector = h.floor();
    let f = h - sector;
    let (p, q, t) = (1.0 - s, 1.0 - s * f, 1.0 - s * (1.0 - f));
    let (r, g, b) = match sector as u32 % 6 {
        0 => (1.0, t, p),
        1 => (q, 1.0, p),
        2 => (p, 1.0, t),
        3 => (p, q, 1.0),
        4 => (t, p, 1.0),
        _ => (1.0, p, q),
    };
    let c = |x: f64| (x * 255.0).round() as u32;
    format!("{{rgb,255:red,{};green,{};blue,{}}}", c(r), c(g), c(b))
}

/// `text` with the characters TeX treats specially escaped, for a title.
pub fn tikz_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\textbackslash{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '_' | '#' | '&' | '%' | '$' | '{' | '}' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

/// A horizontal strip of a [`print_latex_tv`] cell: from `lo` to `hi`, as
/// fractions of the cell's height from its top, filled with `color`.
#[derive(Clone, Debug, PartialEq)]
pub struct Band {
    pub lo: f64,
    pub hi: f64,
    pub color: String,
}

/// How [`print_latex_tv`] fills a cell, by its thread and value: the
/// palettes CuTe DSL names (`PALETTES`), or a function.
#[derive(Clone, Copy)]
pub enum Palette<'a> {
    /// The eight pastels by thread ([`tikz_color_tv`]).
    Pastel,
    /// A hue per thread, spread over the wheel ([`tikz_color_rainbow`]).
    Rainbow,
    /// Two strips: the thread's hue above a paler hue for the value.
    RainbowDual,
    /// No fill: only the labels and the grid.
    White,
    /// The eight grays by thread ([`tikz_color_bwx8`]).
    Bw,
    /// Two strips: the thread's gray above a lighter gray for the value.
    BwDual,
    /// Strips from a function of `(tid, vid, threads, values)`.
    Custom(&'a dyn Fn(usize, usize, usize, usize) -> Vec<Band>),
}

/// The named palettes, as CuTe DSL's `PALETTES` names them.
pub const PALETTES: [(&str, Palette<'static>); 6] = [
    ("pastel", Palette::Pastel),
    ("rainbow", Palette::Rainbow),
    ("rainbow_dual", Palette::RainbowDual),
    ("white", Palette::White),
    ("bw", Palette::Bw),
    ("bw_dual", Palette::BwDual),
];

impl Palette<'_> {
    /// The palette called `name` in [`PALETTES`].
    pub fn from_name(name: &str) -> Option<Palette<'static>> {
        PALETTES.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)
    }

    /// The strips of the cell of thread `tid`, value `vid`, out of
    /// `threads` threads of `values` values.
    pub fn cell(&self, tid: usize, vid: usize, threads: usize, values: usize) -> Vec<Band> {
        let whole = |color: String| vec![Band { lo: 0.0, hi: 1.0, color }];
        let halves = |top: String, bottom: String| {
            vec![Band { lo: 0.0, hi: 0.5, color: top }, Band { lo: 0.5, hi: 1.0, color: bottom }]
        };
        match self {
            Palette::Pastel => whole(tikz_color_tv(tid, vid)),
            Palette::Rainbow => whole(tikz_color_rainbow(tid, threads, 1.0)),
            Palette::RainbowDual => halves(tikz_color_rainbow(tid, threads, 1.0), tikz_color_rainbow(vid, values, 0.4)),
            Palette::White => whole("white".to_string()),
            Palette::Bw => whole(tikz_color_bwx8(tid)),
            Palette::BwDual => halves(tikz_color_bwx8(tid), format!("black!{:02}", BWX8[vid % 8] / 2)),
            Palette::Custom(f) => f(tid, vid, threads, values),
        }
    }
}

/// The options of [`write_latex`]: the fill of a cell by its offset
/// ([`tikz_color_bwx8`] by default) and whether to write only the
/// `tikzpicture` (CuTe DSL's `render_func=`) rather than a standalone
/// document.
#[derive(Clone, Copy, Default)]
pub struct LatexOptions<'a> {
    pub color: Option<&'a dyn Fn(usize) -> String>,
    pub body_only: bool,
}

/// The options of [`write_latex_tv`]: the fill of a cell by its thread and
/// value (`color`, [`tikz_color_tv`] by default), or a [`Palette`] that
/// takes its place; a title above the picture (one line per `\n`, TeX
/// specials escaped); labels for the M (down) and N (right) axes; and
/// whether to write only the `tikzpicture`.
#[derive(Clone, Copy, Default)]
pub struct LatexTvOptions<'a> {
    pub color: Option<&'a dyn Fn(usize, usize) -> String>,
    pub palette: Option<Palette<'a>>,
    pub title: Option<&'a str>,
    pub axis_labels: bool,
    pub body_only: bool,
}

/// The layouts [`print_latex`] draws: a [`Layout`] or a [`ComposedLayout`],
/// static and of rank at most 2.
pub trait LatexLayout {
    /// The layout as it prints in the header comment.
    fn describe(&self) -> String;
    /// The outer layout, whose shape is drawn.
    fn grid(&self) -> &Layout;
    /// The offset at a coordinate of the grid.
    fn offset_at(&self, coord: &Coord) -> IndexExpr;
}

impl LatexLayout for Layout {
    fn describe(&self) -> String {
        self.to_string()
    }
    fn grid(&self) -> &Layout {
        self
    }
    fn offset_at(&self, coord: &Coord) -> IndexExpr {
        self.offset(coord)
    }
}

impl LatexLayout for ComposedLayout {
    fn describe(&self) -> String {
        self.to_string()
    }
    fn grid(&self) -> &Layout {
        self.outer()
    }
    fn offset_at(&self, coord: &Coord) -> IndexExpr {
        self.offset_of(coord)
    }
}

const PICTURE: &str =
    "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n";

fn header(out: &mut impl Write, comment: &str) -> fmt::Result {
    write!(out, "% {comment}\n\\documentclass[convert]{{standalone}}\n\\usepackage{{tikz}}\n\n\\begin{{document}}\n")
}

fn grid_and_labels(out: &mut impl Write, m: usize, n: usize) -> fmt::Result {
    writeln!(out, "\\draw[color=black,thick,shift={{(-0.5,-0.5)}}] (0,0) grid ({m},{n});\n")?;
    for i in 0..m {
        writeln!(out, "\\node at ({i},-1) {{\\Large{{\\texttt{{{i}}}}}}};")?;
    }
    for j in 0..n {
        writeln!(out, "\\node at (-1,{j}) {{\\Large{{\\texttt{{{j}}}}}}};")?;
    }
    Ok(())
}

fn footer(out: &mut impl Write, body_only: bool) -> fmt::Result {
    out.write_str("\\end{tikzpicture}\n")?;
    if !body_only {
        out.write_str("\\end{document}\n")?;
    }
    Ok(())
}

fn static_extent(e: &IndexExpr, what: &str, layout: &str) -> usize {
    e.as_static().unwrap_or_else(|| {
        panic!(
            "{what}: {layout} has runtime extents or strides; only a static layout can be drawn, since it is drawn \
             when the kernel is built"
        )
    })
}

/// The `(M, N)` grid of a layout of rank at most 2 (a rank-1 layout is one
/// column, as CuTe appends `1:0`).
#[track_caller]
fn grid_shape(layout: &Layout, describe: &str, what: &str) -> (usize, usize) {
    assert!(
        layout.rank() <= 2,
        "{what}: {describe} has rank {}, but only a layout of rank 1 or 2 can be drawn (select two modes, or \
         group them)",
        layout.rank()
    );
    let extent = |i: usize| static_extent(&layout.shape().mode(i).product(), what, describe);
    if layout.rank() == 2 {
        (extent(0), extent(1))
    } else {
        (static_extent(&layout.shape().product(), what, describe), 1)
    }
}

/// Write `layout` as LaTeX (CuTe's `print_latex`): each cell `(m, n)` of
/// its grid filled by the colour of its offset and labelled with it, the
/// row and column indices around the grid.
#[track_caller]
pub fn write_latex(out: &mut impl Write, layout: &impl LatexLayout, opts: &LatexOptions) -> fmt::Result {
    let what = "print_latex";
    let describe = layout.describe();
    let grid = layout.grid();
    let (m, n) = grid_shape(grid, &describe, what);
    let two = grid.rank() == 2;
    if !opts.body_only {
        header(out, &format!("Layout: {describe}"))?;
    }
    out.write_str(PICTURE)?;
    for i in 0..m {
        for j in 0..n {
            let coord: Coord = if two { Tree::Branch(vec![i.into(), j.into()]) } else { i.into() };
            let idx = static_extent(&layout.offset_at(&coord), what, &describe);
            let color = match opts.color {
                Some(f) => f(idx),
                None => tikz_color_bwx8(idx),
            };
            writeln!(out, "\\node[fill={color}] at ({i},{j}) {{{idx}}};")?;
        }
    }
    grid_and_labels(out, m, n)?;
    footer(out, opts.body_only)
}

/// `layout` as a standalone LaTeX document (CuTe DSL's
/// `cute.viz.print_latex(x)`), coloured by [`tikz_color_bwx8`]. Panics for a
/// layout with runtime extents or strides, or of rank above 2.
#[track_caller]
pub fn print_latex(layout: &impl LatexLayout) -> String {
    print_latex_with(layout, &LatexOptions::default())
}

/// [`print_latex`] with options: a colouring, or only the `tikzpicture`.
#[track_caller]
pub fn print_latex_with(layout: &impl LatexLayout, opts: &LatexOptions) -> String {
    let mut s = String::new();
    write_latex(&mut s, layout, opts).expect("writing to a String");
    s
}

/// Only the `tikzpicture` of [`print_latex`], to embed in another document
/// (CuTe DSL's `render_func=`).
#[track_caller]
pub fn print_latex_body(layout: &impl LatexLayout) -> String {
    print_latex_with(layout, &LatexOptions { body_only: true, ..Default::default() })
}

/// A coordinate `(m, n)` of the tile: the layout's value itself when it is
/// a pair, or an offset decomposed column-major in the `M x N` tile.
fn tile_coord(v: &Coord, m: usize, n: usize, what: &str, describe: &str) -> (usize, usize) {
    let get = |e: &IndexExpr| static_extent(e, what, describe);
    match v {
        Tree::Leaf(idx) => {
            let idx = get(idx);
            (idx % m, idx / m % n)
        }
        t if t.rank() == 2 => {
            let leaf = |i: usize| match t.mode(i) {
                Tree::Leaf(e) => get(e),
                other => panic!("{what}: {describe} gives the coordinate {other}, not an integer"),
            };
            (leaf(0), leaf(1))
        }
        t => panic!("{what}: {describe} gives {t}, which is neither an offset nor an (m, n) coordinate"),
    }
}

fn fmt_num(x: f64) -> String {
    let s = format!("{x:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".to_string() } else { s.to_string() }
}

/// Write a thread-value layout over an `M x N` tile as LaTeX (CuTe's
/// `print_latex_tv`). `layout_tv` is a static rank-2 layout from `(thread,
/// value)` to an offset in the tile (column-major in `tile_mn`), or to an
/// `(m, n)` coordinate. Each cell is filled and labelled `T<t> V<v>` for the
/// first thread and value that reach it, in thread order then value order.
#[track_caller]
pub fn write_latex_tv(
    out: &mut impl Write,
    layout_tv: &Layout,
    tile_mn: impl Into<Shape>,
    opts: &LatexTvOptions,
) -> fmt::Result {
    let what = "print_latex_tv";
    let describe = layout_tv.to_string();
    let tile = tile_mn.into();
    assert!(
        layout_tv.rank() == 2,
        "{what}: {describe} has rank {}, but a thread-value layout has two modes (threads, values)",
        layout_tv.rank()
    );
    assert!(tile.rank() == 2, "{what}: the tile {tile} is not an (M, N) shape");
    let tile_m = static_extent(&tile.mode(0).product(), what, &format!("the tile {tile}"));
    let tile_n = static_extent(&tile.mode(1).product(), what, &format!("the tile {tile}"));
    let threads = static_extent(&layout_tv.shape().mode(0).product(), what, &describe);
    let values = static_extent(&layout_tv.shape().mode(1).product(), what, &describe);
    if !opts.body_only {
        header(out, &format!("Layout TV: {describe}"))?;
    }
    out.write_str(PICTURE)?;
    let mut filled = vec![false; tile_m * tile_n];
    for tid in 0..threads {
        for vid in 0..values {
            let v = layout_tv.eval(&Tree::Branch(vec![tid.into(), vid.into()]));
            let (m, n) = tile_coord(&v, tile_m, tile_n, what, &describe);
            assert!(m < tile_m && n < tile_n, "{what}: {describe} maps ({tid},{vid}) to ({m},{n}), outside {tile}");
            if std::mem::replace(&mut filled[m + n * tile_m], true) {
                continue;
            }
            let label = format!("{{\\shortstack{{T{tid} \\\\ V{vid}}}}}");
            match opts.palette {
                None => {
                    let color = match opts.color {
                        Some(f) => f(tid, vid),
                        None => tikz_color_tv(tid, vid),
                    };
                    writeln!(out, "\\node[fill={color}] at ({m},{n}) {label};")?;
                }
                Some(p) => {
                    for band in p.cell(tid, vid, threads, values) {
                        let (x0, x1) = (m as f64 - 0.5 + band.lo, m as f64 - 0.5 + band.hi);
                        let (y0, y1) = (n as f64 - 0.5, n as f64 + 0.5);
                        writeln!(
                            out,
                            "\\fill[{}] ({},{}) rectangle ({},{});",
                            band.color,
                            fmt_num(x0),
                            fmt_num(y0),
                            fmt_num(x1),
                            fmt_num(y1)
                        )?;
                    }
                    writeln!(out, "\\node at ({m},{n}) {label};")?;
                }
            }
        }
    }
    grid_and_labels(out, tile_m, tile_n)?;
    if opts.axis_labels {
        let mid_n = fmt_num(tile_n as f64 / 2.0);
        let mid_m = fmt_num(tile_m as f64 / 2.0);
        writeln!(out, "\\node at (-1.75,{mid_n}) {{\\Large $n \\rightarrow$}};")?;
        writeln!(out, "\\node[rotate=90] at ({mid_m},-2) {{\\Large $\\leftarrow m$}};")?;
    }
    if let Some(title) = opts.title {
        let lines: Vec<String> = title.split('\n').map(|l| format!("\\texttt{{{}}}", tikz_escape(l))).collect();
        let at = if opts.axis_labels { -2.25 } else { -1.75 };
        writeln!(
            out,
            "\\node[anchor=south] at ({},{}) {{\\shortstack{{{}}}}};",
            fmt_num(at),
            fmt_num(tile_n as f64 / 2.0),
            lines.join(" \\\\ ")
        )?;
    }
    footer(out, opts.body_only)
}

/// A thread-value layout over an `M x N` tile as a standalone LaTeX
/// document (CuTe DSL's `cute.viz.print_latex_tv(layout_tv, tile_mn)`),
/// coloured by thread ([`tikz_color_tv`]). Everything must be static.
#[track_caller]
pub fn print_latex_tv(layout_tv: &Layout, tile_mn: impl Into<Shape>) -> String {
    print_latex_tv_with(layout_tv, tile_mn, &LatexTvOptions::default())
}

/// [`print_latex_tv`] with options: a colouring or palette, a title, axis
/// labels, or only the `tikzpicture`.
#[track_caller]
pub fn print_latex_tv_with(layout_tv: &Layout, tile_mn: impl Into<Shape>, opts: &LatexTvOptions) -> String {
    let mut s = String::new();
    write_latex_tv(&mut s, layout_tv, tile_mn, opts).expect("writing to a String");
    s
}

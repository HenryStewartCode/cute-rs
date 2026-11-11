//! Layouts and the CuTe layout algebra.
//!
//! A [`Layout`] is a pair of congruent hierarchical tuples, a shape and a
//! stride. It is a function from coordinates to offsets: a coordinate
//! `(c0, c1, ...)` maps to `sum(ci * di)`. A 1-D index is first split into a
//! coordinate in colexicographic (first-mode-fastest) order, so every layout
//! is also a function from `[0, size)` to offsets.
//!
//! The algebra (composition, complement, divides, products, inverses) follows
//! CuTe's definitions; see the CuTe documentation for the theory. Every
//! operation is evaluated when the kernel is built.
//!
//! Shapes are static integers. Strides may be symbolic ([`Layout::with_strides`]),
//! which is how a kernel handles problem sizes that are only known at launch:
//! tiling and partitioning need the tile sizes, which are static, while the
//! stride of a global tensor (its number of columns, say) only enters the
//! offsets that are finally computed. A mode whose extent is only known at
//! launch is declared with any static upper bound that is a multiple of the
//! tile size.

use std::fmt;
use std::ops::RangeFull;

use super::index::IndexExpr;
use super::tree::{IntTuple, Tree, ceil_div, tree_from_tuple};

/// A coordinate whose leaves may be static or runtime values.
pub type Coord = Tree<IndexExpr>;

/// A coordinate for slicing: `None` leaves (written `..` when converting from
/// tuples) keep the corresponding mode, like CuTe's `_`.
pub type SliceCoord = Tree<Option<IndexExpr>>;

/// A tiler for divides and by-mode composition. A leaf layout applies to the
/// whole layout, a tuple applies mode by mode, and `None` (written `..`)
/// leaves a mode untouched, like CuTe's `_`.
pub type Tiler = Tree<Option<Layout>>;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: IntTuple,
    stride: Coord,
}

fn static_strides(t: &IntTuple) -> Coord {
    t.map(&mut |&v| IndexExpr::Static(v))
}

impl Layout {
    pub fn new(shape: impl Into<IntTuple>, stride: impl Into<IntTuple>) -> Layout {
        let shape = shape.into();
        let stride = stride.into();
        assert!(shape.congruent(&stride), "layout shape {shape} and stride {stride} are not congruent");
        Layout { shape, stride: static_strides(&stride) }
    }

    /// A layout whose strides may depend on runtime values, for tensors sized
    /// at launch: `Layout::with_strides((MAX_M, MAX_K), (k, 1))` is a
    /// row-major `(m, k)` matrix with `k` a runtime value. The shape is a
    /// static upper bound on each mode.
    pub fn with_strides(shape: impl Into<IntTuple>, stride: impl Into<Coord>) -> Layout {
        let shape = shape.into();
        let stride = stride.into();
        assert!(shape.congruent(&stride), "layout shape {shape} and stride {stride} are not congruent");
        Layout { shape, stride }
    }

    /// Compact column-major layout of `shape` (CuTe's default).
    pub fn col_major(shape: impl Into<IntTuple>) -> Layout {
        let shape = shape.into();
        let stride = static_strides(&shape.compact_col_major());
        Layout { shape, stride }
    }

    /// Compact row-major layout of `shape`.
    pub fn row_major(shape: impl Into<IntTuple>) -> Layout {
        let shape = shape.into();
        let stride = static_strides(&shape.compact_row_major());
        Layout { shape, stride }
    }

    /// Concatenate layouts as the modes of a new layout.
    pub fn from_modes(modes: impl IntoIterator<Item = Layout>) -> Layout {
        let (shapes, strides): (Vec<_>, Vec<_>) = modes.into_iter().map(|l| (l.shape, l.stride)).unzip();
        Layout { shape: Tree::Branch(shapes), stride: Tree::Branch(strides) }
    }

    /// A flat layout from `(shape, stride)` pairs: no modes becomes `1:0`, one
    /// mode stays a single mode.
    pub fn from_flat(modes: &[(usize, usize)]) -> Layout {
        match modes {
            [] => Layout::new(1, 0),
            [(s, d)] => Layout::new(*s, *d),
            _ => Layout::from_modes(modes.iter().map(|&(s, d)| Layout::new(s, d))),
        }
    }

    /// The hierarchical shape (domain).
    pub fn shape(&self) -> &IntTuple {
        &self.shape
    }

    /// The hierarchical stride, congruent with the shape.
    pub fn stride(&self) -> &Coord {
        &self.stride
    }

    /// The strides as integers. Panics if one depends on a runtime value.
    pub fn static_stride(&self) -> IntTuple {
        self.stride.map(&mut |d| {
            d.as_static()
                .unwrap_or_else(|| panic!("layout {self} has a runtime stride ({d}) where a static one is needed"))
        })
    }

    /// Whether every stride is a static integer.
    pub fn is_static(&self) -> bool {
        self.stride.leaves().iter().all(|d| d.is_static())
    }

    /// Number of top-level modes.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Nesting depth of the shape.
    pub fn depth(&self) -> usize {
        self.shape.depth()
    }

    /// Size of the domain.
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Size of the codomain: one past the largest offset.
    pub fn cosize(&self) -> usize {
        if self.size() == 0 {
            return 0;
        }
        1 + self.flat_modes().iter().map(|&(s, d)| (s - 1) * d).sum::<usize>()
    }

    /// The `i`-th top-level mode as a layout.
    pub fn mode(&self, i: usize) -> Layout {
        Layout { shape: self.shape.mode(i).clone(), stride: self.stride.mode(i).clone() }
    }

    /// All top-level modes.
    pub fn modes(&self) -> Vec<Layout> {
        (0..self.rank()).map(|i| self.mode(i)).collect()
    }

    /// Nested mode selection: `get(&[1, 0])` is mode 0 of mode 1.
    pub fn get(&self, path: &[usize]) -> Layout {
        path.iter().fold(self.clone(), |l, &i| l.mode(i))
    }

    /// `(shape, stride)` of every leaf mode, in order. Static strides only.
    pub fn flat_modes(&self) -> Vec<(usize, usize)> {
        self.shape.flat_leaves().into_iter().zip(self.static_stride().flat_leaves()).collect()
    }

    fn flat_modes_expr(&self) -> Vec<(usize, IndexExpr)> {
        self.shape.flat_leaves().into_iter().zip(self.stride.flat_leaves()).collect()
    }

    fn from_flat_expr(modes: &[(usize, IndexExpr)]) -> Layout {
        let one = |(s, d): &(usize, IndexExpr)| Layout { shape: Tree::Leaf(*s), stride: Tree::Leaf(d.clone()) };
        match modes {
            [] => Layout::new(1, 0),
            [m] => one(m),
            _ => Layout::from_modes(modes.iter().map(one)),
        }
    }

    /// The same layout with all nesting removed.
    pub fn flatten(&self) -> Layout {
        Layout { shape: self.shape.flatten(), stride: self.stride.flatten() }
    }

    /// Evaluate at a 1-D index.
    pub fn call(&self, idx: usize) -> usize {
        self.offset(&Tree::Leaf(IndexExpr::Static(idx))).as_static().unwrap()
    }

    /// Evaluate at a 1-D index, keeping any runtime strides symbolic.
    pub fn offset_at(&self, idx: usize) -> IndexExpr {
        self.offset(&Tree::Leaf(IndexExpr::Static(idx)))
    }

    /// Evaluate at a static (possibly hierarchical) coordinate.
    pub fn crd2idx(&self, coord: &IntTuple) -> usize {
        self.offset(&coord.map(&mut |&c| IndexExpr::Static(c))).as_static().unwrap()
    }

    /// The coordinate of 1-D index `idx`, nested like the shape.
    pub fn idx2crd(&self, idx: usize) -> IntTuple {
        idx2crd(&IndexExpr::Static(idx), &self.shape).map(&mut |c| c.as_static().unwrap())
    }

    /// Evaluate symbolically at `coord`. A leaf coordinate against a
    /// hierarchical mode is a 1-D index into that mode.
    pub fn offset(&self, coord: &Coord) -> IndexExpr {
        offset_impl(coord, &self.shape, &self.stride)
    }

    /// Slice with `coord`: fixed leaves contribute to the returned offset,
    /// `None` leaves keep their mode. The kept modes form the new layout, in
    /// order (CuTe's `slice_and_offset`).
    pub fn slice(&self, coord: &SliceCoord) -> (Layout, IndexExpr) {
        if let Tree::Leaf(None) = coord {
            return (self.clone(), IndexExpr::Static(0));
        }
        let fixed = coord.map(&mut |c| c.clone().unwrap_or(IndexExpr::Static(0)));
        let offset = self.offset(&fixed);
        let mut kept = Vec::new();
        collect_kept(coord, &self.shape, &self.stride, &mut kept);
        (Layout::from_modes(kept), offset)
    }

    /// For each top-level mode, the 1-D coordinate within that mode of the
    /// element this layout maps to `idx`. The layout must be a bijection onto
    /// `[0, size)` (thread layouts usually are).
    pub fn coord_of(&self, idx: impl Into<IndexExpr>) -> Coord {
        let inverse = self.right_inverse();
        assert_eq!(inverse.size(), self.size(), "coord_of: layout {self} is not a bijection onto [0, {})", self.size());
        let linear = inverse.offset(&Tree::Leaf(idx.into()));
        let per_mode = self.shape.product_each();
        match &per_mode {
            Tree::Leaf(_) => Tree::Leaf(linear),
            Tree::Branch(_) => idx2crd(&linear, &per_mode),
        }
    }

    /// Merge adjacent modes that are contiguous with each other and drop
    /// size-1 modes. The result is flat and computes the same function.
    pub fn coalesce(&self) -> Layout {
        let mut out: Vec<(usize, IndexExpr)> = Vec::new();
        for (s, d) in self.flat_modes_expr() {
            if s == 1 {
                continue;
            }
            if let Some(last) = out.last_mut()
                && last.1.clone() * last.0 == d
            {
                last.0 *= s;
                continue;
            }
            out.push((s, d));
        }
        Layout::from_flat_expr(&out)
    }

    /// Coalesce while preserving the structure given by `profile`: a leaf in
    /// the profile coalesces the corresponding sub-layout completely.
    pub fn coalesce_profile(&self, profile: &IntTuple) -> Layout {
        match profile {
            Tree::Leaf(_) => self.coalesce(),
            Tree::Branch(ps) => Layout::from_modes((0..self.rank()).map(|i| match ps.get(i) {
                Some(p) => self.mode(i).coalesce_profile(p),
                None => self.mode(i),
            })),
        }
    }

    /// Replace stride-0 modes with size-1 modes.
    pub fn filter_zeros(&self) -> Layout {
        let mut stride = self.stride.leaves().into_iter();
        let shape =
            self.shape.map(&mut |&s| if matches!(stride.next().unwrap(), IndexExpr::Static(0)) { 1 } else { s });
        Layout { shape, stride: self.stride.clone() }
    }

    /// Remove stride-0 and size-1 modes, then coalesce.
    pub fn filter(&self) -> Layout {
        self.filter_zeros().coalesce()
    }

    /// Functional composition `self ∘ rhs`: `(self ∘ rhs)(c) = self(rhs(c))`.
    /// The result has the shape of `rhs`.
    pub fn compose(&self, rhs: &Layout) -> Layout {
        let lhs = self.coalesce().flat_modes_expr();
        compose_impl(&lhs, &rhs.shape, &rhs.static_stride(), self, rhs)
    }

    /// Composition with a tiler: by mode for tuples, whole for a layout.
    pub fn compose_tiler(&self, tiler: &Tiler) -> Layout {
        match tiler {
            Tree::Leaf(None) => self.clone(),
            Tree::Leaf(Some(b)) => self.compose(b),
            Tree::Branch(ts) => self.by_mode(ts, Layout::compose_tiler, "composition"),
        }
    }

    fn by_mode(&self, ts: &[Tiler], f: impl Fn(&Layout, &Tiler) -> Layout, what: &str) -> Layout {
        assert!(
            ts.len() <= self.rank(),
            "{what}: tiler has {} modes but layout {self} has rank {}",
            ts.len(),
            self.rank()
        );
        Layout::from_modes((0..self.rank()).map(|i| match ts.get(i) {
            Some(t) => f(&self.mode(i), t),
            None => self.mode(i),
        }))
    }

    /// The layout that, concatenated with `self`, covers `[0, cotarget)`
    /// without overlap and in increasing stride order.
    pub fn complement(&self, cotarget: usize) -> Layout {
        let mut modes: Vec<(usize, usize)> =
            self.filter().flat_modes().into_iter().filter(|&(s, d)| s > 1 && d > 0).collect();
        if modes.is_empty() {
            return Layout::new(cotarget.max(1), 1);
        }
        modes.sort_by_key(|&(_, d)| d);
        let mut result = Vec::new();
        let mut current = 1;
        for (s, d) in modes {
            assert!(
                d >= current && d % current == 0,
                "complement: layout {self} is not injective or its strides do not divide each other"
            );
            result.push((d / current, current));
            current = s * d;
        }
        result.push((ceil_div(cotarget, current), current));
        Layout::from_flat(&result).coalesce()
    }

    /// The layout `R` with `self(R(i)) == i` on the largest contiguous range
    /// `[0, size(R))` that `self` covers.
    pub fn right_inverse(&self) -> Layout {
        let flat = self.coalesce().flat_modes();
        let mut rstride = Vec::with_capacity(flat.len());
        let mut running = 1;
        for &(s, _) in &flat {
            rstride.push(running);
            running *= s;
        }
        let mut result = Vec::new();
        let mut next = 1;
        while let Some(i) = flat.iter().position(|&(s, d)| d == next && s > 1) {
            result.push((flat[i].0, rstride[i]));
            next = flat[i].0 * flat[i].1;
        }
        Layout::from_flat(&result).coalesce()
    }

    /// The layout `L` with `L(self(i)) == i` for every `i` in the domain.
    pub fn left_inverse(&self) -> Layout {
        Layout::from_modes([self.clone(), self.complement(self.cosize())]).right_inverse()
    }

    /// The same function on the domain `shape`: `self ∘ make_layout(shape)`
    /// (CuTe's `with_shape`).
    pub fn with_shape(&self, shape: impl Into<IntTuple>) -> Layout {
        self.compose(&Layout::col_major(shape))
    }

    /// The layout of the same offsets counted in units `n` times larger
    /// (CuTe's `upcast<N>`): each non-zero stride is divided by `n` and modes
    /// whose steps fall within one unit shrink, both rounding away from zero
    /// as CuTe's `shape_div` does. Static strides only.
    pub fn upcast(&self, n: usize) -> Layout {
        let stride = self.static_stride();
        let mut strides = stride.flat_leaves().into_iter();
        let mut new_strides = Vec::new();
        let shape = self.shape.map(&mut |&s| {
            let d = strides.next().expect("congruent");
            if d == 0 {
                new_strides.push(0);
                s
            } else {
                new_strides.push(super::tree::shape_div(d, n));
                super::tree::shape_div(s, super::tree::shape_div(n, d))
            }
        });
        Layout::new(shape.clone(), Tree::unflatten(&new_strides, &shape))
    }

    /// Append `mode` as a new last mode (a leaf layout becomes a pair).
    pub fn append(&self, mode: Layout) -> Layout {
        let mut modes = self.as_tuple_modes();
        modes.push(mode);
        Layout::from_modes(modes)
    }

    /// Prepend `mode` as a new first mode.
    pub fn prepend(&self, mode: Layout) -> Layout {
        let mut modes = vec![mode];
        modes.extend(self.as_tuple_modes());
        Layout::from_modes(modes)
    }

    fn as_tuple_modes(&self) -> Vec<Layout> {
        if self.shape.is_leaf() { vec![self.clone()] } else { self.modes() }
    }

    /// Pad with `1:0` modes up to rank `r`.
    pub fn pad_to_rank(&self, r: usize) -> Layout {
        let mut modes = self.as_tuple_modes();
        while modes.len() < r {
            modes.push(Layout::new(1, 0));
        }
        Layout::from_modes(modes)
    }

    /// Group modes `[begin, end)` into a single mode.
    pub fn group(&self, begin: usize, end: usize) -> Layout {
        let modes = self.as_tuple_modes();
        assert!(begin <= end && end <= modes.len(), "group({begin}, {end}) on a rank-{} layout", modes.len());
        let mut out: Vec<Layout> = modes[..begin].to_vec();
        out.push(Layout::from_modes(modes[begin..end].to_vec()));
        out.extend(modes[end..].iter().cloned());
        Layout::from_modes(out)
    }

    /// Select (and reorder) top-level modes.
    pub fn select(&self, modes: &[usize]) -> Layout {
        Layout::from_modes(modes.iter().map(|&i| self.mode(i)))
    }

    /// Divide by `tiler`: each tiled part becomes `(tile, rest)`.
    pub fn logical_divide(&self, tiler: &Tiler) -> Layout {
        match tiler {
            Tree::Leaf(None) => self.clone(),
            Tree::Leaf(Some(b)) => self.compose(&Layout::from_modes([b.clone(), b.complement(self.size())])),
            Tree::Branch(ts) => self.by_mode(ts, Layout::logical_divide, "logical_divide"),
        }
    }

    /// `((tiles...), (rests...))`.
    pub fn zipped_divide(&self, tiler: &Tiler) -> Layout {
        let (tile, rest) = zip2_by(&self.logical_divide(tiler), tiler);
        Layout::from_modes([tile, rest])
    }

    /// `((tiles...), rest0, rest1, ...)`.
    pub fn tiled_divide(&self, tiler: &Tiler) -> Layout {
        let (tile, rest) = zip2_by(&self.logical_divide(tiler), tiler);
        let mut modes = vec![tile];
        modes.extend(rest.as_tuple_modes());
        Layout::from_modes(modes)
    }

    /// `(tile0, tile1, ..., rest0, rest1, ...)`.
    pub fn flat_divide(&self, tiler: &Tiler) -> Layout {
        let (tile, rest) = zip2_by(&self.logical_divide(tiler), tiler);
        let mut modes = tile.as_tuple_modes();
        modes.extend(rest.as_tuple_modes());
        Layout::from_modes(modes)
    }
}

/// Split each tiled mode of a divided layout into its tile and rest parts,
/// gathering tiles into the first result and rests into the second
/// (CuTe's `zip2_by`).
fn zip2_by(layout: &Layout, guide: &Tiler) -> (Layout, Layout) {
    match guide {
        Tree::Leaf(_) => {
            assert_eq!(layout.rank(), 2, "zip2_by: expected a (tile, rest) pair, got {layout}");
            (layout.mode(0), layout.mode(1))
        }
        Tree::Branch(gs) => {
            let mut tiles = Vec::new();
            let mut rests = Vec::new();
            for (i, g) in gs.iter().enumerate() {
                let (t, r) = zip2_by(&layout.mode(i), g);
                tiles.push(t);
                rests.push(r);
            }
            for i in gs.len()..layout.rank() {
                rests.push(layout.mode(i));
            }
            (Layout::from_modes(tiles), Layout::from_modes(rests))
        }
    }
}

fn collect_kept(coord: &SliceCoord, shape: &IntTuple, stride: &Coord, kept: &mut Vec<Layout>) {
    match coord {
        Tree::Leaf(None) => kept.push(Layout { shape: shape.clone(), stride: stride.clone() }),
        Tree::Leaf(Some(_)) => {}
        Tree::Branch(cs) if cs.len() == 1 && shape.is_leaf() => collect_kept(&cs[0], shape, stride, kept),
        Tree::Branch(cs) => {
            assert!(
                !shape.is_leaf() && cs.len() == shape.rank(),
                "slice coordinate {coord:?} does not match shape {shape}"
            );
            for (i, c) in cs.iter().enumerate() {
                collect_kept(c, shape.mode(i), stride.mode(i), kept);
            }
        }
    }
}

fn offset_impl(coord: &Coord, shape: &IntTuple, stride: &Coord) -> IndexExpr {
    match (coord, shape, stride) {
        (Tree::Leaf(c), Tree::Leaf(_), Tree::Leaf(d)) => IndexExpr::product(c.clone(), d.clone()),
        (Tree::Leaf(c), Tree::Branch(ss), Tree::Branch(ds)) => {
            if let IndexExpr::Static(0) = c {
                return IndexExpr::Static(0);
            }
            let mut rest = c.clone();
            let mut acc = IndexExpr::Static(0);
            for (i, (s, d)) in ss.iter().zip(ds).enumerate() {
                let part = if i + 1 == ss.len() {
                    rest.clone()
                } else {
                    let size = s.size();
                    let part = rest.clone() % size;
                    rest = rest / size;
                    part
                };
                acc = acc + offset_impl(&Tree::Leaf(part), s, d);
            }
            acc
        }
        (Tree::Branch(cs), Tree::Leaf(_), Tree::Leaf(_)) if cs.len() == 1 => offset_impl(&cs[0], shape, stride),
        (Tree::Branch(cs), Tree::Branch(ss), Tree::Branch(ds)) => {
            assert_eq!(cs.len(), ss.len(), "coordinate {coord:?} has a different rank than shape {shape}");
            cs.iter().zip(ss).zip(ds).fold(IndexExpr::Static(0), |acc, ((c, s), d)| acc + offset_impl(c, s, d))
        }
        _ => panic!("coordinate {coord:?} is deeper than shape {shape}"),
    }
}

/// Split a 1-D index into a coordinate nested like `shape` (colexicographic,
/// first mode fastest). The index must be smaller than the shape's size.
pub fn idx2crd(idx: &IndexExpr, shape: &IntTuple) -> Coord {
    match shape {
        Tree::Leaf(_) => Tree::Leaf(idx.clone()),
        Tree::Branch(children) => {
            let mut rest = idx.clone();
            let mut out = Vec::with_capacity(children.len());
            for (i, child) in children.iter().enumerate() {
                if i + 1 == children.len() {
                    out.push(idx2crd(&rest, child));
                } else {
                    let size = child.size();
                    out.push(idx2crd(&(rest.clone() % size), child));
                    rest = rest / size;
                }
            }
            Tree::Branch(out)
        }
    }
}

fn compose_impl(lhs: &[(usize, IndexExpr)], shape: &IntTuple, stride: &IntTuple, a: &Layout, b: &Layout) -> Layout {
    match (shape, stride) {
        (Tree::Branch(ss), Tree::Branch(ds)) => {
            Layout::from_modes(ss.iter().zip(ds).map(|(s, d)| compose_impl(lhs, s, d, a, b)))
        }
        (Tree::Leaf(s), Tree::Leaf(d)) => compose_mode(lhs, *s, *d, a, b),
        _ => unreachable!("layout shape and stride are congruent"),
    }
}

/// Compose a flat, coalesced layout with the single mode `s:d`.
fn compose_mode(lhs: &[(usize, IndexExpr)], s: usize, d: usize, a: &Layout, b: &Layout) -> Layout {
    if d == 0 || s == 1 {
        return Layout::new(s, 0);
    }
    if let [(_, d0)] = lhs {
        return Layout::from_flat_expr(&[(s, d0.clone() * d)]);
    }
    let fail = |why: &str| -> ! { panic!("composition {a} ∘ {b} is not defined: {why}") };

    // Divide out the stride: skip the first `d` elements' worth of modes. The
    // last mode is treated as unbounded.
    let mut rest_stride = d;
    let mut modes: Vec<(usize, IndexExpr)> = Vec::new();
    let last = lhs.len() - 1;
    for (i, (si, di)) in lhs.iter().enumerate() {
        let si = *si;
        if i == last {
            modes.push((usize::MAX, di.clone() * rest_stride));
            break;
        }
        if rest_stride == 1 {
            modes.push((si, di.clone()));
        } else if rest_stride.is_multiple_of(si) {
            rest_stride /= si;
        } else if si % rest_stride == 0 {
            modes.push((si / rest_stride, di.clone() * rest_stride));
            rest_stride = 1;
        } else {
            fail(&format!("stride {d} is not compatible with mode size {si}"));
        }
    }

    // Mod out the shape: keep modes until `s` elements are covered.
    let mut rest_shape = s;
    let mut result = Vec::new();
    for (t, e) in modes {
        if rest_shape == 1 {
            break;
        }
        if t == usize::MAX || t % rest_shape == 0 {
            result.push((rest_shape, e));
            rest_shape = 1;
        } else if rest_shape.is_multiple_of(t) {
            result.push((t, e));
            rest_shape /= t;
        } else {
            fail(&format!("shape {s} is not compatible with mode size {t}"));
        }
    }
    let result: Vec<_> = result.into_iter().filter(|&(s, _)| s != 1).collect();
    Layout::from_flat_expr(&result)
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `make_layout(shape, stride)`.
pub fn make_layout(shape: impl Into<IntTuple>, stride: impl Into<IntTuple>) -> Layout {
    Layout::new(shape, stride)
}

/// Merge contiguous modes and drop size-1 modes (same function, flat result).
pub fn coalesce(layout: &Layout) -> Layout {
    layout.coalesce()
}

/// Functional composition `lhs ∘ rhs`; the result has the shape of `rhs`.
pub fn composition(lhs: &Layout, rhs: &Layout) -> Layout {
    lhs.compose(rhs)
}

/// The layout of the offsets in `[0, cotarget)` that `layout` does not reach,
/// ordered so that `(layout, complement)` is injective.
pub fn complement(layout: &Layout, cotarget: usize) -> Layout {
    layout.complement(cotarget)
}

/// `R` with `layout(R(i)) == i` on the largest contiguous range `layout` covers.
pub fn right_inverse(layout: &Layout) -> Layout {
    layout.right_inverse()
}

/// `L` with `L(layout(i)) == i` on the domain of an injective `layout`.
pub fn left_inverse(layout: &Layout) -> Layout {
    layout.left_inverse()
}

/// `(a, complement(a) ∘ b)`: `b` repeated over the codomain of `a`.
pub fn logical_product(a: &Layout, b: &Layout) -> Layout {
    let rest = a.complement(a.size() * b.cosize()).compose(b);
    Layout::from_modes([a.clone(), rest])
}

fn zip_modes(first: &Layout, second: &Layout) -> Layout {
    assert_eq!(first.rank(), second.rank());
    Layout::from_modes((0..first.rank()).map(|i| Layout::from_modes([first.mode(i), second.mode(i)])))
}

/// Repeat `block` according to `tiler`, keeping each block contiguous:
/// mode `i` of the result is `(block_i, tiler_i)`.
pub fn blocked_product(block: &Layout, tiler: &Layout) -> Layout {
    let r = block.rank().max(tiler.rank());
    let p = logical_product(&block.pad_to_rank(r), &tiler.pad_to_rank(r));
    zip_modes(&p.mode(0), &p.mode(1))
}

/// Interleave `block` across `tiler`: mode `i` of the result is
/// `(tiler_i, block_i)`, so neighbouring elements come from different blocks.
pub fn raked_product(block: &Layout, tiler: &Layout) -> Layout {
    let r = block.rank().max(tiler.rank());
    let p = logical_product(&block.pad_to_rank(r), &tiler.pad_to_rank(r));
    zip_modes(&p.mode(1), &p.mode(0))
}

/// Build a thread-value layout from a thread layout and a per-thread value
/// layout (CuTe DSL's `make_layout_tv`). Returns `(tiler_mn, layout_tv)`
/// where `layout_tv` maps `(thread, value)` to a colexicographic index into a
/// tile of shape `tiler_mn`. Thread `t` owns a `val_layout`-shaped block at
/// the position `thr_layout` gives it.
pub fn make_layout_tv(thr_layout: &Layout, val_layout: &Layout) -> (IntTuple, Layout) {
    let layout_mn = raked_product(thr_layout, val_layout);
    let tv = Layout::col_major((thr_layout.size(), val_layout.size()));
    let layout_tv = layout_mn.right_inverse().compose(&tv);
    (layout_mn.shape().product_each(), layout_tv)
}

/// Repeat `block` (column-major) until it fills `shape`. Each mode of
/// `shape` must be a multiple of the corresponding mode of `block`.
pub fn tile_to_shape(block: &Layout, shape: impl Into<IntTuple>) -> Layout {
    let shape = shape.into();
    let block_shape = block.shape().product_each();
    let r = shape.rank();
    assert!(block.rank() <= r, "tile_to_shape: block {block} has higher rank than {shape}");
    let block = block.pad_to_rank(r);
    let reps: Vec<usize> = (0..r)
        .map(|i| {
            let target = shape.mode(i).size();
            let b = if i < block_shape.rank() { block_shape.mode(i).size() } else { 1 };
            assert!(target % b == 0, "tile_to_shape: {shape} is not a multiple of block {block}");
            target / b
        })
        .collect();
    blocked_product(&block, &Layout::col_major(reps))
}

/// `((block...), (tiler...))`: `tiler` repeating `block`, with the two
/// gathered.
pub fn zipped_product(block: &Layout, tiler: &Layout) -> Layout {
    let r = block.rank().max(tiler.rank());
    logical_product(&block.pad_to_rank(r), &tiler.pad_to_rank(r))
}

/// `((block...), tiler0, tiler1, ...)`.
pub fn tiled_product(block: &Layout, tiler: &Layout) -> Layout {
    let z = zipped_product(block, tiler);
    let mut modes = vec![z.mode(0)];
    modes.extend(z.mode(1).modes());
    Layout::from_modes(modes)
}

/// `(block0, ..., tiler0, ...)`.
pub fn flat_product(block: &Layout, tiler: &Layout) -> Layout {
    let z = zipped_product(block, tiler);
    let mut modes = z.mode(0).modes();
    modes.extend(z.mode(1).modes());
    Layout::from_modes(modes)
}

/// The layout of the longest run of elements that correspond in `a` and `b`
/// and are contiguous in `b` (CuTe's `max_common_layout`).
pub fn max_common_layout(a: &Layout, b: &Layout) -> Layout {
    let inverse = b.right_inverse();
    let common = a.compose(&inverse).coalesce();
    let first = common.flat_modes()[0];
    if first.1 == 1 { inverse.compose(&Layout::new(first.0, 1)) } else { Layout::new(1, 0) }
}

/// How many leading elements correspond in `a` and `b` and are contiguous
/// (CuTe's `max_common_vector`): the widest vector copy between them.
pub fn max_common_vector(a: &Layout, b: &Layout) -> usize {
    let first = a.compose(&b.right_inverse()).coalesce().flat_modes()[0];
    if first.1 == 1 { first.0 } else { 1 }
}

/// Keep the modes where `mask` is true (CuTe's `dice`).
pub fn dice(layout: &Layout, mask: &[bool]) -> Layout {
    assert_eq!(layout.rank(), mask.len(), "dice: layout {layout} and mask differ in rank");
    let keep: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
    layout.select(&keep)
}

/// A compact layout of `shape` whose modes are filled in the order given by
/// `order`: smaller values get smaller strides (CuTe's `make_ordered_layout`).
pub fn make_ordered_layout(shape: impl Into<IntTuple>, order: &[usize]) -> Layout {
    let shape = shape.into();
    let sizes = shape.product_each().flat_leaves();
    assert_eq!(sizes.len(), order.len(), "make_ordered_layout: shape {shape} and order differ in rank");
    let mut by_order: Vec<usize> = (0..order.len()).collect();
    by_order.sort_by_key(|&i| order[i]);
    let mut stride = vec![0; order.len()];
    let mut running = 1;
    for i in by_order {
        stride[i] = running;
        running *= sizes[i];
    }
    Layout::from_modes(sizes.iter().zip(&stride).map(|(&s, &d)| Layout::new(s, d)))
}

/// A compact layout with `layout`'s shape and the same relative order of
/// strides (CuTe's `make_layout_like`).
pub fn make_layout_like(layout: &Layout) -> Layout {
    let flat = layout.flatten();
    let order: Vec<usize> = flat.flat_modes().iter().map(|&(_, d)| d).collect();
    make_ordered_layout(flat.shape().clone(), &order)
}

/// Whether two layouts have the same nesting.
pub fn is_congruent(a: &Layout, b: &Layout) -> bool {
    a.shape().congruent(b.shape())
}

/// Append size-1 modes up to rank `r`.
pub fn append_ones(layout: &Layout, r: usize) -> Layout {
    layout.pad_to_rank(r)
}

/// Prepend size-1 modes up to rank `r`.
pub fn prepend_ones(layout: &Layout, r: usize) -> Layout {
    let mut out = layout.clone();
    while out.rank() < r {
        out = out.prepend(Layout::new(1, 0));
    }
    out
}

/// The smallest multiple of `y` that is at least `x`.
pub fn round_up(x: usize, y: usize) -> usize {
    x.div_ceil(y) * y
}

/// Things with a layout that can be re-tiled: layouts and tensors.
pub trait Tileable: Sized {
    fn layout_ref(&self) -> &Layout;
    fn with_layout(&self, layout: Layout) -> Self;
}

impl Tileable for Layout {
    fn layout_ref(&self) -> &Layout {
        self
    }
    fn with_layout(&self, layout: Layout) -> Self {
        layout
    }
}

/// Split `x` into tiles: each tiled mode becomes `(tile, rest)`.
pub fn logical_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    x.with_layout(x.layout_ref().logical_divide(&tiler.into()))
}

/// [`logical_divide`] with the tile modes and the rest modes gathered:
/// `((tiles...), (rests...))`.
pub fn zipped_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    x.with_layout(x.layout_ref().zipped_divide(&tiler.into()))
}

/// [`zipped_divide`] with the rest modes flattened: `((tiles...), rest0, rest1, ...)`.
pub fn tiled_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    x.with_layout(x.layout_ref().tiled_divide(&tiler.into()))
}

/// [`zipped_divide`] with everything flattened: `(tile0, ..., rest0, ...)`.
pub fn flat_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    x.with_layout(x.layout_ref().flat_divide(&tiler.into()))
}

macro_rules! coord_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Coord {
            fn from(v: $t) -> Self { Tree::Leaf(IndexExpr::from(v)) }
        }
        impl From<$t> for SliceCoord {
            fn from(v: $t) -> Self { Tree::Leaf(Some(IndexExpr::from(v))) }
        }
        impl From<$t> for Tiler {
            fn from(v: $t) -> Self { Tree::Leaf(Some(Layout::new(IntTuple::from(v), 1))) }
        }
    )*};
}
coord_from_int!(usize, u32, u64, i32, i64);

impl From<IndexExpr> for Coord {
    fn from(e: IndexExpr) -> Self {
        Tree::Leaf(e)
    }
}

impl From<IndexExpr> for SliceCoord {
    fn from(e: IndexExpr) -> Self {
        Tree::Leaf(Some(e))
    }
}

impl From<RangeFull> for SliceCoord {
    fn from(_: RangeFull) -> Self {
        Tree::Leaf(None)
    }
}

impl From<RangeFull> for Tiler {
    fn from(_: RangeFull) -> Self {
        Tree::Leaf(None)
    }
}

impl From<&Coord> for SliceCoord {
    fn from(c: &Coord) -> Self {
        c.map(&mut |e| Some(e.clone()))
    }
}

impl From<&IntTuple> for Coord {
    fn from(t: &IntTuple) -> Self {
        t.map(&mut |&v| IndexExpr::Static(v))
    }
}

impl From<Layout> for Tiler {
    fn from(l: Layout) -> Self {
        Tree::Leaf(Some(l))
    }
}

impl From<&Layout> for Tiler {
    fn from(l: &Layout) -> Self {
        Tree::Leaf(Some(l.clone()))
    }
}

impl From<&IntTuple> for Tiler {
    fn from(t: &IntTuple) -> Self {
        t.map(&mut |&n| Some(Layout::new(n, 1)))
    }
}

impl From<&Tiler> for Tiler {
    fn from(t: &Tiler) -> Self {
        t.clone()
    }
}

macro_rules! tree_from_seq {
    ($($target:ty),*) => {$(
        impl<T: Into<$target>> From<Vec<T>> for $target {
            fn from(v: Vec<T>) -> Self {
                Tree::Branch(v.into_iter().map(Into::into).collect())
            }
        }
        impl<T: Into<$target>, const N: usize> From<[T; N]> for $target {
            fn from(v: [T; N]) -> Self {
                Tree::Branch(v.into_iter().map(Into::into).collect())
            }
        }
    )*};
}
tree_from_seq!(Coord, SliceCoord, Tiler);

tree_from_tuple!(Coord;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));
tree_from_tuple!(SliceCoord;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));
tree_from_tuple!(Tiler;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F));

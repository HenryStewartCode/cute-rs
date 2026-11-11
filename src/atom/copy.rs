//! Copy atoms and tiled copies.
//!
//! A [`CopyAtom`] describes one copy instruction as CuTe does: which threads
//! take part (`thr_id`) and which elements each thread reads (`src`) and
//! receives (`dst`), as `(thread, value)` layouts over the atom's reference
//! element space. A [`TiledCopy`] tiles an atom over a thread-value layout so
//! that a whole tile is copied by a group of threads; [`ThrCopy`] is one
//! thread's view, which partitions source and destination tensors.

use std::panic::Location;

use crate::atom::mma::TiledMma;
use crate::builder;
use crate::control::{if_, if_else_yield};
use crate::ir::{DType, Literal, MemSpace, Op, ValueId};
use crate::layout::{IndexExpr, IntTuple, Layout, Tileable, Tiler, Tree, make_layout_tv};
use crate::tensor::{Tensor, load_vector_at, store_vector_at};
use crate::value::{Boolean, IntoValue, Numeric, Value};

/// The instruction a copy atom issues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CopyOp {
    /// Plain loads and stores moving `bits` bits per instruction (a multiple
    /// of the element size, at most 128).
    Universal { bits: usize },
    /// `cp.async` from global to shared memory, 4, 8 or 16 bytes per thread.
    /// `bypass_l1` selects `.cg` (16 bytes only); otherwise `.ca`.
    /// Predicated copies zero-fill the destination.
    CpAsync { bytes: usize, bypass_l1: bool },
    /// `ldmatrix.m8n8.x{num}` (`num` in 1, 2, 4) from shared memory into
    /// registers, optionally transposed. 16-bit elements only.
    LdMatrix { num: usize, trans: bool },
}

/// One copy instruction's thread/value mapping.
#[derive(Clone, Debug)]
pub struct CopyAtom<T> {
    op: CopyOp,
    thr_id: Layout,
    src: Layout,
    dst: Layout,
    reference: Layout,
    _t: std::marker::PhantomData<fn() -> T>,
}

impl<T: Numeric> CopyAtom<T> {
    /// The atom for `op` on elements of type `T`.
    pub fn new(op: CopyOp) -> CopyAtom<T> {
        let bits = T::DTYPE.bits().max(8);
        let simple = |vals: usize| {
            let l = Layout::new((1, vals), (0, 1));
            (Layout::new(1, 0), l.clone(), l.clone(), l)
        };
        let (thr_id, src, dst, reference) = match op {
            CopyOp::Universal { bits: b } => {
                assert!(
                    b % bits == 0 && b <= 128 && b >= bits,
                    "Universal copy of {b} bits is not a whole number of {} elements (max 128 bits)",
                    T::DTYPE
                );
                simple(b / bits)
            }
            CopyOp::CpAsync { bytes, bypass_l1 } => {
                assert!(matches!(bytes, 4 | 8 | 16), "cp.async copies 4, 8 or 16 bytes, not {bytes}");
                assert!(!bypass_l1 || bytes == 16, "cp.async.cg (bypass_l1) requires 16 bytes");
                assert!(bytes * 8 >= bits, "cp.async of {bytes} bytes is smaller than one {}", T::DTYPE);
                simple(bytes * 8 / bits)
            }
            CopyOp::LdMatrix { num, trans } => {
                assert_eq!(T::DTYPE.bits(), 16, "ldmatrix moves 16-bit elements, not {}", T::DTYPE);
                let src = match num {
                    1 => Layout::new(((8, 4), 8), ((8, 0), 1)),
                    2 => Layout::new(((16, 2), 8), ((8, 0), 1)),
                    4 => Layout::new((32, 8), (8, 1)),
                    _ => panic!("ldmatrix loads 1, 2 or 4 matrices, not {num}"),
                };
                let dst = match (num, trans) {
                    (1, false) => Layout::new((32, 2), (2, 1)),
                    (2, false) => Layout::new((32, (2, 2)), (2, (1, 64))),
                    (4, false) => Layout::new((32, (2, 4)), (2, (1, 64))),
                    (1, true) => Layout::new(((4, 8), 2), ((16, 1), 8)),
                    (2, true) => Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))),
                    (4, true) => Layout::new(((4, 8), (2, 4)), ((16, 1), (8, 64))),
                    _ => unreachable!(),
                };
                (Layout::new(32, 1), src, dst.clone(), dst)
            }
        };
        CopyAtom { op, thr_id, src, dst, reference, _t: std::marker::PhantomData }
    }

    /// The instruction this atom issues.
    pub fn op(&self) -> CopyOp {
        self.op
    }

    /// The element type the layouts count in (CuTe DSL's `value_type`).
    pub fn value_type(&self) -> DType {
        T::DTYPE
    }

    /// Atom thread id to thread index (`1:0` for one thread, `32:1` for a
    /// warp).
    pub fn thr_id(&self) -> &Layout {
        &self.thr_id
    }

    /// `(thread, value) ->` offset in the atom's elements of the values each
    /// thread supplies (CuTe DSL's `layout_src_tv`).
    pub fn layout_src_tv(&self) -> &Layout {
        &self.src
    }

    /// `(thread, value) ->` offset in the atom's elements of the values each
    /// thread receives (CuTe DSL's `layout_dst_tv`).
    pub fn layout_dst_tv(&self) -> &Layout {
        &self.dst
    }

    /// The reference `(thread, value)` layout that tiled copies are
    /// expressed in (the destination for `ldmatrix`).
    pub fn layout_ref_tv(&self) -> &Layout {
        &self.reference
    }

    /// Threads that cooperate in one instruction (1, or 32 for `ldmatrix`).
    pub fn num_threads(&self) -> usize {
        self.thr_id.size()
    }

    /// Source elements each thread supplies per instruction.
    pub fn num_val_src(&self) -> usize {
        self.src.mode(1).size()
    }

    /// Destination elements each thread receives per instruction.
    pub fn num_val_dst(&self) -> usize {
        self.dst.mode(1).size()
    }

    fn num_val_ref(&self) -> usize {
        self.reference.mode(1).size()
    }
}

/// `make_copy_atom(op)` with the element type as a type parameter.
pub fn make_copy_atom<T: Numeric>(op: CopyOp) -> CopyAtom<T> {
    CopyAtom::new(op)
}

/// A copy atom tiled across threads and values.
#[derive(Clone, Debug)]
pub struct TiledCopy<T> {
    atom: CopyAtom<T>,
    /// `(thread, value) -> ` colexicographic index into the tile.
    layout_tv: Layout,
    /// The tile one application of the tiled copy covers: a shape (a `n:1`
    /// layout per mode), or a layout per mode selecting scattered
    /// coordinates (`make_tiled_copy_c_atom`).
    tiler: Tiler,
}

/// The shape of the tile a tiler selects (CuTe's `shape(Tiler_MN)`).
fn tiler_shape(tiler: &Tiler) -> IntTuple {
    match tiler {
        Tree::Leaf(Some(l)) => l.shape().clone(),
        Tree::Leaf(None) => panic!("cute: a tiled copy's tiler cannot leave a mode untiled"),
        Tree::Branch(ts) => Tree::Branch(ts.iter().map(tiler_shape).collect()),
    }
}

/// A tiled copy from an explicit `(thread, value) ->` tile-coordinate layout
/// and the tile it covers (CuTe DSL's `make_tiled_copy(atom, layout_tv,
/// tiler_mn)`). `tiler_mn` is a shape such as `(32, 16)` or a [`Tiler`] of
/// one layout per mode. The thread count must be a multiple of the atom's
/// threads and each thread's values a multiple of the atom's values; the
/// atom's threads are the first, consecutive thread indices.
#[track_caller]
pub fn make_tiled_copy<T: Numeric>(atom: &CopyAtom<T>, layout_tv: &Layout, tiler_mn: impl Into<Tiler>) -> TiledCopy<T> {
    let tiler = tiler_mn.into();
    assert_eq!(layout_tv.rank(), 2, "cute: make_tiled_copy needs a (thread, value) layout, not {layout_tv}");
    let (threads, values) = (layout_tv.mode(0).size(), layout_tv.mode(1).size());
    assert_eq!(
        threads % atom.num_threads(),
        0,
        "cute: a tiled copy of {threads} threads cannot be built from a {:?} atom of {} threads",
        atom.op,
        atom.num_threads()
    );
    assert_eq!(
        values % atom.num_val_ref(),
        0,
        "cute: each thread's {values} values must be a multiple of the {:?} atom's {}",
        atom.op,
        atom.num_val_ref()
    );
    let tile = tiler_shape(&tiler).size();
    assert!(
        layout_tv.cosize() <= tile,
        "cute: layout_tv {layout_tv} reaches index {} outside the {tile}-element tile",
        layout_tv.cosize() - 1
    );
    TiledCopy { atom: atom.clone(), layout_tv: layout_tv.clone(), tiler }
}

/// Tile `atom` so that each thread of `thr_layout` (a compact `(m, n) ->
/// thread` layout) copies a `val_layout`-shaped block of values (CuTe DSL's
/// `make_tiled_copy_tv`). Warp-level atoms such as `ldmatrix` work as long
/// as each thread's source values end up contiguous in memory.
#[track_caller]
pub fn make_tiled_copy_tv<T: Numeric>(atom: &CopyAtom<T>, thr_layout: &Layout, val_layout: &Layout) -> TiledCopy<T> {
    let (tiler, layout_tv) = make_layout_tv(thr_layout, val_layout);
    make_tiled_copy(atom, &layout_tv, &tiler)
}

/// A tiled copy that produces exactly the A-operand fragments of `mma`.
#[track_caller]
pub fn make_tiled_copy_a<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    make_tiled_copy(atom, &mma.layout_a_tv(), (mma.tile_size(0), mma.tile_size(2)))
}

/// A tiled copy that produces exactly the B-operand fragments of `mma`.
#[track_caller]
pub fn make_tiled_copy_b<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    make_tiled_copy(atom, &mma.layout_b_tv(), (mma.tile_size(1), mma.tile_size(2)))
}

/// A tiled copy matching the C-operand (accumulator) fragments of `mma`.
#[track_caller]
pub fn make_tiled_copy_c<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    make_tiled_copy(atom, &mma.layout_c_tv(), (mma.tile_size(0), mma.tile_size(1)))
}

/// A tiled copy with `atom` that partitions like the source side of
/// `tiled_copy` (CuTe DSL's `make_tiled_copy_S`).
#[track_caller]
pub fn make_tiled_copy_s<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, tiled_copy: &TiledCopy<U>) -> TiledCopy<T> {
    make_tiled_copy(atom, &tiled_copy.layout_src_tv_tiled(), tiled_copy.tiler_mn())
}

/// A tiled copy with `atom` that partitions like the destination side of
/// `tiled_copy` (CuTe DSL's `make_tiled_copy_D`).
#[track_caller]
pub fn make_tiled_copy_d<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, tiled_copy: &TiledCopy<U>) -> TiledCopy<T> {
    make_tiled_copy(atom, &tiled_copy.layout_dst_tv_tiled(), tiled_copy.tiler_mn())
}

/// The smallest tiled copy that can retile `mma`'s accumulators at `atom`'s
/// granularity, for epilogues that store the C tile in pieces (CuTe DSL's
/// `make_tiled_copy_C_atom`). Each thread takes the first `atom` values of
/// its accumulator fragment, in the fragment's order; the tiler picks out
/// the C coordinates those values cover.
#[track_caller]
pub fn make_tiled_copy_c_atom<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    // Truncate the value mode to one atom, keeping the order of the values.
    let layout_c_tv = mma.layout_c_tv();
    let copy_v = atom.num_val_src();
    let mma_v = layout_c_tv.mode(1).size();
    assert!(
        copy_v <= mma_v,
        "cute: the copy atom's {copy_v} source values are more than the {mma_v} accumulators per thread of the MMA"
    );
    let layout_tv = layout_c_tv.compose(&Layout::col_major((layout_c_tv.mode(0).size(), copy_v)));
    // A tiler per mode selecting the C coordinates those values touch.
    let mma_tiler = (mma.tile_size(0), mma.tile_size(1));
    let tiler_0 = Layout::new(mma_tiler, (1, 0)).compose(&layout_tv).filter();
    let tiler_1 = Layout::new(mma_tiler, (0, 1)).compose(&layout_tv).filter();
    let tiler: Tiler = Tree::Branch(vec![Tree::Leaf(Some(tiler_0)), Tree::Leaf(Some(tiler_1))]);
    // Re-express the values as coordinates into the selected tile.
    let tile2mma = Layout::col_major(mma_tiler).compose_tiler(&tiler);
    let layout_tv = tile2mma.left_inverse().compose(&layout_tv);
    make_tiled_copy(atom, &layout_tv, tiler)
}

impl<T: Numeric> TiledCopy<T> {
    /// The tiled atom.
    pub fn atom(&self) -> &CopyAtom<T> {
        &self.atom
    }

    /// `(thread, value) ->` index into the tile (CuTe DSL's
    /// `layout_tv_tiled`).
    pub fn layout_tv_tiled(&self) -> &Layout {
        &self.layout_tv
    }

    /// Same as [`layout_tv_tiled`](Self::layout_tv_tiled).
    pub fn layout_tv(&self) -> &Layout {
        &self.layout_tv
    }

    /// The tile one application covers (CuTe DSL's `tiler_mn`).
    pub fn tiler_mn(&self) -> &Tiler {
        &self.tiler
    }

    /// The shape of that tile.
    pub fn tile_shape(&self) -> IntTuple {
        tiler_shape(&self.tiler)
    }

    /// Threads taking part in the tiled copy (CuTe DSL's `size`).
    pub fn size(&self) -> usize {
        self.layout_tv.mode(0).size()
    }

    /// Same as [`size`](Self::size).
    pub fn num_threads(&self) -> usize {
        self.size()
    }

    /// The atom's element type.
    pub fn value_type(&self) -> DType {
        T::DTYPE
    }

    /// The atom's thread id layout.
    pub fn thr_id(&self) -> &Layout {
        self.atom.thr_id()
    }

    /// The atom's source `(thread, value)` layout.
    pub fn layout_src_tv(&self) -> &Layout {
        self.atom.layout_src_tv()
    }

    /// The atom's destination `(thread, value)` layout.
    pub fn layout_dst_tv(&self) -> &Layout {
        self.atom.layout_dst_tv()
    }

    /// `(thread, value) ->` tile index of the values each thread supplies
    /// (CuTe DSL's `layout_src_tv_tiled`).
    pub fn layout_src_tv_tiled(&self) -> Layout {
        self.layout_trg_tv_tiled(true)
    }

    /// `(thread, value) ->` tile index of the values each thread receives
    /// (CuTe DSL's `layout_dst_tv_tiled`).
    pub fn layout_dst_tv_tiled(&self) -> Layout {
        self.layout_trg_tv_tiled(false)
    }

    fn layout_trg_tv_tiled(&self, src: bool) -> Layout {
        let reference = Layout::from_modes([Layout::col_major(self.tile_shape()), Layout::new(1, 0)]);
        let l = self.tile2thrfrg(&reference, &self.ref2trg(src));
        Layout::from_modes([l.mode(0), l.mode(1)])
    }

    /// Thread `thr_idx`'s view.
    pub fn get_slice(&self, thr_idx: impl Into<IndexExpr>) -> ThrCopy<T> {
        ThrCopy { tiled: self.clone(), thr_idx: thr_idx.into() }
    }

    /// View a thread's fragment (a register tensor whose mode 0 holds that
    /// thread's values in the order of this copy's value layout, such as an
    /// MMA fragment from `partition_fragment_a/b/c`) as a source or
    /// destination of this copy: `((atom_values, rest), rest_m, rest_n, ...)`,
    /// with the values of one copy tile gathered from the fragment's modes
    /// (CuTe's `retile`).
    #[track_caller]
    pub fn retile<U>(&self, tensor: &Tensor<U>) -> Tensor<U> {
        let v = tensor.layout().mode(0).size();
        // (m, n) -> which block of `v` values of a thread holds the element.
        let frg_layout_mn = self.layout_tv.right_inverse().with_shape(self.tile_shape()).upcast(self.size() * v);
        // (atom values, rest values) -> (v, m, n).
        let frg_layout_v = crate::layout::logical_product(&Layout::new(v, 1), &frg_layout_mn.right_inverse())
            .zipped_divide(&Layout::new(self.atom.num_val_ref(), 1).into());
        let mut tile = vec![v];
        match frg_layout_mn.shape().product_each() {
            Tree::Leaf(n) => tile.push(n),
            Tree::Branch(ns) => tile.extend(ns.iter().map(IntTuple::size)),
        }
        let mut t = tensor.clone();
        if t.rank() < tile.len() {
            assert!(
                tile[t.rank()..].iter().all(|&n| n == 1),
                "cute: cannot retile a rank-{} tensor {t:?} for a copy over {} modes",
                t.rank(),
                tile.len() - 1
            );
            let padded = t.layout().pad_to_rank(tile.len());
            t = Tileable::with_layout(&t, padded);
        }
        let r = t.rank();
        let tiled = crate::layout::zipped_divide(&t, Tiler::from(&IntTuple::from(tile)));
        let viewed = tiled.compose(Tree::Branch(vec![Tree::Leaf(Some(frg_layout_v)), Tree::Leaf(None)]));
        let mut rest = vec![Tree::Leaf(Some(IndexExpr::Static(0)))];
        rest.extend((1..r).map(|_| Tree::Leaf(None)));
        viewed.slice(Tree::Branch(vec![Tree::Leaf(None), Tree::Branch(rest)]))
    }

    /// `(source or destination thread, value) ->` reference `(thread, value)`.
    fn ref2trg(&self, src: bool) -> Layout {
        let target = if src { &self.atom.src } else { &self.atom.dst };
        self.atom.reference.right_inverse().compose(target)
    }

    /// `((tile...), (rest...))` to `(thread, value, (rest...))` (CuTe's
    /// `tile2thrfrg`).
    fn tile2thrfrg(&self, tiled: &Layout, ref2trg: &Layout) -> Layout {
        let atom = &self.atom;
        // The threads and values of one atom, and how the atoms tile.
        let atom_tv = self.layout_tv.zipped_divide(&(atom.num_threads(), atom.num_val_ref()).into());
        let trg_tv = atom_tv.compose_tiler(&Tree::Branch(vec![Tree::Leaf(Some(ref2trg.clone())), Tree::Leaf(None)]));
        let zipped = Layout::from_modes([
            Layout::from_modes([trg_tv.get(&[0, 0]), trg_tv.get(&[1, 0])]),
            Layout::from_modes([trg_tv.get(&[0, 1]), trg_tv.get(&[1, 1])]),
        ]);
        let thrval2mn = zipped.coalesce_profile(&(1, (1, 1)).into());
        let tv = tiled.compose_tiler(&Tree::Branch(vec![Tree::Leaf(Some(thrval2mn)), Tree::Leaf(None)]));
        Layout::from_modes([tv.get(&[0, 0]), tv.get(&[0, 1]), tv.mode(1)])
    }

    /// `(thread, value, rest...)` view of a tensor layout, for the source
    /// (`src == true`) or destination side of the atom.
    fn tidfrg(&self, layout: &Layout, src: bool) -> Layout {
        self.tile2thrfrg(&layout.zipped_divide(&self.tiler), &self.ref2trg(src))
    }
}

/// One thread's slice of a [`TiledCopy`].
#[derive(Clone, Debug)]
pub struct ThrCopy<T> {
    tiled: TiledCopy<T>,
    thr_idx: IndexExpr,
}

impl<T: Numeric> ThrCopy<T> {
    fn partition<U>(&self, t: &Tensor<U>, src: bool) -> Tensor<U> {
        let l = self.tiled.tidfrg(t.layout(), src);
        let rest = l.mode(2).rank();
        let viewed = Tileable::with_layout(t, l);
        viewed.slice(Tree::Branch(vec![
            Tree::Leaf(Some(self.thr_idx.clone())),
            Tree::Leaf(None),
            Tree::Branch(vec![Tree::Leaf(None); rest]),
        ]))
    }

    /// This thread's part of a source tile: `(CPY, CPY_M, CPY_N, ...)`.
    pub fn partition_s<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, true)
    }

    /// This thread's part of a destination tile: `(CPY, CPY_M, CPY_N, ...)`.
    pub fn partition_d<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, false)
    }

    /// View a fragment (such as an MMA fragment from `partition_fragment_a/b/c`)
    /// as a destination of this copy ([`TiledCopy::retile`]).
    #[track_caller]
    pub fn retile_d<U>(&self, frag: &Tensor<U>) -> Tensor<U> {
        self.tiled.retile(frag)
    }

    /// View a fragment as a source of this copy ([`TiledCopy::retile`]).
    #[track_caller]
    pub fn retile_s<U>(&self, frag: &Tensor<U>) -> Tensor<U> {
        self.tiled.retile(frag)
    }
}

/// Anything that can drive [`copy`]: an atom or a tiled copy.
pub trait AsCopyAtom<T> {
    fn copy_atom(&self) -> &CopyAtom<T>;
}

impl<T> AsCopyAtom<T> for CopyAtom<T> {
    fn copy_atom(&self) -> &CopyAtom<T> {
        self
    }
}

impl<T> AsCopyAtom<T> for TiledCopy<T> {
    fn copy_atom(&self) -> &CopyAtom<T> {
        &self.atom
    }
}

/// Copy `src` to `dst` with the instruction of `atom`. Both tensors are
/// split into atom-sized chunks along their first mode; the remaining modes
/// are iterated (CuTe's `cute.copy`).
#[track_caller]
pub fn copy<T: Numeric>(atom: &impl AsCopyAtom<T>, src: &Tensor<T>, dst: &Tensor<T>) {
    copy_impl(atom.copy_atom(), src, dst, None);
}

/// [`copy`] with a predicate per element of `src` (only the first element
/// of each atom-sized chunk is consulted). Predicated-off register and
/// memory destinations are left unchanged, except for `cp.async`, which
/// zero-fills them.
#[track_caller]
pub fn copy_if<T: Numeric>(atom: &impl AsCopyAtom<T>, pred: &Tensor<bool>, src: &Tensor<T>, dst: &Tensor<T>) {
    assert_eq!(pred.size(), src.size(), "copy_if: predicate tensor must have one flag per source element");
    copy_impl(atom.copy_atom(), src, dst, Some(pred));
}

/// Element-wise copy that uses the widest vector loads and stores both
/// tensors allow (CuTe DSL's `autovec_copy`).
#[track_caller]
pub fn autovec_copy<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>) {
    assert_eq!(src.size(), dst.size(), "autovec_copy: tensors have different sizes");
    let bits = T::DTYPE.bits().max(8);
    let src1 = as_1d(src);
    let dst1 = as_1d(dst);
    let mut v = 128 / bits;
    while v > 1 && !(vectorizable(&src1, v) && vectorizable(&dst1, v)) {
        v /= 2;
    }
    copy_impl(&CopyAtom::<T>::new(CopyOp::Universal { bits: v * bits }), &src1, &dst1, None);
}

/// Element-wise scalar copy.
#[track_caller]
pub fn basic_copy<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>) {
    assert_eq!(src.size(), dst.size(), "basic_copy: tensors have different sizes");
    let bits = T::DTYPE.bits().max(8);
    copy_impl(&CopyAtom::<T>::new(CopyOp::Universal { bits }), &as_1d(src), &as_1d(dst), None);
}

fn as_1d<T>(t: &Tensor<T>) -> Tensor<T> {
    let flat = t.layout().compose(&Layout::new(t.size(), 1));
    crate::layout::Tileable::with_layout(t, Layout::from_modes([flat]))
}

/// Whether every `v`-element chunk of mode 0 is contiguous and aligned.
fn vectorizable<T>(t: &Tensor<T>, v: usize) -> bool {
    if t.is_register() {
        return true;
    }
    let m0 = t.layout().mode(0);
    if !m0.size().is_multiple_of(v) {
        return false;
    }
    if let Some(sw) = t.swizzle()
        && v > sw.contiguous_run()
    {
        return false;
    }
    let rest = t.size() / m0.size();
    let all = t.layout();
    (0..rest).all(|r| {
        (0..m0.size() / v).all(|c| {
            let start = all.offset_at(r * m0.size() + c * v);
            (0..v).all(|i| all.offset_at(r * m0.size() + c * v + i) == start.clone() + i)
                && (t.base_offset().clone() + start).is_multiple_of(v)
        })
    })
}

/// Split mode 0 of `layout` into `v`-element atom chunks: returns the
/// per-chunk element layout and the static offset of every chunk.
fn chunks(layout: &Layout, v: usize, what: &str) -> (Layout, Vec<IndexExpr>) {
    let m0 = layout.mode(0);
    assert_eq!(m0.size() % v, 0, "copy: {what} mode 0 ({m0}) is not a multiple of the atom's {v} values");
    let divided = m0.logical_divide(&Tree::Leaf(Some(Layout::new(v, 1))));
    let atom = divided.mode(0);
    let mut rest_modes = vec![divided.mode(1)];
    rest_modes.extend((1..layout.rank()).map(|i| layout.mode(i)));
    let rest = Layout::from_modes(rest_modes);
    (atom, (0..rest.size()).map(|c| rest.offset_at(c)).collect())
}

fn contiguous(atom: &Layout) -> bool {
    atom.size() == 1 || atom.coalesce() == Layout::new(atom.size(), 1)
}

#[track_caller]
fn check_vector_access<T>(t: &Tensor<T>, atom: &Layout, off: &IndexExpr, what: &str, op: &CopyOp) {
    let v = atom.size();
    if v == 1 {
        return;
    }
    assert!(
        contiguous(atom),
        "copy: {op:?} needs {v} contiguous {what} elements per thread, but they are laid out as {atom}"
    );
    if let Some(sw) = t.swizzle() {
        assert!(v <= sw.contiguous_run(), "copy: {v}-element vectors are broken up by swizzle {sw}");
    }
    assert!(
        off.is_multiple_of(v),
        "copy: cannot prove that the {what} offset {off} is aligned to {v} elements for {op:?}{}",
        builder::var_value_note(off)
    );
}

#[track_caller]
fn copy_impl<T: Numeric>(atom: &CopyAtom<T>, src: &Tensor<T>, dst: &Tensor<T>, pred: Option<&Tensor<bool>>) {
    let loc = Location::caller();
    let (sv, dv) = (atom.num_val_src(), atom.num_val_dst());
    let (src_atom, src_chunks) = chunks(src.layout(), sv, "source");
    let (dst_atom, dst_chunks) = chunks(dst.layout(), dv, "destination");
    assert_eq!(
        src_chunks.len(),
        dst_chunks.len(),
        "copy: source {:?} and destination {:?} split into different numbers of {:?} chunks",
        src,
        dst,
        atom.op
    );
    let pred_chunks = pred.map(|p| {
        let (_, offs) = chunks(p.layout(), sv, "predicate");
        offs
    });

    for (c, (so, d_o)) in src_chunks.iter().zip(&dst_chunks).enumerate() {
        let s_base = src.base_offset().clone() + so;
        let d_base = dst.base_offset().clone() + d_o;
        let p = match (pred, &pred_chunks) {
            (Some(p), Some(offs)) => Some(p.load_at(p.base_offset().clone() + &offs[c])),
            _ => None,
        };
        emit_chunk(atom, src, &src_atom, s_base, dst, &dst_atom, d_base, p, loc);
    }
}

fn physical(t: &Tensor<impl Sized>, linear: IndexExpr) -> IndexExpr {
    match t.swizzle() {
        Some(sw) => sw.apply_expr(linear),
        None => linear,
    }
}

#[allow(clippy::too_many_arguments)]
#[track_caller]
fn emit_chunk<T: Numeric>(
    atom: &CopyAtom<T>,
    src: &Tensor<T>,
    src_atom: &Layout,
    s_base: IndexExpr,
    dst: &Tensor<T>,
    dst_atom: &Layout,
    d_base: IndexExpr,
    pred: Option<Value<bool>>,
    loc: crate::ir::Loc,
) {
    let op = atom.op;
    match op {
        CopyOp::Universal { .. } => {
            let v = src_atom.size();
            // Addresses are computed before any predicate so that branches
            // only contain the memory access itself.
            let src_index = src.memory_space().map(|_| {
                check_vector_access(src, src_atom, &s_base, "source", &op);
                physical(src, s_base.clone()).into_value().checked()
            });
            let dst_index = dst.memory_space().map(|_| {
                check_vector_access(dst, dst_atom, &d_base, "destination", &op);
                physical(dst, d_base.clone()).into_value().checked()
            });
            let read = || -> Vec<ValueId> {
                match src_index {
                    Some(i) => load_vector_at::<T>(src.buffer().unwrap(), i, v),
                    None => (0..v).map(|k| src.load_at(s_base.clone() + src_atom.call(k)).checked()).collect(),
                }
            };
            match dst_index {
                Some(i) => {
                    let write = || store_vector_at(dst.buffer().unwrap(), i, &read());
                    match pred {
                        Some(p) => if_(p, write),
                        None => write(),
                    }
                }
                None => {
                    let slots: Vec<IndexExpr> = (0..v).map(|k| d_base.clone() + dst_atom.call(k)).collect();
                    let values: Vec<Value<T>> = match pred {
                        None => read().into_iter().map(Value::from_id).collect(),
                        Some(p) => {
                            let old: Vec<Value<T>> = slots.iter().map(|s| old_register_value(dst, s)).collect();
                            if_else_yield(p, || read().into_iter().map(Value::from_id).collect::<Vec<_>>(), || old)
                        }
                    };
                    for (s, v) in slots.into_iter().zip(values) {
                        dst.store_at(s, v);
                    }
                }
            }
        }
        CopyOp::CpAsync { bypass_l1, .. } => {
            assert_eq!(src.memory_space(), Some(MemSpace::Global), "cp.async reads global memory");
            assert_eq!(dst.memory_space(), Some(MemSpace::Shared), "cp.async writes shared memory");
            check_vector_access(src, src_atom, &s_base, "source", &op);
            check_vector_access(dst, dst_atom, &d_base, "destination", &op);
            let src_index = physical(src, s_base).into_value().checked();
            let dst_index = physical(dst, d_base).into_value().checked();
            let pred = pred.map(|p| p.checked());
            let (s, d) = (src.buffer().unwrap(), dst.buffer().unwrap());
            builder::with(|b| {
                b.push(
                    Op::CpAsync {
                        src: s,
                        src_index,
                        dst: d,
                        dst_index,
                        elem: T::DTYPE,
                        elems: src_atom.size() as u32,
                        bypass_l1,
                        pred,
                    },
                    loc,
                )
            });
        }
        CopyOp::LdMatrix { num, trans } => {
            assert!(pred.is_none(), "ldmatrix copies cannot be predicated");
            assert_eq!(src.memory_space(), Some(MemSpace::Shared), "ldmatrix reads shared memory");
            assert!(dst.is_register(), "ldmatrix writes registers");
            check_vector_access(src, src_atom, &s_base, "source", &op);
            let index = physical(src, s_base).into_value().checked();
            let buffer = src.buffer().unwrap();
            let results = builder::with(|b| {
                let results: Vec<ValueId> = (0..2 * num).map(|_| b.new_value(T::DTYPE)).collect();
                b.push(Op::LdMatrix { results: results.clone(), buffer, index, num: num as u32, trans }, loc);
                results
            });
            for (i, r) in results.into_iter().enumerate() {
                dst.store_at(d_base.clone() + dst_atom.call(i), Value::from_id(r));
            }
        }
    }
}

/// The current value of a register slot, or zero if it was never written.
/// This does not count as reading the register in a runtime loop (which
/// the builder reports when the loop then writes it): a predicated copy
/// into a fragment created outside a loop keeps the value from before the
/// loop in the registers it skips, where CuTe DSL keeps the previous
/// iteration's. That differs only if the predicate changes between
/// iterations and the kernel uses the skipped registers, while reporting
/// it would reject the usual pattern of computing on every register and
/// storing under the same predicate.
fn old_register_value<T: Numeric>(t: &Tensor<T>, slot: &IndexExpr) -> Value<T> {
    let i = slot.as_static().expect("register tensors are indexed statically");
    let existing = t.register_storage().and_then(|s| s.peek(i));
    match existing {
        Some(v) => Value::from_id(v),
        None => {
            let loc = Location::caller();
            Value::from_id(builder::with(|b| b.constant(Literal::zero(T::DTYPE), loc)))
        }
    }
}

/// A boolean register fragment shaped like `coords`, filled by `f(coord)`
/// for each element of `coords` (typically a partitioned identity tensor).
#[track_caller]
pub fn make_predicate<F>(coords: &Tensor<crate::tensor::Ident>, mut f: F) -> Tensor<bool>
where
    F: FnMut(&crate::layout::Coord) -> Boolean,
{
    let pred = crate::tensor::make_fragment::<bool>(coords.shape().clone());
    for i in 0..coords.size() {
        let c = coords.coord(i);
        pred.store(i, f(&c));
    }
    pred
}

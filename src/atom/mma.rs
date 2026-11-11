//! MMA atoms and tiled MMAs.
//!
//! An [`MmaAtom`] describes one multiply-accumulate instruction: its
//! `M x N x K` shape and, for each operand, a `(thread, value)` layout giving
//! the coordinate (colexicographic index into `M x K`, `N x K` or `M x N`)
//! that each thread's value holds. A [`TiledMma`] repeats an atom over a
//! layout of atoms (one per warp, for example); [`ThrMma`] is one thread's
//! view, which partitions operand tensors into per-thread fragments.

use std::panic::Location;

use crate::builder;
use crate::ir::{DType, MmaKind, Op};
use crate::layout::{Coord, IndexExpr, IntTuple, Layout, Tiler, Tree};
use crate::tensor::{Tensor, make_fragment_like};
use crate::value::{Numeric, Value};

/// The instruction an MMA atom issues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmaOp {
    /// One thread computing `d = a * b + c` on scalars of the given float
    /// type (CuTe's `UniversalFMA`).
    UniversalFma(DType),
    /// A warp-wide SM80 tensor core `mma.sync`. [`MmaF16BF16Op`] and
    /// [`MmaTF32Op`] pick the kind from CuTe DSL's parameters.
    Sm80(MmaKind),
}

/// CuTe DSL's `cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)`:
/// a warp-level `mma.sync` (SM80 and newer) on f16 or bf16 operands, with A
/// row-major and B column-major (both K-major).
///
/// | A/B  | accumulator | shape_mnk              |
/// |------|-------------|------------------------|
/// | F16  | F16, F32    | (16, 8, 8), (16, 8, 16) |
/// | BF16 | F32         | (16, 8, 8), (16, 8, 16) |
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF16BF16Op {
    ab_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
}

impl MmaF16BF16Op {
    /// The operation, or a panic naming the unsupported parameter (as CuTe
    /// DSL raises at construction).
    #[track_caller]
    pub fn new(ab_dtype: DType, acc_dtype: DType, shape_mnk: (usize, usize, usize)) -> MmaF16BF16Op {
        assert!(
            matches!(ab_dtype, DType::F16 | DType::BF16),
            "cute: MmaF16BF16Op expects the 'ab_dtype' Op parameter to be one of Float16 or BFloat16, not {ab_dtype}"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: MmaF16BF16Op expects the 'acc_dtype' Op parameter to be one of Float16 or Float32, not {acc_dtype}"
        );
        assert!(
            ab_dtype != DType::BF16 || acc_dtype == DType::F32,
            "cute: MmaF16BF16Op expects the 'acc_dtype' Op parameter to be Float32 when 'ab_dtype' is BFloat16"
        );
        assert!(
            matches!(shape_mnk, (16, 8, 8) | (16, 8, 16)),
            "cute: MmaF16BF16Op expects the 'shape_mnk' Op parameter to be one of (16,8,8) or (16,8,16), not {shape_mnk:?}"
        );
        MmaF16BF16Op { ab_dtype, acc_dtype, shape_mnk }
    }

    pub fn ab_dtype(&self) -> DType {
        self.ab_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        self.acc_dtype
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    /// The `mma.sync` instruction it issues.
    pub fn kind(&self) -> MmaKind {
        match (self.ab_dtype, self.acc_dtype, self.shape_mnk.2) {
            (DType::F16, DType::F32, 16) => MmaKind::M16N8K16F32F16,
            (DType::F16, DType::F32, _) => MmaKind::M16N8K8F32F16,
            (DType::F16, _, 16) => MmaKind::M16N8K16F16F16,
            (DType::F16, _, _) => MmaKind::M16N8K8F16F16,
            (_, _, 16) => MmaKind::M16N8K16F32BF16,
            _ => MmaKind::M16N8K8F32BF16,
        }
    }
}

/// CuTe DSL's `cute.nvgpu.warp.MmaTF32Op(shape_mnk)`: a warp-level
/// `mma.sync` (SM80 and newer) on TF32 operands with an f32 accumulator.
/// Operands are `f32` registers; the tensor core ignores their low 13
/// mantissa bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaTF32Op {
    shape_mnk: (usize, usize, usize),
}

impl MmaTF32Op {
    /// The operation for `shape_mnk` `(16, 8, 4)` or `(16, 8, 8)`.
    #[track_caller]
    pub fn new(shape_mnk: (usize, usize, usize)) -> MmaTF32Op {
        assert!(
            matches!(shape_mnk, (16, 8, 4) | (16, 8, 8)),
            "cute: MmaTF32Op expects the 'shape_mnk' Op parameter to be one of (16,8,4) or (16,8,8), not {shape_mnk:?}"
        );
        MmaTF32Op { shape_mnk }
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    /// The `mma.sync` instruction it issues.
    pub fn kind(&self) -> MmaKind {
        if self.shape_mnk.2 == 4 { MmaKind::M16N8K4Tf32 } else { MmaKind::M16N8K8Tf32 }
    }
}

impl From<MmaKind> for MmaOp {
    fn from(kind: MmaKind) -> MmaOp {
        MmaOp::Sm80(kind)
    }
}

impl From<MmaF16BF16Op> for MmaOp {
    fn from(op: MmaF16BF16Op) -> MmaOp {
        MmaOp::Sm80(op.kind())
    }
}

impl From<MmaTF32Op> for MmaOp {
    fn from(op: MmaTF32Op) -> MmaOp {
        MmaOp::Sm80(op.kind())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operand {
    A,
    B,
    C,
}

/// One MMA instruction's shape and fragment layouts.
#[derive(Clone, Debug)]
pub struct MmaAtom {
    op: MmaOp,
    shape_mnk: [usize; 3],
    thr_id: Layout,
    a: Layout,
    b: Layout,
    c: Layout,
}

impl MmaAtom {
    /// The atom for `op`.
    pub fn new(op: MmaOp) -> MmaAtom {
        match op {
            MmaOp::UniversalFma(t) => {
                assert!(t.is_float(), "UniversalFma needs a float type, not {t}");
                let one = Layout::new((1, 1), (0, 0));
                MmaAtom { op, shape_mnk: [1, 1, 1], thr_id: Layout::new(1, 0), a: one.clone(), b: one.clone(), c: one }
            }
            MmaOp::Sm80(kind) => {
                let (m, n, k) = kind.shape_mnk();
                // CuTe's `MMA_Traits<SM80_16x8x*_*_TN>`: thread `(t, g)` is
                // lane `t + 4 * g`; layouts map to the colexicographic index
                // into `M x K`, `N x K` and `M x N`.
                let (a, b) = match kind {
                    MmaKind::M16N8K16F32F16 | MmaKind::M16N8K16F16F16 | MmaKind::M16N8K16F32BF16 => (
                        Layout::new(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
                        Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))),
                    ),
                    MmaKind::M16N8K8F32F16 | MmaKind::M16N8K8F16F16 | MmaKind::M16N8K8F32BF16 => {
                        (Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))), Layout::new(((4, 8), 2), ((16, 1), 8)))
                    }
                    MmaKind::M16N8K4Tf32 => {
                        (Layout::new(((4, 8), 2), ((16, 1), 8)), Layout::new(((4, 8), 1), ((8, 1), 0)))
                    }
                    MmaKind::M16N8K8Tf32 => {
                        (Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))), Layout::new(((4, 8), 2), ((8, 1), 32)))
                    }
                    MmaKind::M8N8K4F64 => {
                        (Layout::new(((4, 8), 1), ((8, 1), 0)), Layout::new(((4, 8), 1), ((8, 1), 0)))
                    }
                };
                let c = match kind {
                    MmaKind::M8N8K4F64 => Layout::new(((4, 8), 2), ((16, 1), 8)),
                    _ => Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))),
                };
                MmaAtom { op, shape_mnk: [m, n, k], thr_id: Layout::new(32, 1), a, b, c }
            }
        }
    }

    /// The instruction this atom issues.
    pub fn op(&self) -> MmaOp {
        self.op
    }

    /// `[M, N, K]` of one instruction.
    pub fn shape_mnk(&self) -> [usize; 3] {
        self.shape_mnk
    }

    /// Threads that cooperate in one instruction (1, or a warp).
    pub fn num_threads(&self) -> usize {
        self.thr_id.size()
    }

    /// `(thread, value) -> (m, k)` index layout of the A operand.
    pub fn layout_a(&self) -> &Layout {
        &self.a
    }

    /// `(thread, value) -> (n, k)` index layout of the B operand.
    pub fn layout_b(&self) -> &Layout {
        &self.b
    }

    /// `(thread, value) -> (m, n)` index layout of the C/D operand.
    pub fn layout_c(&self) -> &Layout {
        &self.c
    }

    /// Element types of A, B and C/D.
    pub fn dtypes(&self) -> (DType, DType, DType) {
        match self.op {
            MmaOp::UniversalFma(t) => (t, t, t),
            MmaOp::Sm80(kind) => kind.dtypes(),
        }
    }

    fn operand(&self, which: Operand) -> (&Layout, usize, usize) {
        let [m, n, k] = self.shape_mnk;
        match which {
            Operand::A => (&self.a, m, k),
            Operand::B => (&self.b, n, k),
            Operand::C => (&self.c, m, n),
        }
    }
}

/// `make_mma_atom(op)`: `op` is an [`MmaOp`], [`MmaF16BF16Op`] or [`MmaTF32Op`].
pub fn make_mma_atom(op: impl Into<MmaOp>) -> MmaAtom {
    MmaAtom::new(op.into())
}

impl From<MmaOp> for MmaAtom {
    fn from(op: MmaOp) -> MmaAtom {
        MmaAtom::new(op)
    }
}

impl From<MmaKind> for MmaAtom {
    fn from(kind: MmaKind) -> MmaAtom {
        MmaAtom::new(MmaOp::Sm80(kind))
    }
}

impl From<MmaF16BF16Op> for MmaAtom {
    fn from(op: MmaF16BF16Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

impl From<MmaTF32Op> for MmaAtom {
    fn from(op: MmaTF32Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

/// An MMA atom repeated over a layout of atoms.
#[derive(Clone, Debug)]
pub struct TiledMma {
    atom: MmaAtom,
    /// `(atom_thread, atom_m, atom_n, atom_k) -> thread index`.
    thr_layout_vmnk: Layout,
    permutation: [Option<Layout>; 3],
}

/// Tile `atom` (an [`MmaAtom`] or any op [`make_mma_atom`] accepts) over
/// `atom_layout` (an `M x N x K` layout of atoms, e.g.
/// `Layout::col_major((2, 2, 1))` for four warps).
pub fn make_tiled_mma(atom: impl Into<MmaAtom>, atom_layout: Layout) -> TiledMma {
    let atom: MmaAtom = atom.into();
    let atom_layout = atom_layout.pad_to_rank(3);
    assert_eq!(atom_layout.rank(), 3, "atom layout must have rank 3 (M, N, K)");
    let t = atom.num_threads();
    let mut modes = vec![atom.thr_id.clone()];
    for i in 0..3 {
        let m = atom_layout.mode(i);
        let stride = m.static_stride().map(&mut |&d| d * t);
        modes.push(Layout::new(m.shape().clone(), stride));
    }
    TiledMma { atom, thr_layout_vmnk: Layout::from_modes(modes), permutation: [None, None, None] }
}

impl TiledMma {
    /// Override the tile size per mode with a permutation layout (CuTe's
    /// `permutation_mnk`), e.g. `Layout::new(32, 1)` to cover 32 rows of M
    /// with one 16-row atom per warp repeated twice.
    pub fn with_permutation_mnk(mut self, perm: [Option<Layout>; 3]) -> TiledMma {
        self.permutation = perm;
        self
    }

    /// The tiled atom.
    pub fn atom(&self) -> &MmaAtom {
        &self.atom
    }

    /// Threads taking part in the tiled MMA.
    pub fn num_threads(&self) -> usize {
        self.thr_layout_vmnk.size()
    }

    /// `(atom thread, atom m, atom n, atom k) ->` thread index.
    pub fn thr_layout_vmnk(&self) -> &Layout {
        &self.thr_layout_vmnk
    }

    /// Tile size of mode `i` (0 = M, 1 = N, 2 = K).
    pub fn tile_size(&self, i: usize) -> usize {
        match &self.permutation[i] {
            Some(p) => p.size(),
            None => self.atom.shape_mnk[i] * self.thr_layout_vmnk.mode(i + 1).size(),
        }
    }

    fn modes_of(which: Operand) -> (usize, usize) {
        match which {
            Operand::A => (0, 2),
            Operand::B => (1, 2),
            Operand::C => (0, 1),
        }
    }

    /// `((ThrV, (ThrM, ThrK)), (FrgV, (RestM, RestK, ...)))` for A, and the
    /// equivalents for B and C (CuTe's `thrfrg_A/B/C`).
    fn thrfrg(&self, layout: &Layout, which: Operand) -> Layout {
        let (i, j) = Self::modes_of(which);
        let (atom_tv, ai, aj) = self.atom.operand(which);
        let perm = |p: &Option<Layout>| Tree::Leaf(p.clone());
        let t = layout.logical_divide(&Tree::Branch(vec![perm(&self.permutation[i]), perm(&self.permutation[j])]));
        let a = t.zipped_divide(&(ai, aj).into());
        let tv = a.compose_tiler(&Tree::Branch(vec![Tree::Leaf(Some(atom_tv.clone())), Tree::Leaf(None)]));
        let thr_i = self.thr_layout_vmnk.mode(i + 1).size();
        let thr_j = self.thr_layout_vmnk.mode(j + 1).size();
        let thr_tile: Tiler = Tree::Branch(vec![
            Tree::Leaf(None),
            Tree::Branch(vec![Tree::Leaf(Some(Layout::new(thr_i, 1))), Tree::Leaf(Some(Layout::new(thr_j, 1)))]),
        ]);
        tv.zipped_divide(&thr_tile)
    }

    /// `(thread index, value) -> ` index into the operand tile, for building
    /// matching tiled copies (CuTe's `get_layoutA_TV` and friends).
    fn layout_tv(&self, which: Operand) -> Layout {
        let (i, j) = Self::modes_of(which);
        let reference = Layout::col_major((self.tile_size(i), self.tile_size(j)));
        let frg = self.thrfrg(&reference, which);
        // Map a (v, m, n, k) thread coordinate onto this operand's (v, i, j).
        let shape: Vec<usize> = (0..4).map(|d| self.thr_layout_vmnk.mode(d).size()).collect();
        let mut stride = vec![0; 4];
        stride[0] = 1;
        stride[i + 1] = shape[0];
        stride[j + 1] = shape[0] * shape[i + 1];
        let to_operand = Layout::new(shape, stride);
        let thread_map = to_operand.compose(&self.thr_layout_vmnk.right_inverse());
        let thr = frg.mode(0).compose(&thread_map);
        Layout::from_modes([thr, frg.mode(1)])
    }

    /// `(thread, value) ->` index into the `(tile_m, tile_k)` A tile.
    pub fn layout_a_tv(&self) -> Layout {
        self.layout_tv(Operand::A)
    }

    /// `(thread, value) ->` index into the `(tile_n, tile_k)` B tile.
    pub fn layout_b_tv(&self) -> Layout {
        self.layout_tv(Operand::B)
    }

    /// `(thread, value) ->` index into the `(tile_m, tile_n)` C tile.
    pub fn layout_c_tv(&self) -> Layout {
        self.layout_tv(Operand::C)
    }

    /// Thread `thr_idx`'s view.
    pub fn get_slice(&self, thr_idx: impl Into<IndexExpr>) -> ThrMma {
        let coord = self.thr_layout_vmnk.coord_of(thr_idx);
        ThrMma { mma: self.clone(), thr_vmnk: coord }
    }
}

/// One thread's slice of a [`TiledMma`].
#[derive(Clone, Debug)]
pub struct ThrMma {
    mma: TiledMma,
    thr_vmnk: Coord,
}

impl ThrMma {
    fn partition<U>(&self, t: &Tensor<U>, which: Operand) -> Tensor<U> {
        let (i, j) = TiledMma::modes_of(which);
        let l = self.mma.thrfrg(t.layout(), which);
        let rest = l.get(&[1, 1]).rank();
        let c = |d: usize| Tree::Leaf(Some(self.thr_vmnk.mode(d).as_leaf().unwrap().clone()));
        let coord = Tree::Branch(vec![
            Tree::Branch(vec![c(0), Tree::Branch(vec![c(i + 1), c(j + 1)])]),
            Tree::Branch(vec![Tree::Leaf(None), Tree::Branch(vec![Tree::Leaf(None); rest])]),
        ]);
        crate::layout::Tileable::with_layout(t, l).slice(coord)
    }

    /// This thread's part of an `(M, K, ...)` tile: `(MMA, MMA_M, MMA_K, ...)`.
    pub fn partition_a<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, Operand::A)
    }

    /// This thread's part of an `(N, K, ...)` tile: `(MMA, MMA_N, MMA_K, ...)`.
    pub fn partition_b<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, Operand::B)
    }

    /// This thread's part of an `(M, N, ...)` tile: `(MMA, MMA_M, MMA_N, ...)`.
    pub fn partition_c<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, Operand::C)
    }

    fn fragment<T: Numeric, U>(&self, t: &Tensor<U>, which: Operand) -> Tensor<T> {
        let (a, b, c) = self.mma.atom.dtypes();
        let expected = match which {
            Operand::A => a,
            Operand::B => b,
            Operand::C => c,
        };
        assert_eq!(
            T::DTYPE,
            expected,
            "{which:?} fragments of {:?} hold {expected}, not {}",
            self.mma.atom.op,
            T::DTYPE
        );
        let p = self.partition(t, which);
        make_fragment_like::<T, U>(&p)
    }

    /// A register fragment for this thread's A operand of tile `t`.
    pub fn partition_fragment_a<T: Numeric, U>(&self, t: &Tensor<U>) -> Tensor<T> {
        self.fragment(t, Operand::A)
    }

    /// A register fragment for this thread's B operand of tile `t`.
    pub fn partition_fragment_b<T: Numeric, U>(&self, t: &Tensor<U>) -> Tensor<T> {
        self.fragment(t, Operand::B)
    }

    /// A register fragment for this thread's accumulators of tile `t`.
    pub fn partition_fragment_c<T: Numeric, U>(&self, t: &Tensor<U>) -> Tensor<T> {
        self.fragment(t, Operand::C)
    }
}

/// `d = a * b + c` over per-thread fragments: `a` is `(MMA, M, K)`, `b` is
/// `(MMA, N, K)`, `c` and `d` are `(MMA, M, N)`. `d` may be `c`.
#[track_caller]
pub fn gemm<TA: Numeric, TB: Numeric, TC: Numeric>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: &Tensor<TA>,
    b: &Tensor<TB>,
    c: &Tensor<TC>,
) {
    let loc = Location::caller();
    let (ta, tb, tc) = mma.atom.dtypes();
    assert_eq!((TA::DTYPE, TB::DTYPE, TC::DTYPE), (ta, tb, tc), "gemm operand types do not match {:?}", mma.atom.op);
    let size = |t: &IntTuple, i: usize| if i < t.rank() { t.mode(i).size() } else { 1 };
    let (m, n, k) = (size(c.shape(), 1), size(c.shape(), 2), size(a.shape(), 2));
    assert_eq!(size(a.shape(), 1), m, "gemm: A has {} M-blocks but C has {m}", size(a.shape(), 1));
    assert_eq!(size(b.shape(), 1), n, "gemm: B has {} N-blocks but C has {n}", size(b.shape(), 1));
    assert_eq!(size(b.shape(), 2), k, "gemm: A has {k} K-blocks but B has {}", size(b.shape(), 2));
    let at = |t: &IntTuple, idx: [usize; 2]| -> crate::layout::SliceCoord {
        let mut modes = vec![Tree::Leaf(None)];
        for (i, &x) in idx.iter().enumerate() {
            if i + 1 < t.rank() {
                modes.push(Tree::Leaf(Some(IndexExpr::Static(x))));
            }
        }
        Tree::Branch(modes)
    };
    for kk in 0..k {
        // Read each operand fragment once per k-block (these are loads when
        // the operands live in shared memory).
        let av: Vec<_> = (0..m).map(|mm| a.slice(at(a.shape(), [mm, kk])).values()).collect();
        let bv: Vec<_> = (0..n).map(|nn| b.slice(at(b.shape(), [nn, kk])).values()).collect();
        for (mm, a_frag) in av.iter().enumerate() {
            for step in 0..n {
                // Serpentine over N so consecutive MMAs share an A fragment.
                let nn = if mm % 2 == 0 { step } else { n - 1 - step };
                let cv = c.slice(at(c.shape(), [mm, nn])).values();
                let dv = emit_mma(&mma.atom, a_frag, &bv[nn], &cv, loc);
                d.slice(at(d.shape(), [mm, nn])).store_values(&dv);
            }
        }
    }
}

fn emit_mma<TA: Numeric, TB: Numeric, TC: Numeric>(
    atom: &MmaAtom,
    a: &[Value<TA>],
    b: &[Value<TB>],
    c: &[Value<TC>],
    loc: crate::ir::Loc,
) -> Vec<Value<TC>> {
    match atom.op {
        MmaOp::UniversalFma(_) => {
            assert!(a.len() == 1 && b.len() == 1 && c.len() == 1, "UniversalFma fragments hold one value");
            let (a, b, c) = (a[0].checked(), b[0].checked(), c[0].checked());
            vec![Value::from_id(builder::with(|bs| bs.fma(a, b, c, loc)))]
        }
        MmaOp::Sm80(kind) => {
            let (na, nb, nc) = kind.fragment_sizes();
            assert_eq!(
                (a.len(), b.len(), c.len()),
                (na, nb, nc),
                "{kind:?} fragments must hold {na}, {nb} and {nc} values"
            );
            let (a, b, c): (Vec<_>, Vec<_>, Vec<_>) = (
                a.iter().map(|v| v.checked()).collect(),
                b.iter().map(|v| v.checked()).collect(),
                c.iter().map(|v| v.checked()).collect(),
            );
            let d = builder::with(|bs| {
                let d: Vec<_> = (0..nc).map(|_| bs.new_value(TC::DTYPE)).collect();
                bs.push(Op::Mma { kind, d: d.clone(), a, b, c }, loc);
                d
            });
            d.into_iter().map(Value::from_id).collect()
        }
    }
}

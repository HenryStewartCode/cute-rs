//! cute_ext's math: [`dot`], [`dot_block_scaled`] and [`dot_sparse`], MMAs
//! accumulating into `c` over whole fragments.

use crate::atom::mma::{MmaAtom, MmaOp, TiledMma, gemm, gemm_sparse, make_tiled_mma};
use crate::builder;
use crate::layout::{Layout, Tileable};
use crate::tensor::Tensor;
use crate::value::Numeric;

/// What [`dot`] and its forms take as the MMA: a [`TiledMma`], or a bare
/// [`MmaAtom`] (as a one-atom tiled MMA), as CuTe DSL's `TiledMma` is an
/// `MmaAtom`.
pub trait AsTiledMma {
    /// The MMA as a tiled MMA.
    fn as_tiled_mma(&self) -> TiledMma;
}

impl AsTiledMma for TiledMma {
    fn as_tiled_mma(&self) -> TiledMma {
        self.clone()
    }
}

impl AsTiledMma for MmaAtom {
    fn as_tiled_mma(&self) -> TiledMma {
        make_tiled_mma(self.clone(), Layout::col_major((1, 1, 1)))
    }
}

fn is_block_scaled(atom: &MmaAtom) -> bool {
    matches!(atom.op(), MmaOp::Sm120(_) | MmaOp::Sm100(crate::ir::UmmaKind { scale: Some(_), .. }))
}

/// `t` with trailing size-1 modes up to rank 3: `(MMA, REST_M, REST_K)`.
fn rank3<T>(t: &Tensor<T>) -> Tensor<T> {
    if t.rank() >= 3 { t.clone() } else { t.map_layout(t.layout().pad_to_rank(3), |l| l.pad_to_rank(3)) }
}

/// Check that `a` `(MMA, M, K)`, `b` `(MMA, N, K)` and `c` `(MMA, M, N)`
/// agree in their tile counts.
#[track_caller]
fn check_shapes<A, B, C>(a: &Tensor<A>, b: &Tensor<B>, c: &Tensor<C>, func: &str) {
    let n = |t: &Layout, i: usize| t.mode(i).size();
    let (la, lb, lc) = (a.layout(), b.layout(), c.layout());
    assert!(
        n(la, 1) == n(lc, 1) && n(lb, 1) == n(lc, 2) && n(la, 2) == n(lb, 2),
        "{func}: operand shapes do not form an MMA: A {} is (MMA, M, K), B {} (MMA, N, K) and C {} (MMA, M, N), so \
         A's mode 1 must match C's mode 1, B's mode 1 C's mode 2, and A's mode 2 B's mode 2",
        a.shape(),
        b.shape(),
        c.shape()
    );
}

/// `c += a * b` over per-thread fragments (cute_ext's `dot(mma_atom, a, b,
/// c)`): `a` is `(MMA, M, K)`, `b` `(MMA, N, K)` and `c` `(MMA, M, N)`;
/// operands of lower rank get trailing modes of size 1. `mma` is a
/// [`TiledMma`] or an [`MmaAtom`]. Block-scaled and sparse MMAs take
/// [`dot_block_scaled`] and [`dot_sparse`].
#[track_caller]
pub fn dot<A: Numeric, B: Numeric, C: Numeric>(mma: &impl AsTiledMma, a: &Tensor<A>, b: &Tensor<B>, c: &Tensor<C>) {
    let _at = builder::LocationScope::new();
    let mma = mma.as_tiled_mma();
    let atom = mma.atom();
    assert!(
        !is_block_scaled(atom),
        "experimental::dot: {:?} is a block-scaled MMA, which takes scale factors: use dot_block_scaled",
        atom.op()
    );
    assert!(
        !atom.is_sparse(),
        "experimental::dot: {:?} is a sparse MMA, which takes metadata: use dot_sparse",
        atom.op()
    );
    let (a, b, c) = (rank3(a), rank3(b), rank3(c));
    check_shapes(&a, &b, &c, "experimental::dot");
    gemm(&mma, &c, &a, &b, &c);
}

/// `c += (a * sfa) * (b * sfb)` for a block-scaled MMA (cute_ext's
/// `dot_block_scaled(mma_atom, a, sfa, b, sfb, c)`): an SM120 block-scaled
/// warp MMA or a tcgen05 block-scaled kind, with the scale-factor operands
/// of [`gemm`]'s `(&a, &sfa)` form.
#[track_caller]
pub fn dot_block_scaled<A: Numeric, SA: Numeric, B: Numeric, SB: Numeric, C: Numeric>(
    mma: &impl AsTiledMma,
    a: &Tensor<A>,
    sfa: &Tensor<SA>,
    b: &Tensor<B>,
    sfb: &Tensor<SB>,
    c: &Tensor<C>,
) {
    let _at = builder::LocationScope::new();
    let mma = mma.as_tiled_mma();
    assert!(
        is_block_scaled(mma.atom()),
        "experimental::dot_block_scaled: {:?} is not a block-scaled MMA: use dot",
        mma.atom().op()
    );
    let (a, b, c) = (rank3(a), rank3(b), rank3(c));
    check_shapes(&a, &b, &c, "experimental::dot_block_scaled");
    gemm(&mma, &c, (&a, sfa), (&b, sfb), &c);
}

/// `c += a * b` for a sparse MMA (cute_ext's `dot_sparse(mma_atom, a, e, b,
/// c)`): `a` is the compressed A fragment and `e` its metadata, as
/// [`gemm_sparse`] takes them.
#[track_caller]
pub fn dot_sparse<A: Numeric, B: Numeric, C: Numeric>(
    mma: &impl AsTiledMma,
    a: &Tensor<A>,
    e: &Tensor<u32>,
    b: &Tensor<B>,
    c: &Tensor<C>,
) {
    let _at = builder::LocationScope::new();
    let mma = mma.as_tiled_mma();
    assert!(mma.atom().is_sparse(), "experimental::dot_sparse: {:?} is not a sparse MMA: use dot", mma.atom().op());
    let (a, b, c) = (rank3(a), rank3(b), rank3(c));
    check_shapes(&a, &b, &c, "experimental::dot_sparse");
    gemm_sparse(&mma, &c, &a, e, &b, &c);
}

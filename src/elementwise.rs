//! The flat machinery under [`TensorSsa`](crate::TensorSsa): element arrays
//! in colexicographic order, broadcasting as index maps, zips and folds.
//! Nothing here knows about nested shapes; the tensor type flattens its
//! shapes to leaf extents before calling in, so another face (a row-major
//! vector type, say) can use the same functions.

/// For every element of a compact column-major array of extents `dst`, the
/// index of the element of a compact column-major array of extents `src` it
/// is broadcast from. `src` and `dst` have the same number of extents, and
/// each `src[k]` is `dst[k]` or 1 (a broadcast extent, whose index is 0).
pub(crate) fn broadcast_map(src: &[usize], dst: &[usize]) -> Vec<usize> {
    assert_eq!(src.len(), dst.len(), "cute: broadcast_map of {src:?} to {dst:?}");
    let mut strides = Vec::with_capacity(src.len());
    let mut stride = 1;
    for (&s, &d) in src.iter().zip(dst) {
        assert!(s == d || s == 1, "cute: broadcast_map of {src:?} to {dst:?}");
        strides.push(if s == 1 { 0 } else { stride });
        stride *= s;
    }
    let n: usize = dst.iter().product();
    (0..n)
        .map(|mut j| {
            let mut i = 0;
            for (&d, &st) in dst.iter().zip(&strides) {
                i += (j % d) * st;
                j /= d;
            }
            i
        })
        .collect()
}

/// `xs[map[k]]` for every `k`.
pub(crate) fn gather<E: Copy>(xs: &[E], map: &[usize]) -> Vec<E> {
    map.iter().map(|&i| xs[i]).collect()
}

/// `f(a[k], b[k])` for every `k`; the arrays have equal lengths.
pub(crate) fn zip_with<A: Copy, B: Copy, C>(a: &[A], b: &[B], mut f: impl FnMut(A, B) -> C) -> Vec<C> {
    assert_eq!(a.len(), b.len(), "cute: zip of {} and {} elements", a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
}

/// The elements combined pairwise in a balanced tree (`log2 n` deep): the
/// halves `[0, n/2)` and `[n/2, n)` folded, then combined. For an
/// associative and commutative operation every order gives the same result.
pub(crate) fn tree_fold<E: Copy>(xs: &[E], f: &mut impl FnMut(E, E) -> E) -> E {
    assert!(!xs.is_empty(), "cute: a fold of no elements");
    if xs.len() == 1 {
        return xs[0];
    }
    let (lo, hi) = xs.split_at(xs.len() / 2);
    let (a, b) = (tree_fold(lo, f), tree_fold(hi, f));
    f(a, b)
}

/// `acc = f(acc, x)` over `xs` in order, from `init`.
pub(crate) fn seq_fold<E: Copy>(init: E, xs: &[E], f: &mut impl FnMut(E, E) -> E) -> E {
    xs.iter().fold(init, |acc, &x| f(acc, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_maps() {
        // (4, 1) to (4, 3): every column repeats the source column.
        assert_eq!(broadcast_map(&[4, 1], &[4, 3]), vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        // (1, 3) to (2, 3): every row repeats the source row.
        assert_eq!(broadcast_map(&[1, 3], &[2, 3]), vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(broadcast_map(&[2, 2], &[2, 2]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn folds() {
        let xs = [1, 2, 3, 4, 5];
        let mut trace = Vec::new();
        let r = tree_fold(&xs, &mut |a, b| {
            trace.push((a, b));
            a + b
        });
        assert_eq!(r, 15);
        assert_eq!(trace, vec![(1, 2), (4, 5), (3, 9), (3, 12)]);
        assert_eq!(seq_fold(10, &xs, &mut |a, b| a - b), -5);
    }
}

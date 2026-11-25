//! The base DSL's swizzle on shared-memory byte addresses (CuTe DSL's `cutlass.Swizzle`,
//! `apply_swizzle`, `load_swizzled` and `store_swizzled`).
//!
//! [`ByteSwizzle`] is a frozen `(bbits, mbase, sshift)` descriptor that acts
//! on byte addresses: `yyy = (addr >> (mbase + sshift)) & ((1 << bbits) -
//! 1)`, and the address becomes `addr ^ (yyy << mbase)`. cute-rs's layout
//! [`Swizzle`] is the same function on whatever offsets it is applied to
//! (elements, for a swizzled layout); [`ByteSwizzle::to_elements`] and
//! [`ByteSwizzle::from_elements`] convert between the two for an element
//! width. Two differences in spelling: the identity is `(0, 0, 0)` here and
//! `S<0,4,3>` in CuTe, and `s128b` is `(3, 4, 3)` on bytes, which is
//! `S<3,3,3>` on Float16 elements.
//!
//! The hardware swizzles (TMA, WGMMA, UMMA) use absolute shared-memory
//! address bits, so a swizzled buffer must start on a multiple of the
//! swizzle's [`period`](ByteSwizzle::period) (1024 bytes for `s128b`;
//! allocate with [`alloc_smem`](crate::alloc_smem) or
//! [`make_smem_tensor_aligned`](crate::make_smem_tensor_aligned) at that
//! alignment), and the functions here refuse a buffer aligned below it.

use std::fmt;

use crate::layout::IndexExpr;
use crate::layout::Swizzle;
use crate::nvgpu::tcgen05::UmmaLayoutType;
use crate::tensor::{Ptr, load_vector_in, ptr_alignment, recast_ptr, store_vector_in};
use crate::tma::TensorMapSwizzle;
use crate::value::{Numeric, Value};
use crate::vector::Vector;

/// CuTe DSL's `cutlass.Swizzle(bbits, mbase, sshift)`: a swizzle of
/// shared-memory byte addresses (see the [module](self) documentation).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ByteSwizzle {
    pub bbits: u32,
    pub mbase: u32,
    pub sshift: u32,
}

impl ByteSwizzle {
    /// No swizzle, `(0, 0, 0)`.
    pub const NONE: ByteSwizzle = ByteSwizzle { bbits: 0, mbase: 0, sshift: 0 };
    /// The 32-byte swizzle, `(1, 4, 3)`.
    pub const S32B: ByteSwizzle = ByteSwizzle { bbits: 1, mbase: 4, sshift: 3 };
    /// The 64-byte swizzle, `(2, 4, 3)`.
    pub const S64B: ByteSwizzle = ByteSwizzle { bbits: 2, mbase: 4, sshift: 3 };
    /// The 128-byte swizzle, `(3, 4, 3)`.
    pub const S128B: ByteSwizzle = ByteSwizzle { bbits: 3, mbase: 4, sshift: 3 };

    /// `(bbits, mbase, sshift)`. Panics if `sshift < bbits` (the bits read
    /// would overlap the bits changed) or the swizzle reaches past bit 31.
    #[track_caller]
    pub fn new(bbits: u32, mbase: u32, sshift: u32) -> ByteSwizzle {
        assert!(
            sshift >= bbits,
            "ByteSwizzle({bbits}, {mbase}, {sshift}): the shift must be at least the number of bits"
        );
        assert!(bbits + mbase + sshift < 32, "ByteSwizzle({bbits}, {mbase}, {sshift}) reaches past bit 31");
        ByteSwizzle { bbits, mbase, sshift }
    }

    /// The preset of a name: `"none"`, `"s32b"`, `"s64b"` or `"s128b"`.
    pub fn from_name(name: &str) -> Result<ByteSwizzle, String> {
        Ok(match name {
            "none" => ByteSwizzle::NONE,
            "s32b" => ByteSwizzle::S32B,
            "s64b" => ByteSwizzle::S64B,
            "s128b" => ByteSwizzle::S128B,
            _ => return Err(format!("{name:?} names no swizzle preset; the presets are none, s32b, s64b and s128b")),
        })
    }

    /// Whether the swizzle changes nothing (`bbits == 0`).
    pub fn is_identity(&self) -> bool {
        self.bbits == 0
    }

    /// The swizzled byte address.
    pub fn apply(&self, addr: u64) -> u64 {
        let yyy = (addr >> (self.mbase + self.sshift)) & ((1 << self.bbits) - 1);
        addr ^ (yyy << self.mbase)
    }

    /// The bytes after which the pattern repeats, `2^(bbits + mbase +
    /// sshift)` (1 without a swizzle): a swizzled buffer starts on a multiple
    /// of it.
    pub fn period(&self) -> usize {
        if self.is_identity() { 1 } else { 1 << (self.bbits + self.mbase + self.sshift) }
    }

    /// The swizzle as cute-rs's [`Swizzle`] on byte offsets (`S<0,4,3>` for
    /// none), the form a swizzled [`Ptr`] carries.
    pub fn to_swizzle(&self) -> Swizzle {
        if self.is_identity() { Swizzle::identity() } else { Swizzle::new(self.bbits, self.mbase, self.sshift as i32) }
    }

    /// The swizzle on the offsets of `elem_bits`-bit elements (`s128b` on
    /// 16-bit elements is `S<3,3,3>`), or why it would split elements.
    pub fn to_elements(&self, elem_bits: usize) -> Result<Swizzle, String> {
        self.to_swizzle().to_elements(elem_bits)
    }

    /// The byte swizzle of a swizzle on the offsets of `elem_bits`-bit
    /// elements (`NONE` for an identity), or why it has none (a negative
    /// shift, or a swizzle that splits bytes).
    pub fn from_elements(sw: Swizzle, elem_bits: usize) -> Result<ByteSwizzle, String> {
        if sw.is_identity() {
            return Ok(ByteSwizzle::NONE);
        }
        if !elem_bits.is_power_of_two() {
            return Err(format!("{elem_bits}-bit elements are not a power of two bits wide"));
        }
        if elem_bits < 8 && sw.base < (8 / elem_bits).trailing_zeros() {
            return Err(format!("{sw} on {elem_bits}-bit elements splits bytes"));
        }
        ByteSwizzle::try_from(sw.to_bytes(elem_bits))
    }

    /// CuTe DSL's `.to(TensorMapSwizzle)`: `none`, `s32b`, `s64b` or `s128b`
    /// for the presets, an error otherwise.
    pub fn to_tensor_map_swizzle(&self) -> Result<TensorMapSwizzle, String> {
        Ok(match *self {
            ByteSwizzle::NONE => TensorMapSwizzle::none,
            ByteSwizzle::S32B => TensorMapSwizzle::s32b,
            ByteSwizzle::S64B => TensorMapSwizzle::s64b,
            ByteSwizzle::S128B => TensorMapSwizzle::s128b,
            _ => return Err(format!("{self} has no tensor-map swizzle (only none, s32b, s64b and s128b do)")),
        })
    }

    /// CuTe DSL's `.to(Tcgen05SmemSwizzle)`: `SwizzleNone`, `Swizzle32B`,
    /// `Swizzle64B` or `Swizzle128B` for the presets, an error otherwise.
    pub fn to_tcgen05_smem_swizzle(&self) -> Result<UmmaLayoutType, String> {
        Ok(match *self {
            ByteSwizzle::NONE => UmmaLayoutType::SwizzleNone,
            ByteSwizzle::S32B => UmmaLayoutType::Swizzle32B,
            ByteSwizzle::S64B => UmmaLayoutType::Swizzle64B,
            ByteSwizzle::S128B => UmmaLayoutType::Swizzle128B,
            _ => return Err(format!("{self} has no tcgen05 descriptor swizzle (only none, s32b, s64b and s128b do)")),
        })
    }
}

/// `Swizzle(b, m, s)`, as CuTe DSL prints it.
impl fmt::Display for ByteSwizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Swizzle({}, {}, {})", self.bbits, self.mbase, self.sshift)
    }
}

impl From<ByteSwizzle> for Swizzle {
    fn from(sw: ByteSwizzle) -> Swizzle {
        sw.to_swizzle()
    }
}

/// A cute-rs [`Swizzle`] read as one on byte addresses (an identity is
/// `NONE`); fails for a negative shift, which the base DSL's swizzle lacks.
impl TryFrom<Swizzle> for ByteSwizzle {
    type Error = String;
    fn try_from(sw: Swizzle) -> Result<ByteSwizzle, String> {
        if sw.is_identity() {
            return Ok(ByteSwizzle::NONE);
        }
        if sw.shift < 0 {
            return Err(format!("{sw} shifts left, which a byte-address swizzle (shift >= bits) cannot"));
        }
        Ok(ByteSwizzle { bbits: sw.bits, mbase: sw.base, sshift: sw.shift as u32 })
    }
}

impl TryFrom<ByteSwizzle> for TensorMapSwizzle {
    type Error = String;
    fn try_from(sw: ByteSwizzle) -> Result<TensorMapSwizzle, String> {
        sw.to_tensor_map_swizzle()
    }
}

impl TryFrom<ByteSwizzle> for UmmaLayoutType {
    type Error = String;
    fn try_from(sw: ByteSwizzle) -> Result<UmmaLayoutType, String> {
        sw.to_tcgen05_smem_swizzle()
    }
}

/// What [`load_swizzled`] returns and [`store_swizzled`] takes: one element
/// (`count` of `None`) or a rank-1 [`Vector`].
#[derive(Clone, Debug)]
pub enum ScalarOrVector<T: Numeric> {
    Scalar(Value<T>),
    Vector(Vector<T>),
}

impl<T: Numeric> ScalarOrVector<T> {
    /// The element. Panics for a vector.
    #[track_caller]
    pub fn scalar(self) -> Value<T> {
        match self {
            ScalarOrVector::Scalar(v) => v,
            ScalarOrVector::Vector(v) => panic!("cute::prims: a vector of {} elements, not a scalar", v.size()),
        }
    }

    /// The vector. Panics for a scalar.
    #[track_caller]
    pub fn vector(self) -> Vector<T> {
        match self {
            ScalarOrVector::Vector(v) => v,
            ScalarOrVector::Scalar(_) => panic!("cute::prims: a scalar, not a vector"),
        }
    }
}

impl<T: Numeric> From<Value<T>> for ScalarOrVector<T> {
    fn from(v: Value<T>) -> Self {
        ScalarOrVector::Scalar(v)
    }
}

impl<T: Numeric> From<Vector<T>> for ScalarOrVector<T> {
    fn from(v: Vector<T>) -> Self {
        ScalarOrVector::Vector(v)
    }
}

impl<T: Numeric> From<&Vector<T>> for ScalarOrVector<T> {
    fn from(v: &Vector<T>) -> Self {
        ScalarOrVector::Vector(v.clone())
    }
}

/// The pointer swizzled by `sw`, checked: a plain (unswizzled) pointer into
/// shared memory whose buffer is aligned to the swizzle's period.
#[track_caller]
fn swizzled<T: Numeric>(ptr: &Ptr<T>, sw: ByteSwizzle, func: &str) -> Ptr<T> {
    assert!(
        ptr.swizzle().is_none_or(|s| s.is_identity()),
        "{func}: the pointer already carries the swizzle {}; pass the plain pointer",
        ptr.swizzle().unwrap()
    );
    assert!(
        ptr.memspace() == crate::tensor::AddressSpace::Smem,
        "{func}: the swizzle is on shared-memory addresses, but the pointer is in {:?}",
        ptr.memspace()
    );
    assert!(
        T::DTYPE.bits() >= 8 && !T::DTYPE.is_word_packed(),
        "{func}: {} elements are narrower than a byte; load whole bytes",
        T::DTYPE.cute_name()
    );
    let align = crate::builder::with(|b| b.buffer(ptr.buffer).align);
    assert!(
        align >= sw.period(),
        "{func}: {sw} acts on absolute shared-memory addresses and repeats every {} bytes, but the buffer is only \
         known to be {align}-byte aligned (allocate it aligned to {}: alloc_smem(count, {}))",
        sw.period(),
        sw.period(),
        sw.period()
    );
    recast_ptr::<T, T>(ptr, Some(sw.to_swizzle()))
}

/// CuTe DSL's `apply_swizzle(ptr, swizzle)`: the plain pointer to where the
/// element `ptr` names lives once `sw` permutes the shared-memory byte
/// addresses (`ptr` itself without a swizzle), for instructions that take an
/// address. `ptr` is a plain shared-memory pointer whose buffer is aligned to
/// the swizzle's [`period`](ByteSwizzle::period).
#[track_caller]
pub fn apply_swizzle<T: Numeric>(ptr: &Ptr<T>, sw: ByteSwizzle) -> Ptr<T> {
    if sw.is_identity() {
        return ptr.clone();
    }
    swizzled(ptr, sw, "apply_swizzle").physical_ptr()
}

/// The element offsets of the contiguous pieces `count` elements from `ptr`
/// are moved in: whole 16-byte granules (no wider than the swizzle's
/// unchanged run or `align`), narrower pieces where the start or the end is
/// not on one.
fn pieces(count: usize, elem_bytes: usize, align: usize, sw: ByteSwizzle) -> Vec<(usize, usize)> {
    let mut g = 16.min(align.max(elem_bytes));
    if !sw.is_identity() {
        g = g.min(1 << sw.mbase);
    }
    let max = (g / elem_bytes).max(1);
    let mut out = Vec::new();
    let mut i = 0;
    while i < count {
        let mut p = max;
        while p > 1 && (i + p > count || !(i * elem_bytes).is_multiple_of(p * elem_bytes)) {
            p /= 2;
        }
        out.push((i, p));
        i += p;
    }
    out
}

/// The alignment of `ptr` in bytes: `alignment` when given (and the pointer's
/// offset assumed to honour it), what is provable otherwise.
fn offset_and_align<T: Numeric>(ptr: &Ptr<T>, alignment: Option<usize>, func: &str) -> (IndexExpr, usize) {
    let eb = T::DTYPE.bytes();
    match alignment {
        Some(a) => {
            assert!(
                a.is_power_of_two() && a >= eb,
                "{func}: the alignment {a} is not a power of two of whole elements"
            );
            let off =
                if a > eb { crate::value::assume(ptr.offset.clone(), Some(a / eb), false) } else { ptr.offset.clone() };
            (off, a)
        }
        None => (ptr.offset.clone(), ptr_alignment(ptr.buffer, &ptr.offset, eb)),
    }
}

/// CuTe DSL's `load_swizzled(ptr, swizzle, alignment, count=...)`: the
/// element at `ptr` (`count` of `None`) or a [`Vector`] of the `count`
/// elements from `ptr`, read from where `sw` puts them in shared memory.
/// `alignment` is the pointer's alignment in bytes when more than can be
/// proven. The swizzle leaves each 16-byte granule (of the presets) in one
/// piece, so the elements are read a granule at a time with vector loads
/// (`ld.shared.v4.b32` for 16 bytes); a scalar or a count of 1 reads one
/// element.
#[track_caller]
pub fn load_swizzled<T: Numeric>(
    ptr: &Ptr<T>,
    sw: ByteSwizzle,
    alignment: Option<usize>,
    count: Option<usize>,
) -> ScalarOrVector<T> {
    let func = "load_swizzled";
    let Some(n) = count else {
        return ScalarOrVector::Scalar(apply_swizzle(ptr, sw).load());
    };
    assert!(n > 0, "{func}: a count of 0 elements");
    let base = if sw.is_identity() { ptr.clone() } else { swizzled(ptr, sw, func) };
    let (off, align) = offset_and_align(ptr, alignment, func);
    let base = base.with_offset(off);
    let mut values = Vec::with_capacity(n);
    for (i, p) in pieces(n, T::DTYPE.bytes(), align, sw) {
        let phys = base.offset(i).physical_ptr();
        values.extend(
            load_vector_in::<T>(phys.buffer, phys.offset.clone(), p, phys.peer).into_iter().map(Value::from_id),
        );
    }
    ScalarOrVector::Vector(Vector::new(n, values))
}

/// CuTe DSL's `store_swizzled(ptr, value, swizzle, alignment)`: the inverse
/// of [`load_swizzled`], writing one element or the elements of a rank-1
/// [`Vector`] (a granule at a time) where `sw` puts them.
#[track_caller]
pub fn store_swizzled<T: Numeric>(
    ptr: &Ptr<T>,
    value: impl Into<ScalarOrVector<T>>,
    sw: ByteSwizzle,
    alignment: Option<usize>,
) {
    let func = "store_swizzled";
    let v = match value.into() {
        ScalarOrVector::Scalar(v) => return apply_swizzle(ptr, sw).store(v),
        ScalarOrVector::Vector(v) => v,
    };
    assert!(v.rank() == 1, "{func}: takes a scalar or a rank-1 vector, not one of shape {:?}", v.shape());
    let base = if sw.is_identity() { ptr.clone() } else { swizzled(ptr, sw, func) };
    let (off, align) = offset_and_align(ptr, alignment, func);
    let base = base.with_offset(off);
    let ids: Vec<_> = v.values().iter().map(|x| x.checked()).collect();
    for (i, p) in pieces(v.size(), T::DTYPE.bytes(), align, sw) {
        let phys = base.offset(i).physical_ptr();
        store_vector_in(phys.buffer, phys.offset.clone(), &ids[i..i + p], phys.peer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granules() {
        // 64 Float16 values on a 16-byte-aligned base: eight granules of 8.
        assert_eq!(pieces(64, 2, 1024, ByteSwizzle::S128B).len(), 8);
        // A ragged count: 8, then 2 and 1.
        assert_eq!(pieces(11, 2, 1024, ByteSwizzle::S128B), vec![(0, 8), (8, 2), (10, 1)]);
        // An 8-byte-aligned base: 4 at a time.
        assert_eq!(pieces(8, 2, 8, ByteSwizzle::NONE), vec![(0, 4), (4, 4)]);
    }
}

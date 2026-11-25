//! Pointers made from integer addresses and integers taken from pointers
//! (CuTe DSL's `make_ptr`, `Pointer.toint`, `Pointer.tospace`, `Pointer &
//! mask`, `Pointer.align` and `cutlass.inttoptr`).
//!
//! A [`Ptr`] into a buffer (a parameter, shared memory, a per-thread array)
//! has an address: [`Ptr::toint`] gives it, an `Int64` for global memory and
//! an `Int32` offset in the block's shared-memory window or the thread's
//! local window. [`make_ptr`] goes the other way: from an integer address
//! and an address space to a pointer, whose loads, stores, atomics and the
//! copies built on them go through that address (`llvm.inttoptr` and
//! `llvm.load`/`llvm.store`, generic instructions for a generic address).
//! Its alignment is the one it is made with, which `make_ptr` promises to
//! the compiler (`cute.assume`) and the simulator checks.
//!
//! The simulator gives every buffer an address ([`SIM_GLOBAL_BASE`] and the
//! windows) and resolves each access through a raw pointer to the buffer it
//! falls in, keeping its bounds, alignment and race checks; an address in
//! no buffer is reported as a wild address at the line that made the
//! access.
//!
//! [`SIM_GLOBAL_BASE`]: crate::sim::SIM_GLOBAL_BASE

use std::marker::PhantomData;
use std::ops::BitAnd;
use std::panic::Location;

use crate::builder;
use crate::ir::{BinaryOp, BufferDecl, DType, Literal, MemSpace, Op, ValueId};
use crate::layout::IndexExpr;
use crate::tensor::{AddressSpace, Ptr, index_operand, ptr_alignment, storage_decl};
use crate::value::{AnyValue, Int32, Int64, Numeric, Value};

/// The bytes of an element of `T` as an alignment: its size, at least 1.
fn elem_align<T: Numeric>() -> usize {
    T::DTYPE.storage_bytes(1).max(1)
}

/// The integer type of an address of `space`: `i64` for generic and global
/// addresses, `i32` for the others (CuTe DSL's `toint`).
fn int_type(space: AddressSpace) -> DType {
    match space {
        AddressSpace::Generic | AddressSpace::Gmem => DType::I64,
        _ => DType::I32,
    }
}

/// `x` (an integer value) as an `i64`, zero-extended from 32 bits.
fn widen(b: &mut builder::BuilderState, x: ValueId, loc: crate::ir::Loc) -> ValueId {
    match b.dtype(x) {
        DType::I64 => x,
        DType::U64 => b.bitcast(x, DType::I64, loc),
        DType::U32 => b.cast(x, DType::I64, loc),
        DType::I32 => {
            let u = b.bitcast(x, DType::U32, loc);
            b.cast(u, DType::I64, loc)
        }
        other => panic!("cute: an address is an integer of 32 or 64 bits, not {other}"),
    }
}

/// CuTe DSL's `cute.make_ptr(dtype, value, mem_space, assumed_align=)`: a
/// pointer to `T` at the integer address `value` (an `Int64`, `Uint64`,
/// `Int32` or `Uint32`, such as [`Ptr::toint`] gives) in `mem_space`.
///
/// `assumed_align` (bytes) is the alignment the address is promised to
/// have; it defaults to the element's size, and must be a power of two
/// that divides the element's size or is a multiple of it. The promise is
/// passed on to the compiler (`cute.assume`), proves the pointer's vector
/// accesses aligned, and is checked by the simulator, which reports an
/// address that breaks it.
///
/// The spaces are [`AddressSpace::Generic`] (loads and stores without a
/// state space, which reach global, shared or local memory),
/// [`AddressSpace::Gmem`], [`AddressSpace::Smem`] (an offset in the block's
/// shared-memory window), [`AddressSpace::Rmem`] (an offset in the
/// thread's local window) and [`AddressSpace::Tmem`] (a tensor-memory
/// address, which tcgen05 operations take: loads and stores through it are
/// refused). Constant and distributed shared memory wait for the parts of
/// the roadmap that use them.
///
/// The pointer works with every load, store, atomic and copy that takes a
/// pointer or a tensor ([`make_tensor`](crate::make_tensor) over it);
/// `cp.async`, `ldmatrix`, bulk copies and mbarriers take pointers into
/// buffers only.
///
/// ```
/// use cute::prelude::*;
///
/// let kernel = Kernel::builder("round_trip").block_dim([32, 1, 1]).build(|| {
///     let x = param_ptr::<f32>("x");
///     let t = thread_idx().x;
///     let addr = (x.clone() + t).toint_i64();
///     let p = make_ptr::<f32>(addr, AddressSpace::Gmem, Some(4));
///     p.store(p.load() * 2.0);
/// });
/// let mut args = vec![KernelArg::f32((0..32).map(|i| i as f32).collect())];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[0].as_f32()[3], 6.0);
/// ```
#[track_caller]
pub fn make_ptr<T: Numeric>(
    value: impl Into<AnyValue>,
    mem_space: AddressSpace,
    assumed_align: Option<usize>,
) -> Ptr<T> {
    let value = value.into();
    let size = elem_align::<T>();
    let align = assumed_align.unwrap_or(size);
    assert!(
        align.is_power_of_two() && (align.is_multiple_of(size) || size.is_multiple_of(align)),
        "make_ptr: an assumed alignment of {align} bytes must be a power of two that divides the {size} bytes of {} \
         or is a multiple of them",
        T::DTYPE
    );
    raw_ptr::<T>(value.0, mem_space, align, "make_ptr")
}

/// CuTe DSL's `cutlass.inttoptr(value, mem_space, dtype)`: [`make_ptr`] with
/// no alignment assumed beyond the element's size.
#[track_caller]
pub fn inttoptr<T: Numeric>(value: impl Into<AnyValue>, mem_space: AddressSpace) -> Ptr<T> {
    make_ptr::<T>(value, mem_space, None)
}

/// A raw pointer to `T` at the address `value` of `space`, assumed `align`
/// bytes aligned.
#[track_caller]
fn raw_ptr<T: Numeric>(value: ValueId, space: AddressSpace, align: usize, what: &str) -> Ptr<T> {
    let loc = Location::caller();
    match space {
        AddressSpace::Generic | AddressSpace::Gmem | AddressSpace::Smem | AddressSpace::Rmem | AddressSpace::Tmem => {}
        other => panic!(
            "{what}: pointers into {other} memory are not supported yet (constant memory and distributed shared \
             memory come with the cluster and TMA parts of the roadmap)"
        ),
    }
    assert!(
        T::DTYPE != DType::Bool,
        "{what}: a Boolean is an i1 in registers and a byte in memory; point at Uint8 flags instead"
    );
    builder::with(|b| {
        let vt = b.dtype(value);
        assert!(
            matches!(vt, DType::I32 | DType::U32 | DType::I64 | DType::U64),
            "{what}: an address is an Int64, Uint64, Int32 or Uint32, not {vt}"
        );
        if space == AddressSpace::Tmem {
            assert!(
                matches!(vt, DType::I32 | DType::U32),
                "{what}: a tensor-memory address is an Int32 (CuTe DSL's make_ptr takes Int32 for tmem), not {vt}"
            );
        }
        let mut addr = widen(b, value, loc);
        if align > 1 {
            addr = b.assume(addr, align as u64, false, loc);
        }
        let n = b.raw_buffer_count();
        let elem = storage_decl(T::DTYPE, 0).0;
        let decl = BufferDecl {
            name: format!("raw{n}"),
            elem,
            len: None,
            space: MemSpace::Raw(space),
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        };
        let buffer = b.declare_raw_buffer(decl, addr);
        Ptr {
            buffer,
            space: MemSpace::Raw(space),
            offset: IndexExpr::Static(0),
            swizzle: None,
            peer: None,
            _t: PhantomData,
        }
    })
}

impl<T: Numeric> Ptr<T> {
    /// Whether the pointer was made from an integer address ([`make_ptr`],
    /// [`inttoptr`], [`tospace`](Ptr::tospace), [`align`](Ptr::align) or a
    /// mask) rather than into a buffer.
    pub fn is_raw(&self) -> bool {
        self.space.raw().is_some()
    }

    /// The pointer's byte offset from its buffer's start (or raw base).
    #[track_caller]
    fn byte_offset(&self, what: &str) -> IndexExpr {
        // The integer address of another CTA's shared memory is not modelled
        // (roadmap part m8-dsmem: a follow-up).
        crate::tensor::refuse_dsmem(self.peer, &format!("{what} of the pointer {self:?}"));
        let bits = T::DTYPE.storage_bits();
        if bits >= 8 {
            return self.offset.clone() * (bits / 8);
        }
        let b = self.offset.clone() * bits;
        assert!(b.is_multiple_of(8), "{what}: the pointer {self:?} is not at a whole byte");
        b / 8
    }

    /// The address (CuTe DSL's `Pointer.toint()`, `cute.ptrtoint`): an
    /// [`Int64`] for a global or generic pointer, an [`Int32`] for the others
    /// (a shared pointer's offset in the block's shared-memory window, a
    /// per-thread array's in the thread's local window, a tensor-memory
    /// address). A swizzled pointer gives the address before its swizzle,
    /// as CuTe DSL's does. [`toint_i64`](Ptr::toint_i64) and
    /// [`toint_i32`](Ptr::toint_i32) give it typed.
    #[track_caller]
    pub fn toint(&self) -> AnyValue {
        let loc = Location::caller();
        let space = self.memspace();
        let ty = int_type(space);
        let byte =
            if space == AddressSpace::Tmem { tmem_word_offset::<T>(&self.offset) } else { self.byte_offset("toint") };
        if let Some(base) = builder::with(|b| b.raw_base(self.buffer)) {
            let off = index_operand(byte);
            return AnyValue(builder::with(|b| {
                let wide = if b.dtype(off) == DType::I64 { off } else { b.cast(off, DType::I64, loc) };
                let sum = b.binary(BinaryOp::Add, base, wide, loc);
                if ty == DType::I64 { sum } else { b.cast(sum, ty, loc) }
            }));
        }
        let index = index_operand(byte);
        AnyValue(builder::with(|b| {
            let result = b.new_value(ty);
            b.push(Op::PtrToInt { result, buffer: self.buffer, index, elem: DType::U8 }, loc);
            result
        }))
    }

    /// [`toint`](Ptr::toint) as an [`Int64`], a 32-bit address zero-extended.
    #[track_caller]
    pub fn toint_i64(&self) -> Int64 {
        let loc = Location::caller();
        let v = self.toint();
        Value::from_id(builder::with(|b| widen(b, v.0, loc)))
    }

    /// [`toint`](Ptr::toint) of a pointer whose address is 32 bits (shared,
    /// per-thread or tensor memory) as an [`Int32`].
    #[track_caller]
    pub fn toint_i32(&self) -> Int32 {
        let space = self.memspace();
        assert!(
            int_type(space) == DType::I32,
            "toint_i32: a {space} pointer's address is 64 bits (toint_i64 gives it)"
        );
        Value::from_id(self.toint().0)
    }

    /// The same address in another address space (CuTe DSL's
    /// `Pointer.tospace`): the pointer itself for its own space; otherwise
    /// one of the two spaces must be generic, and the address is converted
    /// (`cvta` to generic, `cvta.to` from it; LLVM's `addrspacecast`). The
    /// element type and the alignment are kept. A generic address outside
    /// the target space (a shared address cast to global, say) is undefined
    /// in PTX; the simulator reports it at this line.
    #[track_caller]
    pub fn tospace(&self, space: AddressSpace) -> Ptr<T> {
        let loc = Location::caller();
        let from = self.memspace();
        if from == space {
            return self.clone();
        }
        assert!(
            from == AddressSpace::Generic || space == AddressSpace::Generic,
            "tospace: a {from} pointer cannot become a {space} pointer directly: an address only converts between \
             the generic space and a specific one (go through AddressSpace::Generic)"
        );
        for s in [from, space] {
            assert!(
                matches!(s, AddressSpace::Generic | AddressSpace::Gmem | AddressSpace::Smem | AddressSpace::Rmem),
                "tospace: {s} addresses do not convert to or from generic ones"
            );
        }
        self.require_plain("tospace");
        let align = self.alignment();
        let x = self.toint().0;
        let result = builder::with(|b| {
            let result = b.new_value(int_type(space));
            b.push(Op::CvtAddr { result, x, from, to: space }, loc);
            result
        });
        raw_ptr::<T>(result, space, align.max(1), "tospace")
    }

    /// The pointer rounded up to a multiple of `min_align` bytes (CuTe DSL's
    /// `Pointer.align(min_align)`): itself when its alignment is already at
    /// least that, a pointer further into its buffer when its offset is
    /// static and the buffer that aligned, and otherwise the rounded address
    /// `(toint + a - 1) & !(a - 1)` as a pointer made with that alignment.
    /// `min_align` must be a power of two; a tensor-memory pointer is
    /// refused.
    #[track_caller]
    pub fn align(&self, min_align: usize) -> Ptr<T> {
        let loc = Location::caller();
        assert!(
            min_align.is_power_of_two(),
            "Ptr::align: {min_align} bytes cannot be an alignment, being no power of two"
        );
        let space = self.memspace();
        assert!(space != AddressSpace::Tmem, "Ptr::align: a tensor-memory address is not a byte address to align");
        if self.alignment() >= min_align {
            return self.clone();
        }
        self.require_plain("Ptr::align");
        let size = elem_align::<T>();
        let buffer_align = builder::with(|b| b.buffer(self.buffer).align);
        if let Some(off) = self.offset.as_static()
            && !self.is_raw()
            && buffer_align >= min_align
            && min_align.is_multiple_of(size)
            && T::DTYPE.storage_bits() >= 8
        {
            let bytes = (off * size).next_multiple_of(min_align);
            return self.with_offset(IndexExpr::Static(bytes / size));
        }
        let x = self.toint().0;
        let aligned = builder::with(|b| {
            let ty = b.dtype(x);
            let lit = |v: i64| if ty == DType::I64 { Literal::I64(v) } else { Literal::I32(v as i32) };
            let up = b.constant(lit(min_align as i64 - 1), loc);
            let sum = b.binary(BinaryOp::Add, x, up, loc);
            let m = b.constant(lit(!(min_align as i64 - 1)), loc);
            b.binary(BinaryOp::And, sum, m, loc)
        });
        raw_ptr::<T>(aligned, space, min_align, "Ptr::align")
    }

    /// The alignment in bytes the address is known to have (CuTe DSL's
    /// `Pointer.alignment`): the alignment its buffer is declared or its raw
    /// address assumed with ([`make_ptr`]'s `assumed_align`), halved until
    /// the offset from there is provably a multiple of it. A swizzle is not
    /// counted ([`max_alignment`](Ptr::max_alignment) counts it).
    pub fn alignment(&self) -> usize {
        ptr_alignment(self.buffer, &self.offset, T::DTYPE.bytes())
    }

    #[track_caller]
    fn require_plain(&self, what: &str) {
        if let Some(sw) = self.swizzle
            && !sw.is_identity()
        {
            panic!("{what}: the pointer is swizzled by {sw}; drop the swizzle first (recast_ptr(&ptr, None))");
        }
    }

    /// `self & mask` (CuTe DSL's `Pointer.__and__`), as `&`'s operand types
    /// allow: the address with the bits `mask` clears cleared, a pointer of
    /// the same space and type.
    #[track_caller]
    fn masked(&self, mask: MaskValue) -> Ptr<T> {
        let loc = Location::caller();
        let space = self.memspace();
        assert!(space != AddressSpace::Tmem, "Ptr & mask: a tensor-memory address is not a byte address to mask");
        self.require_plain("Ptr & mask");
        let x = self.toint().0;
        let size = elem_align::<T>();
        let (result, align) = builder::with(|b| {
            let ty = b.dtype(x);
            let (m, known) = match mask {
                MaskValue::Const(c) => {
                    let lit = if ty == DType::I64 { Literal::I64(c) } else { Literal::I32(c as i32) };
                    (b.constant(lit, loc), Some(c))
                }
                MaskValue::Value(v) => {
                    let vt = b.dtype(v);
                    assert!(
                        vt.bits() == ty.bits() && vt.is_int(),
                        "Ptr & mask: a {space} address is {ty}, so the mask must be too, not {vt}"
                    );
                    let v = if vt == ty { v } else { b.bitcast(v, ty, loc) };
                    (v, None)
                }
            };
            // The low bits a constant mask clears make the result that aligned.
            let cleared = known.map_or(0, |c| (c as u64 | (1 << 63)).trailing_zeros().min(31));
            (b.binary(BinaryOp::And, x, m, loc), (1usize << cleared).max(size))
        });
        raw_ptr::<T>(result, space, align, "Ptr & mask")
    }
}

/// A tensor-memory pointer's element offset as an offset of its 32-bit
/// address (CuTe C++'s `tmem_ptr<T>`, include/cute/pointer.hpp): offsets
/// count elements of `T`, and `32 / bits(T)` of them share a column, so the
/// address moves by the offset shifted right by `log2(32 / bits(T))`. A lane
/// step is then `1 << 16` elements of a 32-bit `T` and `1 << 17` of a 16-bit
/// one. (C++ rotates rather than shifts, keeping a sub-word index in unused
/// high bits; here it is dropped.)
#[track_caller]
pub(crate) fn tmem_word_offset<T: Numeric>(offset: &IndexExpr) -> IndexExpr {
    let bits = T::DTYPE.storage_bits();
    assert!(
        bits <= 32 && 32 % bits == 0,
        "a tensor-memory pointer's element type packs into 32-bit cells, so it has at most 32 bits, not {bits} ({})",
        T::DTYPE
    );
    let shift = (32 / bits).trailing_zeros() as usize;
    IndexExpr::shift_right(offset.clone(), IndexExpr::Static(shift))
}

/// A mask [`Ptr`]'s `&` takes: an integer constant (`i64`, `u64`, `i32`,
/// `u32`) or a runtime `Int64`, `Uint64`, `Int32` or `Uint32` of the
/// address's width.
pub enum MaskValue {
    Const(i64),
    Value(ValueId),
}

macro_rules! mask_const {
    ($($t:ty),*) => {$(
        impl From<$t> for MaskValue {
            fn from(c: $t) -> Self { MaskValue::Const(c as i64) }
        }
    )*};
}
mask_const!(i64, u64, i32, u32);

macro_rules! mask_value {
    ($($t:ty),*) => {$(
        impl From<Value<$t>> for MaskValue {
            #[track_caller]
            fn from(v: Value<$t>) -> Self { MaskValue::Value(v.checked()) }
        }
    )*};
}
mask_value!(i64, u64, i32, u32);

/// `ptr & mask` (CuTe DSL's `Pointer.__and__`): the pointer at `toint() &
/// mask`, of the same space and type, as rounding an address down
/// (`p & !127` for a 128-byte boundary) does. A constant mask that clears
/// the low `k` bits makes the result `2^k`-byte aligned.
impl<T: Numeric, M: Into<MaskValue>> BitAnd<M> for &Ptr<T> {
    type Output = Ptr<T>;
    #[track_caller]
    fn bitand(self, mask: M) -> Ptr<T> {
        self.masked(mask.into())
    }
}

impl<T: Numeric, M: Into<MaskValue>> BitAnd<M> for Ptr<T> {
    type Output = Ptr<T>;
    #[track_caller]
    fn bitand(self, mask: M) -> Ptr<T> {
        self.masked(mask.into())
    }
}

/// A pointer as an operand of [`inline_ptx`](crate::inline_ptx),
/// [`printf`](crate::printf) and the other functions that take
/// [`AnyValue`]s: its address ([`Ptr::toint`]), an `l` (64-bit) register for
/// a global or generic pointer and an `r` (32-bit) one for the others, in
/// place of CuTe DSL's `llvm_ptr`.
impl<T: Numeric> From<&Ptr<T>> for AnyValue {
    #[track_caller]
    fn from(p: &Ptr<T>) -> AnyValue {
        p.toint()
    }
}

impl<T: Numeric> From<Ptr<T>> for AnyValue {
    #[track_caller]
    fn from(p: Ptr<T>) -> AnyValue {
        p.toint()
    }
}

/// Check that the device address `address` given for the parameter `name`
/// is `align`-byte aligned, as the kernel declares ([`param_ptr_aligned`](crate::param_ptr_aligned),
/// [`param_tensor_aligned`](crate::param_tensor_aligned), 16 bytes for
/// [`param_ptr`](crate::param_ptr) and [`param_tensor`](crate::param_tensor)).
/// The CUDA launcher checks every buffer argument with it, and the
/// simulator its own addresses.
pub fn check_pointer_alignment(name: &str, address: u64, align: usize) -> Result<(), String> {
    if !address.is_multiple_of(align as u64) {
        return Err(format!(
            "`{name}` is at {address:#x}, which is not {align}-byte aligned as the kernel declares (its assumed \
             alignment)"
        ));
    }
    Ok(())
}

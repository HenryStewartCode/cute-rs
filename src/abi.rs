//! Build-time declarations that shape the kernel's ABI and module: by-value parameters
//! ([`param_struct`], [`param_bytes`]), module globals ([`module_global`], CuTe DSL's
//! module-scope `cutlass.Array`) and shared-memory overlays ([`smem_overlay`], cute-rs's
//! form of `smem_merge_branch_allocs`).

use std::panic::Location;

use crate::builder::{self, user_panic};
use crate::ir::{BufferDecl, BufferId, MemAttrs, MemSpace, Op};
use crate::layout::Layout;
use crate::params::NativeLayout;
use crate::tensor::{AddressSpace, Engine, Tensor};
use crate::value::{Numeric, Value};

/// A by-value kernel parameter: a C struct ([`NativeLayout`]) or raw bytes (a 128-byte TMA
/// descriptor), passed in the kernel's parameter space as one `.param` aggregate. The
/// kernel reads its fields; it cannot write them.
#[derive(Clone, Debug)]
pub struct StructParam {
    buffer: BufferId,
    name: String,
    size: usize,
    layout: Option<NativeLayout>,
}

/// Declare a by-value struct parameter with `layout` (CuTe DSL's native struct as one
/// `.param`, at natural or packed alignment). The host passes its bytes:
/// [`NativeLayout::pack`] builds them, [`KernelArg::by_value`](crate::KernelArg::by_value)
/// carries them. Fields of a packed layout that are not at a multiple of their size cannot
/// be read.
#[track_caller]
pub fn param_struct(name: &str, layout: &NativeLayout) -> StructParam {
    declare(name, layout.size, layout.align, false, Some(layout.clone()))
}

/// Declare `size` bytes passed by value, `align`-byte aligned (a power of two), as a grid
/// constant when `grid_constant` (`nvvm.grid_constant`: the kernel may take the
/// parameter's address without a local copy, as TMA descriptors need). Read them with
/// [`StructParam::load`].
#[track_caller]
pub fn param_bytes(name: &str, size: usize, align: usize, grid_constant: bool) -> StructParam {
    declare(name, size, align, grid_constant, None)
}

#[track_caller]
fn declare(name: &str, size: usize, align: usize, grid_constant: bool, layout: Option<NativeLayout>) -> StructParam {
    let loc = Location::caller();
    if size == 0 || !align.is_power_of_two() || !size.is_multiple_of(align) {
        user_panic(
            loc,
            format!(
                "cute: by-value parameter `{name}` of {size} bytes aligned to {align}: the size must be a nonzero \
                 multiple of the alignment, a power of two"
            ),
        );
    }
    let _at = builder::LocationScope::new();
    let buffer = builder::with(|b| b.declare_by_value(name, size, align, grid_constant));
    StructParam { buffer, name: name.to_string(), size, layout }
}

impl StructParam {
    pub(crate) fn buffer_id(&self) -> BufferId {
        self.buffer
    }

    /// The generic address of the parameter's bytes in the kernel's parameter space (a
    /// grid constant's is its own, `cvta.param`; another's is that of the local copy the
    /// compiler makes). TMA instructions take a descriptor's this way.
    #[track_caller]
    pub fn address(&self) -> crate::value::Int64 {
        crate::tma::param_address(self.buffer)
    }

    /// The parameter's bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The struct layout it was declared with.
    pub fn layout(&self) -> Option<&NativeLayout> {
        self.layout.as_ref()
    }

    /// Load a `T` at byte `offset`, which must be a multiple of `T`'s size.
    #[track_caller]
    pub fn load<T: Numeric>(&self, offset: usize) -> Value<T> {
        let loc = Location::caller();
        let size = T::DTYPE.bytes();
        if T::DTYPE.is_sub_byte() || size == 0 {
            user_panic(
                loc,
                format!("cute: `{}`: a by-value parameter holds whole-byte types, not {}", self.name, T::DTYPE),
            );
        }
        if !offset.is_multiple_of(size) || offset + size > self.size {
            user_panic(
                loc,
                format!(
                    "cute: `{}`: a {} at byte {offset} is {} the parameter's {} bytes",
                    self.name,
                    T::DTYPE,
                    if offset + size > self.size { "outside" } else { "not naturally aligned in" },
                    self.size
                ),
            );
        }
        let result = builder::with(|b| {
            let index = b.constant(crate::ir::Literal::I32((offset / size) as i32), loc);
            let result = b.new_value(T::DTYPE);
            b.push(
                Op::Load {
                    results: vec![result],
                    buffer: self.buffer,
                    index,
                    attrs: MemAttrs::default(),
                    pred: None,
                    pass_thru: vec![],
                },
                loc,
            );
            result
        });
        Value::from_id(result)
    }

    /// Load field `name` of the struct layout (a field path such as `pos_x`), which must be
    /// a `T`.
    #[track_caller]
    pub fn field<T: Numeric>(&self, name: &str) -> Value<T> {
        let loc = Location::caller();
        let Some(layout) = &self.layout else {
            user_panic(loc, format!("cute: `{}` was declared as bytes, with no fields: use load", self.name));
        };
        let Some(f) = layout.fields.iter().find(|f| f.name == name) else {
            user_panic(loc, format!("cute: struct parameter `{}` has no field `{name}`", self.name));
        };
        if f.dtype != T::DTYPE {
            user_panic(loc, format!("cute: field `{name}` of `{}` is a {}, read as {}", self.name, f.dtype, T::DTYPE));
        }
        self.load::<T>(f.offset)
    }
}

impl NativeLayout {
    /// The bytes of a struct with this layout holding `values` (scalar
    /// [`KernelArg`](crate::KernelArg)s in field order), zero in the padding: what the host
    /// passes for a [`param_struct`].
    pub fn pack(&self, values: &[crate::KernelArg]) -> Result<Vec<u8>, String> {
        if values.len() != self.fields.len() {
            return Err(format!("the struct has {} fields, {} values were given", self.fields.len(), values.len()));
        }
        let mut out = vec![0u8; self.size];
        for (f, v) in self.fields.iter().zip(values) {
            match v.scalar_bits() {
                Some((t, bits)) if t == f.dtype => {
                    let n = t.bytes();
                    out[f.offset..f.offset + n].copy_from_slice(&bits.to_le_bytes()[..n]);
                }
                _ => return Err(format!("field `{}` is a {}, got {}", f.name, f.dtype, v.describe())),
            }
        }
        Ok(out)
    }
}

/// Where a module global lives.
fn global_space(space: AddressSpace) -> Option<MemSpace> {
    match space {
        AddressSpace::Gmem => Some(MemSpace::Global),
        AddressSpace::Cmem => Some(MemSpace::Constant),
        _ => None,
    }
}

/// The most constant memory a module may declare (`.const` banks are 64 KB).
pub const MAX_CONSTANT_BYTES: usize = 64 * 1024;

/// Declare a module global of `len` elements of `T` (CuTe DSL's module-scope
/// `cutlass.Array(dtype, shape, space=, name=, init=, readonly=)`): a `__device__` variable
/// in global memory (`AddressSpace::Gmem`) or a `__constant__` one (`AddressSpace::Cmem`),
/// which the kernel reads (and, in global memory, writes) as a tensor.
///
/// - `name` is its C identifier, which the host looks it up by
///   ([`CudaFunction::global`](crate::cuda), [`SimModule::write_global`](crate::sim::SimModule));
///   one is generated if `None`.
/// - `init` gives its initial values (zeros otherwise).
/// - `readonly` makes it an LLVM constant: the kernel may fold its loads, so it needs
///   `init`, and host writes are refused.
///
/// A module's constant globals hold at most 64 KB together, and the device cannot store to
/// them. Globals keep their values between launches of one loaded module.
#[track_caller]
pub fn module_global<T: Numeric>(
    name: Option<&str>,
    len: usize,
    space: AddressSpace,
    init: Option<&[T]>,
    readonly: bool,
) -> Tensor<T> {
    let loc = Location::caller();
    let Some(mem) = global_space(space) else {
        user_panic(loc, format!("cute: a module global lives in gmem or cmem, not {space:?}"));
    };
    if T::DTYPE.is_sub_byte() || T::DTYPE.bytes() == 0 || len == 0 {
        user_panic(loc, format!("cute: a module global holds one or more whole-byte values, not {len} {}", T::DTYPE));
    }
    if readonly && init.is_none() {
        user_panic(loc, "cute: a readonly module global needs init: its value is folded into the kernel".to_string());
    }
    if let Some(v) = init
        && v.len() != len
    {
        user_panic(loc, format!("cute: module global init has {} values for {len} elements", v.len()));
    }
    let size = T::DTYPE.bytes();
    let bytes = init.map(|v| v.iter().flat_map(|x| x.raw_bits().to_le_bytes().into_iter().take(size)).collect());
    let _at = builder::LocationScope::new();
    let buffer = builder::with(|b| {
        let name = name.map_or_else(|| format!("__cute_global_{}", b.global_count()), str::to_string);
        if !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
            user_panic(loc, format!("cute: module global name {name:?} is not a C identifier"));
        }
        let decl = BufferDecl {
            name,
            elem: T::DTYPE,
            len: Some(len),
            space: mem,
            align: size.max(4),
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        };
        b.declare_module_global(decl, bytes, readonly)
    });
    Tensor::new(Engine::Memory { buffer, space: mem, peer: None }, Layout::col_major((len,)))
}

/// CuTe DSL's `extern=True` module globals, declared here and defined in another module,
/// need linking against other device code, which cute-rs does not support for globals. This
/// always fails, naming that.
#[track_caller]
pub fn extern_global<T: Numeric>(name: &str, _len: usize, _space: AddressSpace) -> Tensor<T> {
    user_panic(
        Location::caller(),
        format!(
            "cute: extern module global `{name}`: extern globals are defined in another module and need device \
             code linking, which cute-rs does not support for globals; declare it with \
             module_global instead"
        ),
    )
}

/// Place the shared buffers of two alternatives at one offset, so that they share bytes
/// (cute-rs's form of CuTe DSL's `smem_merge_branch_allocs`): each closure allocates one
/// shared buffer (a tensor or a [`cute_struct!`](crate::cute_struct)); both start at the
/// group's offset, which is aligned for both, and the group spans the larger. Use them in
/// phases separated by a barrier: the simulator's race checker sees overlaid accesses as
/// accesses to the same bytes. A kernel with overlays keeps its shared memory in the
/// dynamic window.
#[track_caller]
pub fn smem_overlay<A, B>(first: impl FnOnce() -> A, second: impl FnOnce() -> B) -> (A, B) {
    let loc = Location::caller();
    let mut members = Vec::new();
    let mut alternative = |f: &mut dyn FnMut()| {
        let before = builder::with(|b| b.buffer_ids().count());
        f();
        let added: Vec<BufferId> = builder::with(|b| {
            b.buffer_ids().skip(before).filter(|&id| b.buffer(id).space == MemSpace::Shared).collect()
        });
        if added.len() != 1 || builder::with(|b| b.buffer(added[0]).len.is_none()) {
            user_panic(
                loc,
                format!(
                    "cute: each alternative of smem_overlay allocates exactly one shared buffer sized at build \
                     time; this one allocated {} (group several in a cute_struct!)",
                    added.len()
                ),
            );
        }
        members.push(vec![added[0]]);
    };
    let mut a = None;
    let mut first = Some(first);
    alternative(&mut || a = Some((first.take().expect("called once"))()));
    let mut b = None;
    let mut second = Some(second);
    alternative(&mut || b = Some((second.take().expect("called once"))()));
    builder::with(|bld| bld.overlays.push(members));
    (a.expect("ran"), b.expect("ran"))
}

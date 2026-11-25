//! Struct parameters ([`kernel_params!`](crate::kernel_params): CuTe DSL's
//! dataclasses and NamedTuples as kernel arguments) and native structs
//! (CuTe DSL's `cute.native_struct`).

use crate::ir::DType;

/// One field of a native struct's C layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeField {
    /// The field's path (`pos_x` for field `x` of field `pos`).
    pub name: String,
    pub dtype: DType,
    /// Bytes from the start of the struct.
    pub offset: usize,
}

/// What a field adds to a native struct's layout: a scalar, or a nested
/// native struct, placed whole with its own layout (its padding kept).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeMember {
    Scalar(String, DType),
    Struct(NativeLayout),
}

/// The C layout of a native struct (CuTe DSL's `native_struct(packed=)`):
/// each scalar field at its natural alignment (its size), or one byte
/// after another when packed; constant fields take no space. A nested
/// native struct is one member at its own alignment (1 inside a packed
/// struct), with its own padding, and its fields are listed with their
/// offsets from the outer start. Recorded for the by-value `.param`
/// structs that kernel-argument marshaling adds; a native struct is passed
/// today as one scalar parameter per field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeLayout {
    pub fields: Vec<NativeField>,
    pub size: usize,
    pub align: usize,
    pub packed: bool,
}

impl NativeLayout {
    /// The layout of the scalar `fields` in order.
    pub fn new(fields: Vec<(String, DType)>, packed: bool) -> NativeLayout {
        NativeLayout::from_members(fields.into_iter().map(|(n, t)| NativeMember::Scalar(n, t)).collect(), packed)
    }

    /// The layout of `members` in order.
    pub fn from_members(members: Vec<NativeMember>, packed: bool) -> NativeLayout {
        let (mut offset, mut align) = (0usize, 1usize);
        let mut out = Vec::with_capacity(members.len());
        for m in members {
            match m {
                NativeMember::Scalar(name, dtype) => {
                    let size = dtype.bytes().max(1);
                    let a = if packed { 1 } else { size };
                    offset = offset.next_multiple_of(a);
                    align = align.max(a);
                    out.push(NativeField { name, dtype, offset });
                    offset += size;
                }
                NativeMember::Struct(inner) => {
                    let a = if packed { 1 } else { inner.align };
                    offset = offset.next_multiple_of(a);
                    align = align.max(a);
                    out.extend(inner.fields.into_iter().map(|f| NativeField { offset: offset + f.offset, ..f }));
                    offset += inner.size;
                }
            }
        }
        NativeLayout { fields: out, size: offset.next_multiple_of(align), align, packed }
    }
}

/// Declare a struct of kernel parameters (CuTe DSL's dataclass or
/// NamedTuple argument), with its spec and host structs:
///
/// ```
/// use cute::prelude::*;
///
/// kernel_params! {
///     /// What the copy kernel takes.
///     pub struct CopyParams(CopyParamsSpec, CopyParamsHost) {
///         pub src: Tensor<f32>,
///         pub dst: Tensor<f32>,
///         pub n: Int32,
///         /// A build-time constant.
///         pub threads: usize,
///     }
/// }
/// # let _ = |s: CopyParamsSpec, h: CopyParamsHost| (s.threads, h.n);
/// ```
///
/// Every field's type is a [`KernelParam`](crate::KernelParam): runtime
/// fields (tensors, pointers, values, layouts, nested parameter structs)
/// become parameters named after the field (`src`, `dst`, `n` at the top of
/// a [`Kernel::jit`](crate::Kernel::jit) signature, `p_src` under a name
/// `p`), and plain Rust fields (`usize`, `bool`, ... and
/// [`Constexpr`](crate::Constexpr)) stay build-time constants that a launch
/// checks. `CopyParamsSpec` has each field's spec and `CopyParamsHost` each
/// field's host value (both `Clone` and `Debug`); the struct itself gets
/// `Clone` and [`Carry`](crate::Carry), so it flows through
/// [`for_range_carry`](crate::for_range_carry), branches and [`Var`](crate::Var):
/// its runtime values are carried, while tensors in memory, layouts and
/// constants keep the values they came in with, and a body that changes a
/// constant is refused. Don't derive `Clone` yourself.
///
/// A first attribute `#[native_struct(zero_init = .., packed = ..)]` makes
/// it a native struct (CuTe DSL's `@cute.native_struct`), whose fields are
/// scalars, constants and other native structs: `S::make(&spec)` builds one
/// with every field zero (or, without `zero_init`, of unspecified contents:
/// cute-rs has no undefined value, so they are zero too), and
/// `S::native_layout(&spec)` gives its C layout ([`NativeLayout`]), natural
/// or packed. As a kernel argument it is flattened into one scalar
/// parameter per field.
#[macro_export]
macro_rules! kernel_params {
    (
        #[native_struct(zero_init = $zero:literal, packed = $packed:literal)]
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ($spec:ident, $host:ident) {
            $($(#[$field_attr:meta])* $field_vis:vis $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        $crate::kernel_params! {
            @packed $packed;
            $(#[$attr])*
            $vis struct $name ($spec, $host) {
                $($(#[$field_attr])* $field_vis $field : $ty),*
            }
        }

        impl $name {
            /// Whether [`make`](Self::make) zeroes the fields (`zero_init`).
            pub const ZERO_INIT: bool = $zero;
            /// Whether the C layout is packed.
            pub const PACKED: bool = $packed;

            /// A struct whose fields are zero (or, without `zero_init`,
            /// unspecified), constants from `spec`.
            #[track_caller]
            pub fn make(spec: &$spec) -> Self {
                <Self as $crate::KernelParam>::native_init(spec, $zero)
            }

            /// The struct's C layout.
            pub fn native_layout(spec: &$spec) -> $crate::NativeLayout {
                let mut members = ::std::vec::Vec::new();
                <Self as $crate::KernelParam>::native_fields("", spec, &mut members);
                match members.pop() {
                    ::std::option::Option::Some($crate::params::NativeMember::Struct(layout)) => layout,
                    _ => ::std::unreachable!(),
                }
            }
        }
    };
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ($spec:ident, $host:ident) {
            $($(#[$field_attr:meta])* $field_vis:vis $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        $crate::kernel_params! {
            @packed false;
            $(#[$attr])*
            $vis struct $name ($spec, $host) {
                $($(#[$field_attr])* $field_vis $field : $ty),*
            }
        }
    };
    (
        @packed $packed:literal;
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ($spec:ident, $host:ident) {
            $($(#[$field_attr:meta])* $field_vis:vis $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis struct $name {
            $($(#[$field_attr])* $field_vis $field: $ty),*
        }

        /// The build-time description of each field.
        #[derive(Clone, Debug)]
        $vis struct $spec {
            $(pub $field: <$ty as $crate::KernelParam>::Spec),*
        }

        /// The host value of each field.
        #[derive(Clone, Debug)]
        $vis struct $host {
            $(pub $field: <$ty as $crate::KernelParam>::Host),*
        }

        impl ::std::clone::Clone for $name {
            fn clone(&self) -> Self {
                $name { $($field: ::std::clone::Clone::clone(&self.$field)),* }
            }
        }

        impl $crate::KernelParam for $name {
            type Spec = $spec;
            type Host = $host;

            #[track_caller]
            fn declare(name: &str, spec: &$spec) -> Self {
                $(let $field = <$ty as $crate::KernelParam>::declare(
                    &$crate::params::field_name(name, ::std::stringify!($field)), &spec.$field);)*
                $name { $($field),* }
            }

            fn spec_of(host: &$host) -> $spec {
                $spec { $($field: <$ty as $crate::KernelParam>::spec_of(&host.$field)),* }
            }

            fn pack(
                name: &str,
                spec: &$spec,
                host: &$host,
                out: &mut $crate::ArgPack,
            ) -> ::std::result::Result<(), $crate::ArgError> {
                $(<$ty as $crate::KernelParam>::pack(
                    &$crate::params::field_name(name, ::std::stringify!($field)), &spec.$field, &host.$field, out)?;)*
                ::std::result::Result::Ok(())
            }

            fn write_back(
                spec: &$spec,
                host: &mut $host,
                buffers: &mut dyn ::std::iter::Iterator<Item = $crate::KernelArg>,
            ) {
                $(<$ty as $crate::KernelParam>::write_back(&spec.$field, &mut host.$field, buffers);)*
            }

            #[track_caller]
            fn carry_flatten(&self, out: &mut ::std::vec::Vec<$crate::ir::ValueId>) {
                $($crate::KernelParam::carry_flatten(&self.$field, out);)*
            }

            fn carry_rebuild(&self, values: &mut dyn ::std::iter::Iterator<Item = $crate::ir::ValueId>) -> Self {
                $name { $($field: $crate::KernelParam::carry_rebuild(&self.$field, values)),* }
            }

            fn carry_mismatch(&self, other: &Self) -> ::std::option::Option<::std::string::String> {
                let mismatch = ::std::option::Option::None;
                $(let mismatch = mismatch.or_else(|| {
                    $crate::KernelParam::carry_mismatch(&self.$field, &other.$field)
                        .map(|m| ::std::format!("field {}: {m}", ::std::stringify!($field)))
                });)*
                mismatch
            }

            #[track_caller]
            fn native_init(spec: &$spec, zero: bool) -> Self {
                $name { $($field: <$ty as $crate::KernelParam>::native_init(&spec.$field, zero)),* }
            }

            fn native_fields(
                name: &str,
                spec: &$spec,
                out: &mut ::std::vec::Vec<$crate::params::NativeMember>,
            ) {
                let mut members = ::std::vec::Vec::new();
                $(<$ty as $crate::KernelParam>::native_fields(
                    &$crate::params::field_name(name, ::std::stringify!($field)), &spec.$field, &mut members);)*
                out.push($crate::params::NativeMember::Struct($crate::NativeLayout::from_members(members, $packed)));
            }
        }

        impl $crate::Carry for $name {
            #[track_caller]
            fn flatten(&self, out: &mut ::std::vec::Vec<$crate::ir::ValueId>) {
                $crate::KernelParam::carry_flatten(self, out)
            }
            fn rebuild(&self, values: &mut dyn ::std::iter::Iterator<Item = $crate::ir::ValueId>) -> Self {
                $crate::KernelParam::carry_rebuild(self, values)
            }
            fn structure_mismatch(&self, other: &Self) -> ::std::option::Option<::std::string::String> {
                $crate::KernelParam::carry_mismatch(self, other)
            }
        }
    };
}

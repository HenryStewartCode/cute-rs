//! A kernel's launch signature: the one list of parameters and launch requirements that
//! the simulator, the CUDA launcher and the launch checks all read, so they cannot
//! disagree (the launcher half of CuTe DSL's argument marshaling).
//!
//! [`KernelSignature`] comes from the kernel's IR ([`Kernel::signature`]) or from a compiled
//! kernel ([`CompiledKernel::signature`](crate::CompiledKernel::signature), which adds the
//! target and PTX version). Each parameter is a pointer (a buffer), a scalar, or a by-value
//! aggregate; [`KernelSignature::pack`] turns launch values into the bytes of each
//! parameter, checking every leaf (dtype, divisibility, 64-bit leaves, pointer alignment,
//! by-value size) and naming the parameter in any error.

use std::fmt;

use crate::builder::Kernel;
use crate::ir::{DType, KernelIr, MemSpace, Param, SharedMemoryLayout};
use crate::sim::{KernelArg, LaunchConfig};

/// What one kernel parameter is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// A device pointer to elements of `elem`, at least `min_len` of them when known,
    /// `align`-byte aligned.
    Pointer { elem: DType, min_len: Option<usize>, align: usize },
    /// A scalar of `dtype` that must be a multiple of `multiple_of`.
    Scalar { dtype: DType, multiple_of: usize },
    /// `size` bytes passed by value (a C struct or a TMA descriptor), `align`-byte aligned in
    /// the parameter space, `grid_constant` when the kernel may take its address.
    ByValue { size: usize, align: usize, grid_constant: bool },
}

/// One kernel parameter: its name (a leaf name, such as `a_shape_0`) and kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamSignature {
    pub name: String,
    pub kind: ParamKind,
}

/// A module global the kernel declares ([`module_global`](crate::module_global)).
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalSignature {
    pub name: String,
    pub elem: DType,
    pub len: usize,
    /// In `__constant__` memory.
    pub constant: bool,
    pub readonly: bool,
    pub init: Option<Vec<u8>>,
}

/// A parameter that holds a tensor map (a 128-byte grid constant): one the host binds
/// ([`param_tensormap`](crate::param_tensormap)) or a hidden one the launch computes
/// ([`tma_descriptor`](crate::tma_descriptor)).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorMapSignature {
    /// Its index in [`KernelSignature::params`].
    pub param: usize,
    /// For a hidden one, what it describes.
    pub source: Option<TensorMapBinding>,
}

/// What a hidden tensor-map parameter describes: a tensor parameter, and the descriptor
/// with its runtime leaves tied to scalar parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorMapBinding {
    /// The tensor parameter's index.
    pub tensor_param: usize,
    pub spec: crate::tma::TmaDescSpec,
    /// Each runtime leaf of the spec and the index of the scalar parameter it is.
    pub scalars: Vec<(crate::ir::ValueId, usize)>,
}

/// Everything a launch must agree with.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSignature {
    pub name: String,
    /// Every parameter of the compiled kernel; hidden ones (computed tensor maps) last.
    pub params: Vec<ParamSignature>,
    /// The parameters that hold tensor maps.
    pub tensor_maps: Vec<TensorMapSignature>,
    pub globals: Vec<GlobalSignature>,
    /// Where the shared buffers go ([`KernelIr::shared_memory_layout`]).
    pub smem: SharedMemoryLayout,
    /// Whether the kernel has a dynamic shared-memory region ([`get_dyn_smem`](crate::get_dyn_smem)).
    pub dynamic_smem_region: bool,
    pub block_dim: Option<[u32; 3]>,
    pub max_threads: Option<[u32; 3]>,
    pub min_blocks_per_sm: Option<u32>,
    /// The target the kernel was compiled (or built) for.
    pub target: Option<crate::Arch>,
    /// The PTX ISA version of the compiled kernel.
    pub ptx_version: Option<u32>,
}

/// The shared memory of one launch, resolved from [`LaunchConfig`] and the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmemPlan {
    /// Bytes of the dynamic region ([`get_dyn_smem`](crate::get_dyn_smem)).
    pub region_bytes: usize,
    /// Dynamic shared memory passed to the driver (`sharedMemBytes`).
    pub launch_bytes: usize,
    /// Shared memory per block, static and dynamic.
    pub block_bytes: usize,
}

/// One launch value for [`KernelSignature::pack`].
#[derive(Clone, Copy, Debug)]
pub enum LaunchValue<'a> {
    /// A device address: a buffer's, a view's (a buffer plus an offset) or a raw one, with
    /// the bytes behind it when known.
    Pointer { address: u64, bytes: Option<usize> },
    /// A scalar ([`KernelArg::ScalarF32`], [`KernelArg::Scalar`] and the like).
    Scalar(&'a KernelArg),
    /// The bytes of a by-value parameter.
    Bytes(&'a [u8]),
}

/// 64 bytes, 64-byte aligned: a slot of [`PackedArgs`] (a tensor map's alignment).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
struct Slot([u8; 64]);

/// The packed parameter values of one launch, each in its own 64-byte-aligned slot.
#[derive(Clone, Debug, Default)]
pub struct PackedArgs {
    slots: Vec<Vec<Slot>>,
    sizes: Vec<usize>,
}

impl PackedArgs {
    fn push(&mut self, bytes: &[u8]) {
        let mut slot = vec![Slot([0; 64]); bytes.len().div_ceil(64).max(1)];
        for (i, b) in bytes.iter().enumerate() {
            slot[i / 64].0[i % 64] = *b;
        }
        self.slots.push(slot);
        self.sizes.push(bytes.len());
    }

    /// The number of parameters.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Parameter `i`'s bytes, as the kernel receives them.
    pub fn bytes(&self, i: usize) -> Vec<u8> {
        self.slots[i].iter().flat_map(|w| w.0).take(self.sizes[i]).collect()
    }

    /// A pointer to each slot, as `cuLaunchKernel`'s `kernelParams` takes them.
    #[cfg(feature = "cuda")]
    pub(crate) fn pointers(&mut self) -> Vec<*mut std::ffi::c_void> {
        self.slots.iter_mut().map(|s| s.as_mut_ptr().cast()).collect()
    }
}

/// The scalar bits of `arg` for a parameter of `dtype`, or why it does not fit.
fn scalar_for(name: &str, i: usize, dtype: DType, multiple_of: usize, arg: &KernelArg) -> Result<u128, String> {
    // The integer the bits stand for: sign-extended for a signed type, so that -1 is not
    // read as 2^32 - 1 (a multiple of 3).
    let value = |bits: u128| {
        let shift = 128 - (dtype.bits() as u32).min(128);
        if dtype.is_int() && dtype.is_signed() { ((bits as i128) << shift) >> shift } else { bits as i128 }
    };
    match arg.scalar_bits() {
        Some((t, bits)) if t == dtype && value(bits).unsigned_abs().is_multiple_of(multiple_of as u128) => Ok(bits),
        Some((t, bits)) if t == dtype => {
            Err(format!("argument {i} (`{name}`) must be a multiple of {multiple_of}, got {}", value(bits)))
        }
        Some((t, _)) if t.bits() != dtype.bits() && (t.bits() == 64 || dtype.bits() == 64) => Err(format!(
            "argument {i} (`{name}`) is a {}-bit leaf ({dtype}), got a {} scalar: pass a {dtype}",
            dtype.bits(),
            t
        )),
        _ => Err(format!("argument {i} (`{name}`) must be a {dtype} scalar, got {}", arg.describe())),
    }
}

impl KernelSignature {
    /// The signature of `ir`, with no compiled target.
    pub fn of(ir: &KernelIr) -> KernelSignature {
        let params = ir
            .args
            .iter()
            .map(|p| match p {
                Param::Buffer(b) => {
                    let d = ir.buffer(*b);
                    ParamSignature {
                        name: d.name.clone(),
                        kind: ParamKind::Pointer { elem: d.elem, min_len: d.len, align: d.align },
                    }
                }
                Param::Scalar { name, value, multiple_of } => ParamSignature {
                    name: name.clone(),
                    kind: ParamKind::Scalar { dtype: ir.dtype(*value), multiple_of: *multiple_of },
                },
                Param::ByValue { buffer, grid_constant } => {
                    let d = ir.buffer(*buffer);
                    ParamSignature {
                        name: d.name.clone(),
                        kind: ParamKind::ByValue {
                            size: d.len.unwrap_or(0),
                            align: d.align,
                            grid_constant: *grid_constant,
                        },
                    }
                }
            })
            .collect();
        let globals = ir
            .globals
            .iter()
            .map(|g| {
                let d = ir.buffer(g.buffer);
                GlobalSignature {
                    name: d.name.clone(),
                    elem: d.elem,
                    len: d.len.unwrap_or(0),
                    constant: d.space == MemSpace::Constant,
                    readonly: g.readonly,
                    init: g.init.clone(),
                }
            })
            .collect();
        let scalar_index = |v: crate::ir::ValueId| {
            ir.args.iter().position(|p| matches!(p, Param::Scalar { value, .. } if *value == v))
        };
        let tensor_maps = ir
            .tensor_maps
            .iter()
            .map(|t| TensorMapSignature {
                param: t.param,
                source: t.source.as_ref().map(|s| TensorMapBinding {
                    tensor_param: ir
                        .args
                        .iter()
                        .position(|p| matches!(p, Param::Buffer(b) if *b == s.tensor))
                        .expect("a tensor map's tensor is a parameter"),
                    spec: s.spec.clone(),
                    scalars: s
                        .spec
                        .dependencies()
                        .into_iter()
                        .map(|v| (v, scalar_index(v).expect("a tensor map's runtime leaves are scalar parameters")))
                        .collect(),
                }),
            })
            .collect();
        KernelSignature {
            name: ir.name.clone(),
            params,
            tensor_maps,
            globals,
            smem: ir.shared_memory_layout(),
            dynamic_smem_region: ir.dynamic_smem_region().is_some(),
            block_dim: ir.block_dim,
            max_threads: ir.max_threads,
            min_blocks_per_sm: ir.min_blocks_per_sm,
            target: None,
            ptx_version: None,
        }
    }

    /// The parameter named `name`.
    pub fn param(&self, name: &str) -> Option<&ParamSignature> {
        self.params.iter().find(|p| p.name == name)
    }

    /// The module global named `name`.
    pub fn global(&self, name: &str) -> Option<&GlobalSignature> {
        self.globals.iter().find(|g| g.name == name)
    }

    /// Resolve the launch's shared memory: [`LaunchConfig::smem_bytes`] (CuTe DSL's
    /// `smem=`), which must cover the kernel's own allocations, or the build-time bytes plus
    /// [`LaunchConfig::dynamic_smem_bytes`] for the dynamic region.
    pub fn smem_plan(&self, cfg: &LaunchConfig) -> Result<SmemPlan, String> {
        let base = if self.smem.in_window { self.smem.dynamic_start } else { self.smem.static_bytes };
        let region = match cfg.smem_bytes {
            None => cfg.dynamic_smem_bytes as usize,
            Some(_) if cfg.dynamic_smem_bytes != 0 => {
                return Err(format!(
                    "kernel `{}`: the launch sets both smem_bytes ({}) and dynamic_smem_bytes ({}): set one",
                    self.name,
                    cfg.smem_bytes.unwrap_or(0),
                    cfg.dynamic_smem_bytes
                ));
            }
            Some(n) if (n as usize) < base => {
                return Err(format!(
                    "kernel `{}` allocates {base} bytes of shared memory per block, more than the launch's smem \
                     of {n} bytes: the allocations would overlap whatever follows them",
                    self.name
                ));
            }
            Some(n) => n as usize - base,
        };
        Ok(SmemPlan {
            region_bytes: region,
            launch_bytes: self.smem.launch_bytes(region),
            block_bytes: self.smem.block_bytes(region),
        })
    }

    /// Check a launch configuration against the kernel: block and grid sizes, the declared
    /// block and the largest block, clusters (and a fallback cluster), and shared memory up
    /// to `max_smem` bytes per block, which `device` gives (any GPU when `None`).
    pub fn check_config(&self, cfg: &LaunchConfig, max_smem: usize, device: Option<&str>) -> Result<SmemPlan, String> {
        if cfg.threads_per_block() == 0 || cfg.threads_per_block() > 1024 {
            return Err(format!("invalid block size {:?}", cfg.block));
        }
        if cfg.grid.contains(&0) {
            return Err(format!("invalid grid size {:?}", cfg.grid));
        }
        if cfg.cluster.contains(&0) || cfg.grid.iter().zip(cfg.cluster).any(|(&g, c)| g % c != 0) {
            return Err(format!("grid {:?} is not a whole number of clusters of {:?}", cfg.grid, cfg.cluster));
        }
        if let Some(f) = cfg.fallback_cluster {
            if cfg.cluster == [1, 1, 1] {
                return Err(format!(
                    "a fallback cluster ({f:?}) needs a preferred cluster: set LaunchConfig::with_cluster as well"
                ));
            }
            if f.contains(&0) || cfg.cluster.iter().zip(f).any(|(&p, f)| p % f != 0) {
                return Err(format!(
                    "the preferred cluster {:?} is not a whole number of fallback clusters of {f:?} in every mode",
                    cfg.cluster
                ));
            }
        }
        if let Some(dims) = self.block_dim
            && dims != cfg.block
        {
            return Err(format!(
                "kernel `{}` was built for blocks of {dims:?}, launched with {:?}",
                self.name, cfg.block
            ));
        }
        if let Some(max) = self.max_threads
            && (cfg.block.iter().zip(max).any(|(&b, m)| b > m)
                || cfg.threads_per_block() > max.iter().map(|&d| d as usize).product())
        {
            return Err(format!(
                "kernel `{}` takes blocks of at most {max:?} (max_threads), launched with {:?}",
                self.name, cfg.block
            ));
        }
        let plan = self.smem_plan(cfg)?;
        if plan.block_bytes > max_smem {
            let limit = match device {
                Some(d) => format!("{d} allows a block {max_smem} (its opt-in maximum)"),
                None => format!("no GPU gives a block more than {max_smem}"),
            };
            return Err(format!(
                "kernel `{}` needs {} bytes of shared memory per block ({} sized when it was built and {} of \
                 dynamic shared memory at launch); {limit}",
                self.name,
                plan.block_bytes,
                plan.block_bytes - plan.region_bytes,
                plan.region_bytes
            ));
        }
        Ok(plan)
    }

    /// Check host arguments (the simulator's and [`CudaContext::launch`]'s) against the
    /// parameters: one per parameter, buffers of the right element type and length,
    /// scalars of the right type and divisibility, by-value bytes of the right size.
    ///
    /// [`CudaContext::launch`]: https://docs.rs/cute-rs (the `cuda` feature)
    pub fn check_args(&self, args: &[KernelArg]) -> Result<(), String> {
        if args.len() != self.user_param_count() {
            return Err(format!(
                "kernel `{}` takes {} arguments, {} were given",
                self.name,
                self.user_param_count(),
                args.len()
            ));
        }
        for (i, (p, a)) in self.params.iter().zip(args).enumerate() {
            let name = &p.name;
            if self.tensor_maps.iter().any(|t| t.param == i) {
                let bytes = a.byte_data();
                if !matches!(a, KernelArg::Bytes { .. }) || !crate::tma::TmaDesc::is_sim_encoding(bytes) {
                    return Err(format!(
                        "argument {i} (`{name}`) is a tensor map: pass KernelArg::tensor_map of a host-built \
                         TensorMap, got {}",
                        a.describe()
                    ));
                }
                crate::tma::TmaDesc::from_sim_bytes(bytes)
                    .and_then(|d| d.validate())
                    .map_err(|e| format!("argument {i} (`{name}`): {e}"))?;
                continue;
            }
            match &p.kind {
                ParamKind::Pointer { elem, min_len, .. } => match a.buffer_type() {
                    Some((t, len)) if t == *elem && min_len.is_none_or(|n| len >= n) => {}
                    Some((t, len)) if t == *elem => {
                        return Err(format!(
                            "argument {i} (`{name}`) needs at least {} elements, got {len}",
                            min_len.unwrap_or(0)
                        ));
                    }
                    _ => return Err(format!("argument {i} (`{name}`) must be a {elem} buffer, got {}", a.describe())),
                },
                ParamKind::Scalar { dtype, multiple_of } => {
                    scalar_for(name, i, *dtype, *multiple_of, a)?;
                }
                ParamKind::ByValue { size, .. } => match a {
                    KernelArg::Bytes { data, .. } if data.len() == *size => {}
                    KernelArg::Bytes { data, .. } => {
                        return Err(format!(
                            "argument {i} (`{name}`) is passed by value as {size} bytes, got {}",
                            data.len()
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "argument {i} (`{name}`) is passed by value ({size} bytes, KernelArg::by_value), got {}",
                            a.describe()
                        ));
                    }
                },
            }
        }
        Ok(())
    }

    /// The number of hidden parameters (computed tensor maps, the last ones).
    pub fn hidden_param_count(&self) -> usize {
        self.tensor_maps.iter().filter(|t| t.source.is_some()).count()
    }

    /// The number of arguments a launch passes: every parameter but the hidden ones.
    pub fn user_param_count(&self) -> usize {
        self.params.len() - self.hidden_param_count()
    }

    /// The descriptor of hidden tensor map `t` (an entry of [`tensor_maps`](Self::tensor_maps)
    /// with a source) for a launch whose tensor parameter is at `tensor_address` and whose
    /// scalar arguments `scalar(i)` gives (by parameter index): the one evaluation the
    /// simulator and the CUDA launcher share. Errors name the parameter.
    pub fn hidden_tensor_map(
        &self,
        t: &TensorMapSignature,
        tensor_address: u64,
        scalar: &dyn Fn(usize) -> Option<i64>,
    ) -> Result<crate::tma::TmaDesc, String> {
        let p = &self.params[t.param];
        let Some(src) = &t.source else {
            return Err(format!("parameter {} (`{}`) is bound by the host, not computed", t.param, p.name));
        };
        let env = |v: crate::ir::ValueId| src.scalars.iter().find(|(x, _)| *x == v).and_then(|&(_, i)| scalar(i));
        src.spec.evaluate(tensor_address, &env).map_err(|e| {
            format!(
                "the tensor map of `{}` (hidden parameter {}, `{}`) for this launch: {e}",
                self.params[src.tensor_param].name, t.param, p.name
            )
        })
    }

    /// Check every launch value against its parameter and pack it: the one packing function
    /// of the CUDA launcher. Pointers are checked for the declared alignment and, when their
    /// size is known, for the declared length; scalars for dtype, 64-bit width and
    /// divisibility; by-value bytes for their size.
    pub fn pack(&self, values: &[LaunchValue<'_>]) -> Result<PackedArgs, String> {
        if values.len() != self.params.len() {
            return Err(format!(
                "kernel `{}` takes {} arguments, {} were given",
                self.name,
                self.params.len(),
                values.len()
            ));
        }
        let mut out = PackedArgs::default();
        for (i, (p, v)) in self.params.iter().zip(values).enumerate() {
            let name = &p.name;
            match (&p.kind, v) {
                (ParamKind::Pointer { elem, min_len, align }, LaunchValue::Pointer { address, bytes }) => {
                    if let Err(e) = crate::check_pointer_alignment(name, *address, *align) {
                        return Err(format!("argument {i}: {e}"));
                    }
                    if let (Some(n), Some(bytes)) = (min_len, bytes) {
                        let need = elem.storage_bytes(*n);
                        if *bytes < need {
                            return Err(format!(
                                "argument {i} (`{name}`) needs {n} {elem} elements ({need} bytes), the pointer \
                                 reaches {bytes}"
                            ));
                        }
                    }
                    out.push(&address.to_le_bytes());
                }
                (ParamKind::Scalar { dtype, multiple_of }, LaunchValue::Scalar(a)) => {
                    let bits = scalar_for(name, i, *dtype, *multiple_of, a)?;
                    let size = dtype.bytes().max(1);
                    out.push(&bits.to_le_bytes()[..size.min(16)]);
                }
                (ParamKind::ByValue { size, .. }, LaunchValue::Bytes(b)) => {
                    if b.len() != *size {
                        return Err(format!(
                            "argument {i} (`{name}`) is passed by value as {size} bytes, got {}",
                            b.len()
                        ));
                    }
                    out.push(b);
                }
                (kind, _) => {
                    let want = match kind {
                        ParamKind::Pointer { elem, .. } => format!("a pointer to {elem} (a device buffer)"),
                        ParamKind::Scalar { dtype, .. } => format!("a {dtype} scalar"),
                        ParamKind::ByValue { size, .. } => format!("{size} bytes by value"),
                    };
                    return Err(format!("argument {i} (`{name}`) must be {want}"));
                }
            }
        }
        Ok(out)
    }
}

impl fmt::Display for KernelSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| match &p.kind {
                ParamKind::Pointer { elem, align, .. } => format!("{}: ptr<{elem}, align {align}>", p.name),
                ParamKind::Scalar { dtype, multiple_of: 1 } => format!("{}: {dtype}", p.name),
                ParamKind::Scalar { dtype, multiple_of } => format!("{}: {dtype} div {multiple_of}", p.name),
                ParamKind::ByValue { size, align, grid_constant } => format!(
                    "{}: byval[{size}, align {align}]{}",
                    p.name,
                    if *grid_constant { " grid_constant" } else { "" }
                ),
            })
            .collect();
        write!(f, "{}({})", self.name, params.join(", "))
    }
}

/// Why a signature's text ([`KernelSignature::from_text`]) could not be read: the line (from 1)
/// and what is wrong with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for SignatureParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel signature, line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for SignatureParseError {}

/// Whether `s` can stand as one token of the signature text: no whitespace, `=` or control
/// characters, and not empty.
fn is_token(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_graphic() && c != '=')
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

/// The `key=value` fields of one line, which must be exactly `keys`, in that order.
fn fields<'t>(tokens: &[&'t str], keys: &[&str]) -> Result<Vec<&'t str>, String> {
    if tokens.len() != keys.len() {
        let want: Vec<String> = keys.iter().map(|k| format!("{k}=...")).collect();
        return Err(format!("expected {} fields ({}), found {}", keys.len(), want.join(" "), tokens.len()));
    }
    tokens
        .iter()
        .zip(keys)
        .map(|(t, k)| match t.split_once('=') {
            Some((key, value)) if key == *k && !value.is_empty() => Ok(value),
            _ => Err(format!("expected `{k}=...`, found `{t}`")),
        })
        .collect()
}

fn number<T: std::str::FromStr>(what: &str, s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("{what} must be a number, found `{s}`"))
}

fn boolean(what: &str, s: &str) -> Result<bool, String> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{what} must be true or false, found `{s}`")),
    }
}

fn dtype(s: &str) -> Result<DType, String> {
    DType::from_cute_name(s)
        .ok_or_else(|| format!("unknown element type `{s}` (CuTe DSL's names: Float32, Int32, ...)"))
}

fn dims(what: &str, tokens: &[&str]) -> Result<[u32; 3], String> {
    match tokens {
        [x, y, z] => Ok([number(what, x)?, number(what, y)?, number(what, z)?]),
        _ => Err(format!("{what} takes three numbers, found {}", tokens.len())),
    }
}

/// `n` elements of `elem` in bytes, unless that overflows.
fn checked_storage(what: &str, elem: DType, n: usize) -> Result<usize, String> {
    n.checked_mul(elem.storage_bits())
        .map(|bits| bits.div_ceil(8))
        .ok_or_else(|| format!("{what} of {n} {elem} elements is more bytes than this machine addresses"))
}

/// A byte count that the launcher adds to a 32-bit launch size: at most `u32::MAX`.
fn byte_count(what: &str, s: &str) -> Result<usize, String> {
    let n: usize = number(what, s)?;
    if n > u32::MAX as usize {
        return Err(format!("{what} must be at most {}, found {n}", u32::MAX));
    }
    Ok(n)
}

impl KernelSignature {
    /// Why the signature cannot be written as text ([`to_text`](Self::to_text)), if it cannot:
    /// a kernel, parameter or global name that is not one token (empty, or with whitespace,
    /// `=` or characters outside printable ASCII), or a kernel with tensor-map parameters
    /// (the text has no form for them: an export would lose which parameters are hidden and
    /// which must be driver-encoded).
    pub fn text_error(&self) -> Option<String> {
        if !self.tensor_maps.is_empty() {
            return Some(format!(
                "kernel `{}` has tensor-map parameters (tma_descriptor or param_tensormap), which an exported \
                 kernel signature cannot carry yet: launch it with cute-rs",
                self.name
            ));
        }
        let names = std::iter::once(("kernel name", &self.name))
            .chain(self.params.iter().map(|p| ("parameter name", &p.name)))
            .chain(self.globals.iter().map(|g| ("global name", &g.name)));
        for (what, name) in names {
            if !is_token(name) {
                return Some(format!(
                    "{what} `{name}` cannot be written in a kernel signature: use a name of printable ASCII \
                     without spaces or `=`"
                ));
            }
        }
        None
    }

    /// The signature as text, one item per line, that [`from_text`](Self::from_text) reads back:
    /// what an exported kernel carries next to its PTX and cubin
    /// ([`aot`](crate::aot)), so that it can be launched with checked arguments without the
    /// kernel's IR. Shared-buffer starts are left out (a launch needs only the sizes).
    ///
    /// ```text
    /// kernel add_one
    /// target sm_89
    /// ptx_version 78
    /// block_dim 128 1 1
    /// smem static_bytes=0 in_window=false dynamic_start=0 window_align=16 dynamic_region=false
    /// param a pointer Float32 len=1024 align=16
    /// param n scalar Int32 multiple_of=1
    /// param s by_value size=16 align=8 grid_constant=false
    /// global g Float32 len=4 constant=true readonly=true init=0000803f...
    /// ```
    ///
    /// Panics if a name is not one token ([`text_error`](Self::text_error)).
    pub fn to_text(&self) -> String {
        if let Some(e) = self.text_error() {
            panic!("{e}");
        }
        let token = |_: &str, s: &str| s.to_string();
        let mut out = format!("kernel {}\n", token("kernel name", &self.name));
        if let Some(t) = self.target {
            out += &format!("target {}\n", t.chip());
        }
        if let Some(v) = self.ptx_version {
            out += &format!("ptx_version {v}\n");
        }
        if let Some([x, y, z]) = self.block_dim {
            out += &format!("block_dim {x} {y} {z}\n");
        }
        if let Some([x, y, z]) = self.max_threads {
            out += &format!("max_threads {x} {y} {z}\n");
        }
        if let Some(n) = self.min_blocks_per_sm {
            out += &format!("min_blocks_per_sm {n}\n");
        }
        let s = &self.smem;
        out += &format!(
            "smem static_bytes={} in_window={} dynamic_start={} window_align={} dynamic_region={}\n",
            s.static_bytes, s.in_window, s.dynamic_start, s.window_align, self.dynamic_smem_region
        );
        for p in &self.params {
            let name = token("parameter name", &p.name);
            out += &match &p.kind {
                ParamKind::Pointer { elem, min_len, align } => format!(
                    "param {name} pointer {} len={} align={align}\n",
                    elem.cute_name(),
                    min_len.map_or("?".to_string(), |n| n.to_string())
                ),
                ParamKind::Scalar { dtype, multiple_of } => {
                    format!("param {name} scalar {} multiple_of={multiple_of}\n", dtype.cute_name())
                }
                ParamKind::ByValue { size, align, grid_constant } => {
                    format!("param {name} by_value size={size} align={align} grid_constant={grid_constant}\n")
                }
            };
        }
        for g in &self.globals {
            out += &format!(
                "global {} {} len={} constant={} readonly={} init={}\n",
                token("global name", &g.name),
                g.elem.cute_name(),
                g.len,
                g.constant,
                g.readonly,
                g.init.as_deref().map_or("-".to_string(), hex)
            );
        }
        out
    }

    /// Read a signature written by [`to_text`](Self::to_text). Blank lines and lines starting
    /// with `#` are skipped; anything else that is not exactly as `to_text` writes it (an unknown
    /// item, a repeated one, a missing `kernel` or `smem` line, a bad number, type or target)
    /// fails with the line.
    pub fn from_text(text: &str) -> Result<KernelSignature, SignatureParseError> {
        let mut name = None;
        let mut sig = KernelSignature {
            name: String::new(),
            params: vec![],
            tensor_maps: vec![],
            globals: vec![],
            smem: SharedMemoryLayout {
                starts: vec![],
                static_bytes: 0,
                in_window: false,
                dynamic_start: 0,
                window_align: 16,
                reserved_bytes: 0,
            },
            dynamic_smem_region: false,
            block_dim: None,
            max_threads: None,
            min_blocks_per_sm: None,
            target: None,
            ptx_version: None,
        };
        let mut seen: Vec<&str> = vec![];
        for (i, line) in text.lines().enumerate() {
            let fail = |message: String| SignatureParseError { line: i + 1, message };
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let (key, rest) = (tokens[0], &tokens[1..]);
            if !matches!(key, "param" | "global") {
                if seen.contains(&key) {
                    return Err(fail(format!("`{key}` appears twice")));
                }
                seen.push(key);
            }
            let single = |what: &str| match rest {
                [v] => Ok(*v),
                _ => Err(format!("`{what}` takes one value, found {}", rest.len())),
            };
            let result: Result<(), String> = (|| {
                match key {
                    "kernel" => name = Some(single("kernel")?.to_string()),
                    "target" => sig.target = Some(single("target")?.parse::<crate::Arch>().map_err(|e| e.to_string())?),
                    "ptx_version" => sig.ptx_version = Some(number("ptx_version", single("ptx_version")?)?),
                    "block_dim" => sig.block_dim = Some(dims("block_dim", rest)?),
                    "max_threads" => sig.max_threads = Some(dims("max_threads", rest)?),
                    "min_blocks_per_sm" => {
                        sig.min_blocks_per_sm = Some(number("min_blocks_per_sm", single("min_blocks_per_sm")?)?)
                    }
                    "smem" => {
                        let f = fields(
                            rest,
                            &["static_bytes", "in_window", "dynamic_start", "window_align", "dynamic_region"],
                        )?;
                        sig.smem.static_bytes = byte_count("static_bytes", f[0])?;
                        sig.smem.in_window = boolean("in_window", f[1])?;
                        sig.smem.dynamic_start = byte_count("dynamic_start", f[2])?;
                        sig.smem.window_align = number("window_align", f[3])?;
                        sig.dynamic_smem_region = boolean("dynamic_region", f[4])?;
                    }
                    "param" => {
                        let [pname, kind, more @ ..] = rest else {
                            return Err("`param` takes a name and a kind".to_string());
                        };
                        if sig.params.iter().any(|p| p.name == *pname) {
                            return Err(format!("parameter `{pname}` appears twice"));
                        }
                        let kind = match (*kind, more) {
                            ("pointer", [elem, f @ ..]) => {
                                let f = fields(f, &["len", "align"])?;
                                let elem = dtype(elem)?;
                                let min_len = if f[0] == "?" { None } else { Some(number("len", f[0])?) };
                                if let Some(n) = min_len {
                                    checked_storage("a pointer", elem, n)?;
                                }
                                ParamKind::Pointer { elem, min_len, align: number("align", f[1])? }
                            }
                            ("scalar", [t, f @ ..]) => {
                                let f = fields(f, &["multiple_of"])?;
                                let multiple_of = number("multiple_of", f[0])?;
                                if multiple_of == 0 {
                                    return Err("multiple_of must be at least 1".to_string());
                                }
                                ParamKind::Scalar { dtype: dtype(t)?, multiple_of }
                            }
                            ("by_value", f) => {
                                let f = fields(f, &["size", "align", "grid_constant"])?;
                                ParamKind::ByValue {
                                    size: number("size", f[0])?,
                                    align: number("align", f[1])?,
                                    grid_constant: boolean("grid_constant", f[2])?,
                                }
                            }
                            (k, _) => {
                                return Err(format!(
                                    "parameter `{pname}`: expected `pointer <type> ...`, `scalar <type> ...` or \
                                     `by_value ...`, found `{k}`"
                                ));
                            }
                        };
                        sig.params.push(ParamSignature { name: pname.to_string(), kind });
                    }
                    "global" => {
                        let [gname, elem, f @ ..] = rest else {
                            return Err("`global` takes a name and an element type".to_string());
                        };
                        if sig.globals.iter().any(|g| g.name == *gname) {
                            return Err(format!("global `{gname}` appears twice"));
                        }
                        let f = fields(f, &["len", "constant", "readonly", "init"])?;
                        let elem = dtype(elem)?;
                        let len = number("len", f[0])?;
                        let bytes = checked_storage("a global", elem, len)?;
                        let init = match f[3] {
                            "-" => None,
                            h => Some(unhex(h).ok_or_else(|| format!("init must be hex bytes or `-`, found `{h}`"))?),
                        };
                        if let Some(init) = &init
                            && init.len() > bytes
                        {
                            return Err(format!(
                                "global `{gname}` has {} bytes of init for {len} {elem} elements ({bytes} bytes)",
                                init.len()
                            ));
                        }
                        sig.globals.push(GlobalSignature {
                            name: gname.to_string(),
                            elem,
                            len,
                            constant: boolean("constant", f[1])?,
                            readonly: boolean("readonly", f[2])?,
                            init,
                        });
                    }
                    other => return Err(format!("unknown item `{other}`")),
                }
                Ok(())
            })();
            result.map_err(fail)?;
        }
        let last = text.lines().count().max(1);
        sig.name = name.ok_or(SignatureParseError { line: last, message: "no `kernel` line".to_string() })?;
        if !seen.contains(&"smem") {
            return Err(SignatureParseError { line: last, message: "no `smem` line".to_string() });
        }
        Ok(sig)
    }
}

impl Kernel {
    /// The kernel's launch signature.
    pub fn signature(&self) -> KernelSignature {
        let mut s = KernelSignature::of(self.ir());
        s.target = self.target();
        s
    }
}

/// Whether the kernel may write buffer `b`: a store, an atomic, a reduction or a bulk copy
/// into it; inline PTX or an MLIR operation given its address whose text may store; or, once
/// its address is taken as an integer (`Ptr::toint`), any write
/// through a raw pointer or any such inline PTX or MLIR operation, which could reach it
/// through that integer. Conservative: a kernel that only reads through those paths is
/// reported as writing.
pub(crate) fn writes_buffer(ir: &KernelIr, b: crate::ir::BufferId) -> bool {
    use crate::ir::{AsmInput, Op, Region};
    /// Whether inline assembly or an MLIR operation's text may write memory.
    fn may_store(text: &str) -> bool {
        let t = text.to_ascii_lowercase();
        [
            "st.",
            "st ",
            "stmatrix",
            "atom",
            "red.",
            "red ",
            "cp.",
            "store",
            "tensormap",
            "multimem",
            "bulk",
            "memcpy",
            "memset",
        ]
        .iter()
        .any(|m| t.contains(m))
    }
    #[derive(Default)]
    struct Found {
        direct: bool,
        escapes: bool,
        raw_or_asm_write: bool,
    }
    fn walk(ir: &KernelIr, r: &Region, b: crate::ir::BufferId, f: &mut Found) {
        for inst in &r.insts {
            match &inst.op {
                Op::Store { buffer, .. }
                | Op::Atomic { buffer, .. }
                | Op::Red { buffer, .. }
                | Op::StMatrix { buffer, .. } => {
                    f.direct |= *buffer == b;
                    f.raw_or_asm_write |= matches!(ir.buffer(*buffer).space, MemSpace::Raw(_));
                }
                Op::CpAsyncBulk { dst, .. } => {
                    f.direct |= *dst == b;
                    f.raw_or_asm_write |= matches!(ir.buffer(*dst).space, MemSpace::Raw(_));
                }
                // An override store writes wherever its address points.
                Op::TmaStore { overrides: Some(_), .. } => f.direct |= ir.buffer(b).space == MemSpace::Global,
                // A TMA store writes the tensor its descriptor describes: any of the kernel's.
                Op::TmaStore { .. } => {
                    f.direct |= ir.tensor_maps.iter().any(|m| m.source.as_ref().is_some_and(|src| src.tensor == b))
                }
                Op::PtrToInt { buffer, .. } => f.escapes |= *buffer == b,
                Op::InlinePtx { template: text, inputs: operands, .. } | Op::MlirOp { text, operands, .. }
                    if may_store(text) =>
                {
                    f.raw_or_asm_write = true;
                    f.direct |= operands.iter().any(|o| matches!(o, AsmInput::Address { buffer, .. } if *buffer == b));
                }
                _ => {}
            }
            for region in inst.op.regions() {
                walk(ir, region, b, f);
            }
        }
    }
    let mut f = Found::default();
    walk(ir, &ir.body, b, &mut f);
    f.direct || (f.escapes && f.raw_or_asm_write)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_scalars_are_checked_as_signed() {
        // 0xFFFF_FFFF is a multiple of 3; -1 is not.
        assert!(scalar_for("n", 0, DType::I32, 3, &KernelArg::ScalarI32(-1)).is_err());
        assert_eq!(scalar_for("n", 0, DType::I32, 3, &KernelArg::ScalarI32(-6)), Ok(-6i32 as u32 as u128));
        let e = scalar_for("n", 0, DType::I32, 4, &KernelArg::ScalarI32(-6)).unwrap_err();
        assert!(e.ends_with("got -6"), "{e}");
    }
}

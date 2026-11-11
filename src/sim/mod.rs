//! A CPU simulator for kernels.
//!
//! The simulator runs every thread of a block in lockstep over the
//! structured IR, with per-thread masks for divergent branches and loops.
//! Warp-collective instructions (`ldmatrix`, `mma.sync`) are evaluated from
//! the PTX ISA's fragment definitions, written independently of the layout
//! algebra, so a simulated kernel that produces the right answer is evidence
//! that its layouts and partitions are right.
//!
//! Besides computing results it checks for the mistakes that are easy to
//! make in CuTe-style kernels and hard to debug on hardware:
//!
//! * shared-memory data races between warps (accesses not ordered by
//!   `sync_threads`, a named barrier or an mbarrier),
//! * reads of shared memory that a `cp.async` or bulk copy is still
//!   writing, and reads of never-written shared memory,
//! * out-of-bounds and misaligned global/shared accesses,
//! * barriers and warp-collective instructions in divergent code, and named
//!   barriers and mbarriers used inconsistently or unable to complete,
//! * bulk copies that an mbarrier phase's transaction count does not cover
//!   (missing or too small `expect_tx`), and mbarrier parity waits by a warp
//!   that has not seen the previous phase complete, which could pass at
//!   once on hardware however lockstep execution orders them,
//! * integer division by zero and oversized shifts.
//!
//! Errors carry the Rust source location of the operation that failed. The
//! [`SimReport`] also counts shared-memory bank-conflict wavefronts and
//! global-memory sectors, which shows the effect of swizzles and
//! vectorisation without a profiler.

mod exec;
mod float;

use std::fmt;

use crate::builder::Kernel;
use crate::ir::{DType, KernelIr, Loc, Param};
use crate::value::{F16, Numeric};

/// Grid and block dimensions of a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Blocks per cluster (`[1, 1, 1]` without clusters). Each mode of `grid`
    /// must be a multiple of it.
    pub cluster: [u32; 3],
}

impl LaunchConfig {
    /// A launch of `grid` blocks of `block` threads.
    pub fn new(grid: [u32; 3], block: [u32; 3]) -> LaunchConfig {
        LaunchConfig { grid, block, cluster: [1, 1, 1] }
    }

    /// The same launch grouped into block clusters of `cluster`, which run
    /// together.
    pub fn with_cluster(mut self, cluster: [u32; 3]) -> LaunchConfig {
        self.cluster = cluster;
        self
    }

    pub fn threads_per_block(&self) -> usize {
        self.block.iter().map(|&d| d as usize).product()
    }
}

/// A kernel argument: a buffer for each tensor parameter, a scalar for each
/// scalar parameter, in declaration order.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelArg {
    F32(Vec<f32>),
    F16(Vec<F16>),
    I32(Vec<i32>),
    Bool(Vec<bool>),
    ScalarF32(f32),
    ScalarF16(F16),
    ScalarI32(i32),
    ScalarBool(bool),
    /// A buffer of any element type, as little-endian bytes.
    Bytes {
        elem: DType,
        data: Vec<u8>,
    },
    /// A scalar of any type, as raw bits.
    Scalar {
        ty: DType,
        bits: u64,
    },
}

impl KernelArg {
    /// An f32 buffer.
    pub fn f32(v: Vec<f32>) -> KernelArg {
        KernelArg::F32(v)
    }

    /// An f16 buffer, rounding each value to the nearest half.
    pub fn f16_from_f32(v: &[f32]) -> KernelArg {
        KernelArg::F16(v.iter().map(|&x| F16::from_f32(x)).collect())
    }

    /// An i32 buffer.
    pub fn i32(v: Vec<i32>) -> KernelArg {
        KernelArg::I32(v)
    }

    /// A buffer of any element type.
    pub fn buffer<T: Numeric>(values: &[T]) -> KernelArg {
        let size = T::DTYPE.bytes();
        let data = values.iter().flat_map(|v| v.raw_bits().to_le_bytes().into_iter().take(size)).collect();
        KernelArg::Bytes { elem: T::DTYPE, data }
    }

    /// A scalar of any type.
    pub fn scalar<T: Numeric>(value: T) -> KernelArg {
        KernelArg::Scalar { ty: T::DTYPE, bits: value.raw_bits() }
    }

    /// The contents of a buffer as `T`, which must be its element type.
    pub fn to_vec<T: Numeric>(&self) -> Vec<T> {
        match self.buffer_type() {
            Some((t, _)) if t == T::DTYPE => {}
            _ => panic!("argument is {}, not a {} buffer", self.describe(), T::DTYPE),
        }
        self.to_bytes()
            .chunks_exact(T::DTYPE.bytes())
            .map(|c| T::from_bits(c.iter().rev().fold(0u64, |acc, &b| (acc << 8) | b as u64)))
            .collect()
    }

    /// The contents of an f32 buffer.
    pub fn as_f32(&self) -> &[f32] {
        match self {
            KernelArg::F32(v) => v,
            other => panic!("argument is {}, not an f32 buffer", other.describe()),
        }
    }

    /// The contents of an f16 buffer.
    pub fn as_f16(&self) -> &[F16] {
        match self {
            KernelArg::F16(v) => v,
            other => panic!("argument is {}, not an f16 buffer", other.describe()),
        }
    }

    /// The contents of an i32 buffer.
    pub fn as_i32(&self) -> &[i32] {
        match self {
            KernelArg::I32(v) => v,
            other => panic!("argument is {}, not an i32 buffer", other.describe()),
        }
    }

    pub(crate) fn describe(&self) -> String {
        match self {
            KernelArg::F32(v) => format!("an f32 buffer of {}", v.len()),
            KernelArg::F16(v) => format!("an f16 buffer of {}", v.len()),
            KernelArg::I32(v) => format!("an i32 buffer of {}", v.len()),
            KernelArg::Bool(v) => format!("a bool buffer of {}", v.len()),
            KernelArg::ScalarF32(_) => "an f32 scalar".into(),
            KernelArg::ScalarF16(_) => "an f16 scalar".into(),
            KernelArg::ScalarI32(_) => "an i32 scalar".into(),
            KernelArg::ScalarBool(_) => "a bool scalar".into(),
            KernelArg::Bytes { elem, data } => format!("a {elem} buffer of {}", data.len() / elem.bytes()),
            KernelArg::Scalar { ty, .. } => format!("a {ty} scalar"),
        }
    }

    fn buffer_type(&self) -> Option<(DType, usize)> {
        Some(match self {
            KernelArg::F32(v) => (DType::F32, v.len()),
            KernelArg::F16(v) => (DType::F16, v.len()),
            KernelArg::I32(v) => (DType::I32, v.len()),
            KernelArg::Bool(v) => (DType::Bool, v.len()),
            KernelArg::Bytes { elem, data } => (*elem, data.len() / elem.bytes()),
            _ => return None,
        })
    }

    fn scalar_bits(&self) -> Option<(DType, u64)> {
        Some(match self {
            KernelArg::ScalarF32(x) => (DType::F32, x.to_bits() as u64),
            KernelArg::ScalarF16(x) => (DType::F16, x.0 as u64),
            KernelArg::ScalarI32(x) => (DType::I32, *x as u32 as u64),
            KernelArg::ScalarBool(x) => (DType::Bool, *x as u64),
            KernelArg::Scalar { ty, bits } => (*ty, *bits),
            _ => return None,
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        match self {
            KernelArg::F32(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            KernelArg::F16(v) => v.iter().flat_map(|x| x.0.to_le_bytes()).collect(),
            KernelArg::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            KernelArg::Bool(v) => v.iter().map(|&x| x as u8).collect(),
            KernelArg::Bytes { data, .. } => data.clone(),
            _ => unreachable!(),
        }
    }

    fn set_from_bytes(&mut self, bytes: &[u8]) {
        match self {
            KernelArg::F32(v) => v.iter_mut().zip(bytes.as_chunks().0).for_each(|(x, b)| *x = f32::from_le_bytes(*b)),
            KernelArg::F16(v) => {
                v.iter_mut().zip(bytes.as_chunks().0).for_each(|(x, b)| *x = F16(u16::from_le_bytes(*b)))
            }
            KernelArg::I32(v) => v.iter_mut().zip(bytes.as_chunks().0).for_each(|(x, b)| *x = i32::from_le_bytes(*b)),
            KernelArg::Bool(v) => v.iter_mut().zip(bytes).for_each(|(x, &b)| *x = b != 0),
            KernelArg::Bytes { data, .. } => data.copy_from_slice(bytes),
            _ => unreachable!(),
        }
    }
}

/// What the simulator checks.
#[derive(Clone, Debug)]
pub struct SimOptions {
    /// Report shared-memory accesses from different warps that are not
    /// separated by a barrier.
    pub detect_races: bool,
    /// Report reads of shared memory that was never written.
    pub detect_uninitialized: bool,
    /// Iterations of one `while_carry` loop before the simulator reports
    /// that it does not terminate.
    pub max_while_iterations: u64,
}

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions { detect_races: true, detect_uninitialized: true, max_while_iterations: 1 << 20 }
    }
}

/// Execution statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SimReport {
    pub blocks: u64,
    /// Instructions executed, counted once per warp that executed them.
    pub warp_instructions: u64,
    pub mma_instructions: u64,
    pub barriers: u64,
    pub global_bytes_read: u64,
    pub global_bytes_written: u64,
    /// 32-byte global memory sectors touched, per warp-level request.
    pub global_sectors: u64,
    /// The fewest sectors those requests could have touched.
    pub global_sectors_ideal: u64,
    /// Shared memory wavefronts (bank-conflict-serialised passes).
    pub shared_wavefronts: u64,
    /// The wavefronts the same accesses need without bank conflicts.
    pub shared_wavefronts_ideal: u64,
    /// The kernel's `printf` output, one entry per call per thread.
    pub printed: Vec<String>,
}

impl SimReport {
    /// Average shared-memory serialisation from bank conflicts (1.0 = none).
    pub fn bank_conflict_factor(&self) -> f64 {
        if self.shared_wavefronts_ideal == 0 {
            1.0
        } else {
            self.shared_wavefronts as f64 / self.shared_wavefronts_ideal as f64
        }
    }

    /// Fraction of fetched global sectors that the kernel needed (1.0 =
    /// perfectly coalesced).
    pub fn global_efficiency(&self) -> f64 {
        if self.global_sectors == 0 { 1.0 } else { self.global_sectors_ideal as f64 / self.global_sectors as f64 }
    }
}

impl fmt::Display for SimReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "blocks:              {}", self.blocks)?;
        writeln!(f, "warp instructions:   {}", self.warp_instructions)?;
        writeln!(f, "mma.sync:            {}", self.mma_instructions)?;
        writeln!(f, "barriers:            {}", self.barriers)?;
        writeln!(f, "global read/written: {} / {} bytes", self.global_bytes_read, self.global_bytes_written)?;
        writeln!(
            f,
            "global efficiency:   {:.0}% ({} sectors, {} ideal)",
            100.0 * self.global_efficiency(),
            self.global_sectors,
            self.global_sectors_ideal
        )?;
        write!(
            f,
            "smem bank conflicts: {:.2}x ({} wavefronts, {} ideal)",
            self.bank_conflict_factor(),
            self.shared_wavefronts,
            self.shared_wavefronts_ideal
        )
    }
}

/// A simulation failure.
#[derive(Clone, Debug)]
pub struct SimError {
    pub message: String,
    /// Source location of the operation that failed.
    pub loc: Option<Loc>,
    pub block: [u32; 3],
    pub thread: Option<[u32; 3]>,
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(loc) = self.loc {
            write!(f, "{}:{}:{}: ", loc.file(), loc.line(), loc.column())?;
        }
        let [x, y, z] = self.block;
        write!(f, "block ({x},{y},{z})")?;
        if let Some([x, y, z]) = self.thread {
            write!(f, " thread ({x},{y},{z})")?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for SimError {}

/// Check a launch configuration and arguments against a kernel's
/// signature (shared by the simulator and the CUDA launcher).
pub(crate) fn check_launch(ir: &KernelIr, cfg: LaunchConfig, args: &[KernelArg]) -> Result<(), String> {
    if args.len() != ir.args.len() {
        return Err(format!("kernel `{}` takes {} arguments, {} were given", ir.name, ir.args.len(), args.len()));
    }
    if cfg.threads_per_block() == 0 || cfg.threads_per_block() > 1024 {
        return Err(format!("invalid block size {:?}", cfg.block));
    }
    if cfg.grid.contains(&0) {
        return Err(format!("invalid grid size {:?}", cfg.grid));
    }
    if cfg.cluster.contains(&0) || cfg.grid.iter().zip(cfg.cluster).any(|(&g, c)| g % c != 0) {
        return Err(format!("grid {:?} is not a whole number of clusters of {:?}", cfg.grid, cfg.cluster));
    }
    if let Some(dims) = ir.block_dim
        && dims != cfg.block
    {
        return Err(format!("kernel `{}` was built for blocks of {dims:?}, launched with {:?}", ir.name, cfg.block));
    }
    for (i, (p, a)) in ir.args.iter().zip(args).enumerate() {
        match p {
            Param::Buffer(b) => {
                let d = ir.buffer(*b);
                match a.buffer_type() {
                    Some((t, len)) if t == d.elem && d.len.is_none_or(|n| len >= n) => {}
                    Some((t, len)) if t == d.elem => {
                        return Err(format!(
                            "argument {i} (`{}`) needs at least {} elements, got {len}",
                            d.name,
                            d.len.unwrap_or(0)
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "argument {i} (`{}`) must be a {} buffer, got {}",
                            d.name,
                            d.elem,
                            a.describe()
                        ));
                    }
                }
            }
            Param::Scalar { name, value, multiple_of } => match a.scalar_bits() {
                Some((t, bits)) if t == ir.dtype(*value) && (bits as usize).is_multiple_of(*multiple_of) => {}
                Some((t, bits)) if t == ir.dtype(*value) => {
                    return Err(format!("argument {i} (`{name}`) must be a multiple of {multiple_of}, got {bits}"));
                }
                _ => {
                    return Err(format!(
                        "argument {i} (`{name}`) must be a {} scalar, got {}",
                        ir.dtype(*value),
                        a.describe()
                    ));
                }
            },
        }
    }
    Ok(())
}

impl Kernel {
    /// Run the kernel on the CPU. `args` holds one entry per parameter, in
    /// declaration order; output buffers are updated in place.
    pub fn simulate(&self, cfg: LaunchConfig, args: &mut [KernelArg]) -> Result<SimReport, SimError> {
        self.simulate_with(cfg, args, &SimOptions::default())
    }

    /// [`Kernel::simulate`] with explicit checking options.
    pub fn simulate_with(
        &self,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        opts: &SimOptions,
    ) -> Result<SimReport, SimError> {
        let ir = self.ir();
        check_launch(ir, cfg, args).map_err(|message| SimError { message, loc: None, block: [0; 3], thread: None })?;
        let mut globals = vec![None; ir.buffers.len()];
        let mut scalars = Vec::new();
        for (p, a) in ir.args.iter().zip(args.iter()) {
            match p {
                Param::Buffer(b) => globals[b.0 as usize] = Some(a.to_bytes()),
                Param::Scalar { value, .. } => scalars.push((*value, a.scalar_bits().unwrap().1)),
            }
        }
        let mut machine = exec::Machine::new(ir, opts, cfg, globals, scalars);
        machine.run()?;
        let (globals, report) = machine.finish();
        for (p, a) in ir.args.iter().zip(args.iter_mut()) {
            if let Param::Buffer(b) = p {
                a.set_from_bytes(globals[b.0 as usize].as_ref().unwrap());
            }
        }
        Ok(report)
    }
}

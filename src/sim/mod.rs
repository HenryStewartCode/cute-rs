//! A CPU simulator for kernels.
//!
//! The simulator runs each warp on its own over the structured IR, with
//! per-lane masks for loops and a convergence group per side of a
//! divergent branch (independent thread scheduling), and interleaves the
//! warps of a block (or cluster) and their groups with a deterministic or
//! seeded schedule ([`Schedule`]): warp-specialised kernels, spin-waits and
//! barriers whose arrivals come later in program order run as on hardware.
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
//!   once on hardware whatever order the simulator runs them in,
//! * deadlocks (every warp that has not exited blocked), reported with
//!   each blocked warp's line, enclosing loops and branches and what it
//!   waits for, and hangs (nothing visible happening while every warp is
//!   blocked or spinning), reported with what the spinning warps poll,
//! * integer division by zero and oversized shifts.
//!
//! Errors carry the Rust source location of the operation that failed. The
//! [`SimReport`] also counts shared-memory bank-conflict wavefronts and
//! global-memory sectors, which shows the effect of swizzles and
//! vectorisation without a profiler.

mod carry;
mod exec;
pub use exec::{SIM_GLOBAL_BASE, SIM_LOCAL_WINDOW, SIM_SHARED_WINDOW, setmaxnreg_initial_budget};
mod float;
mod int;
mod module;
pub use module::{SimKernelModule, SimModule};
mod printf;
mod stream;
// m11-extern-distributed: multicast replicas.
mod multicast;
/// One 32-bit word of a simulated `multimem.ld_reduce` over the replicas' words, in replica order.
pub use exec::ld_reduce_word as multimem_ld_reduce_word;
pub use multicast::MulticastReplicas;
pub use stream::{SimBuffer, SimStream, StreamArg};

/// [`crate::atom::mma::sparse_metadata_lane`]: where the metadata of a chunk
/// of a sparse A lives, with sparsity selector 0.
pub(crate) fn exec_sparse_metadata_slot(k: usize, row: usize, chunk: usize) -> (usize, usize) {
    exec::sparse_metadata_slot(k, 0, row, chunk)
}

use std::fmt;

use crate::builder::Kernel;
use crate::ir::{DType, Loc};
use crate::numeric::NarrowRounding;
use crate::value::{F16, Numeric};

/// Grid and block dimensions of a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Blocks per cluster (`[1, 1, 1]` without clusters). Each mode of `grid`
    /// must be a multiple of it.
    pub cluster: [u32; 3],
    /// Bytes of the kernel's dynamic shared-memory region
    /// ([`get_dyn_smem`](crate::get_dyn_smem)) per block (0 by default), on
    /// top of the shared memory sized when the kernel was built. Static and
    /// dynamic shared memory together may not exceed what the device gives
    /// a block.
    pub dynamic_smem_bytes: u32,
    /// The launch's shared memory per block, CuTe DSL's `.launch(smem=n)`: it must cover
    /// what the kernel allocates when it is built, and the rest goes to the dynamic region.
    /// `None` (the default) means the kernel's allocations plus
    /// [`dynamic_smem_bytes`](Self::dynamic_smem_bytes). Set one or the other.
    pub smem_bytes: Option<u32>,
    /// The guaranteed cluster shape when [`cluster`](Self::cluster) is only preferred
    /// (CuTe DSL's `fallback_cluster`): the hardware runs each cluster of the grid at the
    /// preferred shape or at this one, so kernels read their cluster shape at run time.
    /// The preferred shape must be a whole number of these in every mode. Preferred
    /// clusters need SM100.
    pub fallback_cluster: Option<[u32; 3]>,
    /// The share of the unified L1 and shared storage to make shared memory, in percent
    /// (CuTe DSL's `preferred_smem_carveout`; `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`):
    /// -1 the driver's default, 0 the most L1, 100 the most shared memory. A hint; the
    /// simulator ignores it.
    pub preferred_smem_carveout: Option<i32>,
    /// Programmatic dependent launch (CuTe DSL's `use_pdl=True`;
    /// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`): the launch may start before
    /// the previous kernel in its stream finishes, and waits for it with
    /// [`griddepcontrol_wait`](crate::griddepcontrol_wait). The overlap needs SM90; on SM89
    /// driver 595 accepts the attribute and runs the kernels one after the other.
    /// [`SimStream`] models it.
    pub use_pdl: bool,
    /// A cooperative launch (CuTe DSL's `cooperative=True`; `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`):
    /// every block of the grid is resident at once, or the launch fails, so blocks may wait
    /// for each other through global memory (a grid-wide barrier). The simulator runs the
    /// whole grid together ([`ResidentClusters::Grid`]).
    pub cooperative: bool,
}

impl LaunchConfig {
    /// A launch of `grid` blocks of `block` threads.
    pub fn new(grid: [u32; 3], block: [u32; 3]) -> LaunchConfig {
        LaunchConfig {
            grid,
            block,
            cluster: [1, 1, 1],
            dynamic_smem_bytes: 0,
            smem_bytes: None,
            fallback_cluster: None,
            preferred_smem_carveout: None,
            use_pdl: false,
            cooperative: false,
        }
    }

    /// The same launch as a programmatic dependent launch, or not
    /// ([`use_pdl`](Self::use_pdl)).
    pub fn with_pdl(mut self, yes: bool) -> LaunchConfig {
        self.use_pdl = yes;
        self
    }

    /// The same launch as a cooperative launch, or not ([`cooperative`](Self::cooperative)).
    pub fn with_cooperative(mut self, yes: bool) -> LaunchConfig {
        self.cooperative = yes;
        self
    }

    /// The same launch grouped into block clusters of `cluster`, which run
    /// together.
    pub fn with_cluster(mut self, cluster: [u32; 3]) -> LaunchConfig {
        self.cluster = cluster;
        self
    }

    /// The same launch with `bytes` bytes of dynamic shared memory per
    /// block: the size of the region [`get_dyn_smem`](crate::get_dyn_smem)
    /// points to (CUDA's `sharedMemBytes`, less the shared memory the kernel
    /// keeps in front of the region).
    pub fn with_dynamic_smem(mut self, bytes: u32) -> LaunchConfig {
        self.dynamic_smem_bytes = bytes;
        self
    }

    /// The same launch with `bytes` bytes of shared memory per block in all (CuTe DSL's
    /// `smem=`), which must cover the kernel's own allocations; the rest goes to the
    /// dynamic region ([`smem_bytes`](Self::smem_bytes)).
    pub fn with_smem(mut self, bytes: u32) -> LaunchConfig {
        self.smem_bytes = Some(bytes);
        self
    }

    /// The same launch with `cluster` as the preferred shape and `fallback` as the
    /// guaranteed one ([`fallback_cluster`](Self::fallback_cluster)).
    pub fn with_fallback_cluster(mut self, cluster: [u32; 3], fallback: [u32; 3]) -> LaunchConfig {
        self.cluster = cluster;
        self.fallback_cluster = Some(fallback);
        self
    }

    /// The same launch with a preferred shared-memory carveout in percent
    /// ([`preferred_smem_carveout`](Self::preferred_smem_carveout)).
    pub fn with_smem_carveout(mut self, percent: i32) -> LaunchConfig {
        assert!((-1..=100).contains(&percent), "a shared-memory carveout is -1 or a percentage, got {percent}");
        self.preferred_smem_carveout = Some(percent);
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
    /// A 128-bit scalar (Int128 or Uint128), as raw bits.
    Scalar128 {
        ty: DType,
        bits: u128,
    },
}

impl KernelArg {
    /// The bytes of a by-value parameter ([`param_struct`](crate::param_struct),
    /// [`NativeLayout::pack`](crate::NativeLayout::pack)).
    pub fn by_value(bytes: Vec<u8>) -> KernelArg {
        KernelArg::Bytes { elem: DType::U8, data: bytes }
    }

    /// The bytes of a [`KernelArg::Bytes`], or none.
    pub(crate) fn byte_data(&self) -> &[u8] {
        match self {
            KernelArg::Bytes { data, .. } => data,
            _ => &[],
        }
    }

    /// An f32 buffer.
    pub fn f32(v: Vec<f32>) -> KernelArg {
        KernelArg::F32(v)
    }

    /// An f16 buffer, rounding each value to the nearest half.
    pub fn f16_from_f32(v: &[f32]) -> KernelArg {
        KernelArg::F16(v.iter().map(|&x| F16::from_f32(x)).collect())
    }

    /// A buffer of an 8-bit float type `T` (an FP8 or scale-factor format)
    /// from f32 values, each rounded on the host as `T::from_f32` rounds: to
    /// nearest even (up for E8M0), saturating to the largest finite value
    /// as `.satfinite` does (CuTe DSL's `create_cute_tensor_for_fp8` has no
    /// other counterpart: a byte buffer needs no DLPack workaround).
    pub fn fp8_from_f32<T: Numeric>(v: &[f32]) -> KernelArg {
        let f = T::DTYPE.float_format().filter(|_| T::DTYPE.is_narrow_float() && T::DTYPE.bits() == 8);
        let f = f.unwrap_or_else(|| panic!("fp8_from_f32: {} is not an 8-bit float type", T::DTYPE));
        let rounding = if T::DTYPE == DType::F8E8M0FNU { NarrowRounding::Up } else { NarrowRounding::NearestEven };
        let data = v.iter().map(|&x| crate::numeric::encode(f, x as f64, rounding, true, false) as u8).collect();
        KernelArg::Bytes { elem: T::DTYPE, data }
    }

    /// The values of a float buffer (an 8-bit float, Float16, BFloat16,
    /// TFloat32 or Float32 one) as f32, exactly.
    pub fn to_f32(&self) -> Vec<f32> {
        // Packed sub-byte floats (from KernelArg::buffer of them).
        if let KernelArg::Bytes { elem, data } = self
            && elem.is_sub_byte()
            && let Some(f) = elem.float_format()
        {
            let bits = elem.bits();
            return (0..data.len() * 8 / bits)
                .map(|i| {
                    let x = (0..bits).fold(0u32, |acc, b| {
                        let at = i * bits + b;
                        acc | ((data[at / 8] >> (at % 8) & 1) as u32) << b
                    });
                    crate::numeric::decode(f, x) as f32
                })
                .collect();
        }
        let Some((t, _)) = self.buffer_type() else { panic!("argument is {}, not a buffer", self.describe()) };
        let f = t.float_format().filter(|_| t.bits() <= 32);
        let f = f.unwrap_or_else(|| {
            panic!("to_f32: argument is {}, not a float buffer of 32 bits or fewer", self.describe())
        });
        self.to_bytes()
            .chunks_exact(t.bytes())
            .map(|c| {
                let bits = c.iter().rev().fold(0u32, |acc, &b| (acc << 8) | b as u32);
                crate::numeric::decode(f, bits) as f32
            })
            .collect()
    }

    /// An i32 buffer.
    pub fn i32(v: Vec<i32>) -> KernelArg {
        KernelArg::I32(v)
    }

    /// A buffer of any element type. Sub-byte values (Int4, FP4, FP6) are
    /// packed as sub-byte tensors store them: element 0 in the low bits of
    /// byte 0, `ceil(n * bits / 8)` bytes in all; the buffer is then a
    /// byte buffer, as the kernel declares it.
    pub fn buffer<T: Numeric>(values: &[T]) -> KernelArg {
        if T::DTYPE.is_word_packed() {
            let bits = T::DTYPE.bits();
            let mut data = vec![0u8; T::DTYPE.storage_bytes(values.len())];
            for (i, v) in values.iter().enumerate() {
                let x = v.raw_bits() & ((1 << bits) - 1);
                for b in 0..bits {
                    let at = i * bits + b;
                    data[at / 8] |= ((x >> b & 1) as u8) << (at % 8);
                }
            }
            return KernelArg::Bytes { elem: T::DTYPE, data };
        }
        let size = T::DTYPE.bytes();
        let data = values.iter().flat_map(|v| v.raw_bits128().to_le_bytes().into_iter().take(size)).collect();
        KernelArg::Bytes { elem: T::DTYPE, data }
    }

    /// A scalar of any type.
    pub fn scalar<T: Numeric>(value: T) -> KernelArg {
        if T::DTYPE.bits() > 64 {
            return KernelArg::Scalar128 { ty: T::DTYPE, bits: value.raw_bits128() };
        }
        KernelArg::Scalar { ty: T::DTYPE, bits: value.raw_bits() }
    }

    /// The contents of a buffer as `T`, which must be its element type. A
    /// sub-byte `T` reads a byte buffer (or one made by [`KernelArg::buffer`]
    /// of `T` values) as packed values, element 0 in the low bits of byte 0:
    /// as many whole values as its bytes hold.
    pub fn to_vec<T: Numeric>(&self) -> Vec<T> {
        if T::DTYPE.is_word_packed() {
            match self.buffer_type() {
                Some((DType::U8, _)) => {}
                _ => panic!("argument is {}, not a byte buffer holding {} values", self.describe(), T::DTYPE),
            }
            let (data, bits) = (self.to_bytes(), T::DTYPE.bits());
            return (0..data.len() * 8 / bits)
                .map(|i| {
                    let x = (0..bits).fold(0u64, |acc, b| {
                        let at = i * bits + b;
                        acc | ((data[at / 8] >> (at % 8) & 1) as u64) << b
                    });
                    T::from_bits(x)
                })
                .collect();
        }
        match self.buffer_type() {
            Some((t, _)) if t == T::DTYPE => {}
            _ => panic!("argument is {}, not a {} buffer", self.describe(), T::DTYPE),
        }
        self.to_bytes()
            .chunks_exact(T::DTYPE.bytes())
            .map(|c| T::from_bits128(c.iter().rev().fold(0u128, |acc, &b| (acc << 8) | b as u128)))
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
            KernelArg::Bytes { elem, data } if elem.is_word_packed() => {
                format!("a byte buffer of {} packed {elem} values ({} bytes)", data.len() * 8 / elem.bits(), data.len())
            }
            KernelArg::Bytes { elem, data } => format!("a {elem} buffer of {}", data.len() / elem.bytes()),
            KernelArg::Scalar { ty, .. } | KernelArg::Scalar128 { ty, .. } => format!("a {ty} scalar"),
        }
    }

    pub(crate) fn buffer_type(&self) -> Option<(DType, usize)> {
        Some(match self {
            KernelArg::F32(v) => (DType::F32, v.len()),
            KernelArg::F16(v) => (DType::F16, v.len()),
            KernelArg::I32(v) => (DType::I32, v.len()),
            KernelArg::Bool(v) => (DType::Bool, v.len()),
            // Packed sub-byte values are a byte buffer, as kernels declare it.
            KernelArg::Bytes { elem, data } if elem.is_word_packed() => (DType::U8, data.len()),
            KernelArg::Bytes { elem, data } => (*elem, data.len() / elem.bytes()),
            _ => return None,
        })
    }

    pub(crate) fn scalar_bits(&self) -> Option<(DType, u128)> {
        Some(match self {
            KernelArg::ScalarF32(x) => (DType::F32, x.to_bits() as u128),
            KernelArg::ScalarF16(x) => (DType::F16, x.0 as u128),
            KernelArg::ScalarI32(x) => (DType::I32, *x as u32 as u128),
            KernelArg::ScalarBool(x) => (DType::Bool, *x as u128),
            KernelArg::Scalar { ty, bits } => (*ty, *bits as u128),
            KernelArg::Scalar128 { ty, bits } => (*ty, *bits),
            _ => return None,
        })
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        match self {
            KernelArg::F32(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            KernelArg::F16(v) => v.iter().flat_map(|x| x.0.to_le_bytes()).collect(),
            KernelArg::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            KernelArg::Bool(v) => v.iter().map(|&x| x as u8).collect(),
            KernelArg::Bytes { data, .. } => data.clone(),
            _ => unreachable!(),
        }
    }

    pub(crate) fn set_from_bytes(&mut self, bytes: &[u8]) {
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

/// What the simulator checks, and how it schedules warps.
#[derive(Clone, Debug)]
pub struct SimOptions {
    /// Report shared-memory accesses from different warps that are not
    /// separated by a barrier.
    pub detect_races: bool,
    /// Report reads of shared memory that was never written.
    pub detect_uninitialized: bool,
    /// Iterations of one `while_carry` loop (by one warp) before the
    /// simulator reports that it does not terminate: a backstop for loops
    /// that do something visible every iteration, which the hang check
    /// ([`SimOptions::max_idle_steps`]) does not catch.
    pub max_while_iterations: u64,
    /// Warp steps (one instruction for one warp) without visible progress
    /// before the simulator reports a hang, if every warp that has not
    /// exited is blocked or spinning by then. Visible progress is a memory
    /// write, an atomic, an arrival at a barrier or mbarrier, a barrier or
    /// mbarrier phase completing, an asynchronous copy landing, a warp
    /// exiting or a `printf`. A warp spins if, since the last visible
    /// progress, it has polled (failed an `mbarrier.test_wait` or a
    /// timed-out `mbarrier.try_wait`, or made an ordered load) and taken a
    /// `while_carry` back-edge or failed such a wait; a loop that only
    /// computes in registers is left to `max_while_iterations`.
    pub max_idle_steps: u64,
    /// The order in which the simulator runs warps.
    pub schedule: Schedule,
    /// How many clusters run together. With the default, one, the clusters
    /// of a launch run one after another; with more (or the whole grid), up
    /// to that many run together and the rest wait for one to finish, as on
    /// a GPU, so cooperative grids and persistent kernels whose clusters
    /// wait for each other through global memory can run. Vector clocks
    /// stay per cluster: global memory is not race-checked, and flags and
    /// atomics between clusters order nothing for the checker.
    pub resident_clusters: ResidentClusters,
    /// When asynchronous operations (`cp.async`, bulk copies) complete.
    pub async_completion: AsyncCompletion,
    /// Check device assertions ([`testing::assert_`](crate::testing::assert_)): a failing
    /// one stops the run with a [`SimError`] naming its line, block and thread. On by
    /// default, since the simulator is the debugging tool; device code includes
    /// assertions only when [`CompileOptions::enable_assertions`](crate::CompileOptions::enable_assertions)
    /// asks. Off, a failing assertion is a warning in [`SimReport::diagnostics`].
    pub assertions: bool,
    /// Which shape each cluster gets in a launch with a fallback cluster
    /// ([`LaunchConfig::fallback_cluster`]).
    pub cluster_shapes: ClusterShapes,
    /// Cluster launch control (roadmap part m10-clc-scheduling): `None`, the
    /// default, keeps every cluster that has not started cancellable, so a
    /// `clusterlaunchcontrol.try_cancel` cancels the next one in launch
    /// order. With a seed, before each query a generator seeded with it
    /// launches some of the clusters at the front of the queue (they start
    /// when a resident slot frees up and can no longer be cancelled), so
    /// that which clusters are left to cancel changes from run to run.
    pub clc_launch_seed: Option<u64>,
    /// Roadmap part m11-prims-mma-forms: report a `tcgen05.mma` that reuses
    /// an operand from a collector buffer (`USE`, `LASTUSE`) after the
    /// shared memory it was filled from changed. On by default: the PTX ISA
    /// lets the hardware reload a collected operand from shared memory, so
    /// such a kernel's result is undefined. Off, the simulator models a
    /// collector that always holds the filled tile.
    pub collector_source_check: bool,
}

/// The shapes of a launch's clusters when it has a preferred and a fallback shape
/// ([`SimOptions::cluster_shapes`]); the hardware may choose either for each cluster, so a
/// kernel must be correct under all three.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterShapes {
    /// Every cluster at the preferred shape.
    #[default]
    Preferred,
    /// Every cluster at the fallback shape.
    Fallback,
    /// Each preferred-sized tile of the grid is one preferred cluster or split into
    /// fallback clusters, as a generator seeded with this decides. The preferred clusters
    /// run first, then the fallback ones (in launch order within each).
    Mixed { seed: u64 },
}

/// When the simulator completes asynchronous operations
/// ([`SimOptions::async_completion`]). Completing one lands its data,
/// publishes it to whoever its completion is ordered before (the warps
/// that wait for its group or mbarrier phase) and fires what depends on it
/// (a `cp.async.mbarrier.arrive`'s arrival, which can complete a phase).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AsyncCompletion {
    /// As late as possible: an operation completes when something requires
    /// it (a wait on its group, a wait on its mbarrier once the phase's
    /// arrivals are in), so every premature access to its data is reported.
    #[default]
    Late,
    /// At issue. A correct kernel must also work this way: it catches
    /// kernels that rely on an operation still being in flight (an
    /// `expect_tx` issued after its copies, a phase counted on to stay open).
    Early,
    /// At random scheduling points ([`Schedule::Random`] lists them), from
    /// a deterministic generator seeded with this: before each one, an
    /// operation in flight may complete. Waits still complete what they
    /// require, but a failed `mbarrier_test_wait` no longer forces the
    /// arrivals and copies its phase misses, so a kernel that polls once
    /// and reads anyway is caught.
    Random(u64),
}

/// How many clusters of a launch the simulator runs together
/// ([`SimOptions::resident_clusters`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentClusters {
    /// Up to this many (at least one) at a time, in launch order.
    Count(u32),
    /// Every cluster of the grid at once (a cooperative launch).
    Grid,
}

impl Default for ResidentClusters {
    fn default() -> Self {
        ResidentClusters::Count(1)
    }
}

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions {
            detect_races: true,
            detect_uninitialized: true,
            max_while_iterations: 1 << 20,
            max_idle_steps: 1 << 20,
            schedule: Schedule::default(),
            resident_clusters: ResidentClusters::default(),
            async_completion: AsyncCompletion::default(),
            assertions: true,
            cluster_shapes: ClusterShapes::default(),
            clc_launch_seed: None,
            collector_source_check: true,
        }
    }
}

/// How the simulator interleaves warps (and the convergence groups of a
/// divergent warp). Every schedule is deterministic: the same kernel,
/// arguments and seed run the same way each time. A correct kernel gives
/// the same results under each of them ([`Kernel::simulate_all_orders`],
/// [`Kernel::simulate_schedules`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Schedule {
    /// Each round, every warp that can run executes one instruction, in the
    /// order of (block rank in the cluster, warp). Warps that run the same
    /// code stay together, as if the block ran in lockstep.
    #[default]
    RoundRobin,
    /// One warp runs until it blocks (at a barrier, a cluster barrier or an
    /// `mbarrier.try_wait`), yields or exits; then the next warp that can
    /// run (in the same order, after it) takes over. This is the most
    /// sequential interleaving, with the fewest switches. A warp yields at
    /// an `mbarrier.test_wait` that fails and at every `while_carry`
    /// back-edge, so spin loops let the other warps run.
    RunToBlock,
    /// A seeded random interleaving: at every scheduling point the next
    /// group to run (a warp, or a convergence group of a divergent warp) is
    /// drawn from those that can run, by a generator seeded with `seed`
    /// (SplitMix64). The scheduling points are every shared or global
    /// memory access, atomic, barrier, mbarrier and cluster-barrier
    /// operation, fence, asynchronous issue and wait, inline PTX or MLIR
    /// statement and `printf`, and every `while_carry` back-edge;
    /// instructions in between only touch registers, so where they run
    /// changes nothing. With the same seed, a blocking `mbarrier_try_wait`
    /// sometimes returns false at once (PTX's time limit), so a kernel that
    /// does not retry is caught. A failure records the seed and the step
    /// count ([`SimError::seed`]); simulating again with the same seed
    /// replays it exactly.
    Random { seed: u64 },
}

impl Schedule {
    /// The built-in deterministic schedules (not [`Schedule::Random`], which
    /// takes a seed).
    pub const ALL: [Schedule; 2] = [Schedule::RoundRobin, Schedule::RunToBlock];
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
    /// Non-fatal findings: failed device assertions when
    /// [`SimOptions::assertions`] is off (one per assertion and block).
    pub diagnostics: Vec<crate::Diagnostic>,
    /// The seed of a [`Schedule::Random`] run.
    pub seed: Option<u64>,
    /// Scheduler steps (one instruction for a warp, or for the warps a
    /// round-robin step runs together).
    pub steps: u64,
    /// TMA tile loads issued (`cp.async.bulk.tensor`), one per issuing
    /// thread.
    pub tma_instructions: u64,
    /// The bytes those loads wrote to shared memory (whole boxes, filled
    /// out-of-bounds elements included, as their transaction bytes count;
    /// a multicast load's in every CTA it writes).
    pub tma_bytes: u64,
    // ---- m7-tma-store and m7-cluster-copies ----
    /// TMA tile stores and reduce stores issued, one per issuing thread.
    pub tma_stores: u64,
    /// The bytes of those stores' boxes (clipped elements included).
    pub tma_store_bytes: u64,
    /// L2 prefetches of TMA boxes and bulk ranges (`cp.async.bulk.prefetch`).
    pub prefetches: u64,
    /// Descriptor prefetches (`prefetch.tensormap`).
    pub tensormap_prefetches: u64,
    /// Copies issued with an L2 cache policy (`.L2::cache_hint`).
    pub cache_hints: u64,
    /// Asynchronous copies (multicast TMA and bulk loads, peer bulk copies, `st.async`)
    /// into another CTA's shared memory, one per destination CTA.
    pub cluster_copies: u64,
    // ---- m10-umma-f16 ----
    /// `tcgen05.mma` instructions issued (one per warp-level `gemm` MMA).
    pub tcgen05_mma_instructions: u64,
    // ---- m10-clc-scheduling ----
    /// Clusters cancelled by `clusterlaunchcontrol.try_cancel` (which never
    /// started).
    pub clc_cancelled: u64,
    // ---- m11-prims-nvvm-layer ----
    /// Warps that executed `exit` (`prims::exit`).
    pub exits: u64,
    /// `nanosleep`s executed, one per warp (each yields the warp).
    pub nanosleeps: u64,
    /// The markers the simulator skips (`pmevent`, `brkpt`, PTX comments),
    /// one per warp that executed one.
    pub markers: u64,
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
        if self.tma_instructions > 0 {
            writeln!(f, "TMA loads:           {} ({} bytes)", self.tma_instructions, self.tma_bytes)?;
        }
        if self.tma_stores > 0 {
            writeln!(f, "TMA stores:          {} ({} bytes)", self.tma_stores, self.tma_store_bytes)?;
        }
        if self.prefetches + self.tensormap_prefetches > 0 {
            writeln!(f, "prefetches:          {} (+{} descriptors)", self.prefetches, self.tensormap_prefetches)?;
        }
        if self.cluster_copies > 0 {
            writeln!(f, "copies to peer CTAs: {}", self.cluster_copies)?;
        }
        if self.tcgen05_mma_instructions > 0 {
            writeln!(f, "tcgen05.mma:         {}", self.tcgen05_mma_instructions)?;
        }
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
    /// The seed of the [`Schedule::Random`] run that failed: simulating
    /// again with the same options replays the failure exactly.
    pub seed: Option<u64>,
    /// Scheduler steps taken before the failure ([`SimReport::steps`]).
    pub steps: u64,
}

impl SimError {
    /// An error with no location.
    pub(crate) fn plain(message: String) -> SimError {
        SimError { message, loc: None, block: [0; 3], thread: None, seed: None, steps: 0 }
    }
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
        write!(f, ": {}", self.message)?;
        if let Some(seed) = self.seed {
            write!(
                f,
                "\n(under Schedule::Random {{ seed: {seed} }}, after {} steps: simulate again with the same \
                 SimOptions, schedule and async_completion alike, to replay it)",
                self.steps
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for SimError {}

/// Check a launch configuration and arguments against a kernel's
/// signature (shared by the simulator and the CUDA launcher), and resolve its shared memory.
pub(crate) fn check_launch(
    sig: &crate::KernelSignature,
    cfg: LaunchConfig,
    args: &[KernelArg],
) -> Result<crate::signature::SmemPlan, String> {
    sig.check_args(args)?;
    sig.check_config(&cfg, crate::validate::MAX_SHARED_BYTES, None)
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
        let mut module = self.sim_module();
        module.simulate_with(cfg, args, opts)
    }

    /// [`Kernel::simulate_with`] under every built-in [`Schedule`], each with
    /// late and early completion of asynchronous operations
    /// ([`AsyncCompletion`]), `opts`'s own first; all must succeed and give
    /// the same results: the
    /// same output buffers and the same `printf` lines (in any order). A
    /// kernel whose results depend on how its warps interleave (a race the
    /// checks miss, or a reduction whose rounding depends on the order) is
    /// reported. Kernels whose results may legitimately differ (an atomic
    /// counter handing out output slots, a floating-point atomic sum) should
    /// use [`Kernel::simulate_with`] instead. Returns the report of the run
    /// with `opts`.
    pub fn simulate_all_orders(
        &self,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        opts: &SimOptions,
    ) -> Result<SimReport, SimError> {
        self.simulate_schedules(cfg, args, opts, &[])
    }

    /// [`Kernel::simulate_all_orders`], then [`Schedule::Random`] with
    /// [`AsyncCompletion::Random`] for each of `seeds` (the same seed for
    /// both): every run must succeed and give the same output buffers and
    /// `printf` lines. A difference is reported with the first element that
    /// differs and the two schedules (`` `out`[17] is 3 under
    /// Schedule::RoundRobin with AsyncCompletion::Late and 5 under
    /// Schedule::Random { seed: 4 } ``), which catches atomics and flag
    /// protocols whose result depends on timing; a failing run names its
    /// seed, which replays it. Returns the report of the run with `opts`.
    pub fn simulate_schedules(
        &self,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        opts: &SimOptions,
        seeds: &[u64],
    ) -> Result<SimReport, SimError> {
        let original = args.to_vec();
        let report = self.simulate_with(cfg, args, opts)?;
        let mut printed = report.printed.clone();
        printed.sort();
        let name = |o: &SimOptions| match o.schedule {
            Schedule::Random { seed } if o.async_completion == AsyncCompletion::Random(seed) => {
                format!("Schedule::Random {{ seed: {seed} }}")
            }
            s => format!("Schedule::{s:?} with AsyncCompletion::{:?}", o.async_completion),
        };
        let first = name(opts);
        let mut orders: Vec<SimOptions> = Schedule::ALL
            .into_iter()
            .flat_map(|s| [AsyncCompletion::Late, AsyncCompletion::Early].map(|c| (s, c)))
            .filter(|&o| o != (opts.schedule, opts.async_completion))
            .map(|(schedule, async_completion)| SimOptions { schedule, async_completion, ..opts.clone() })
            .collect();
        orders.extend(seeds.iter().map(|&seed| SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..opts.clone()
        }));
        for other_opts in orders {
            let order = name(&other_opts);
            let mut other = original.clone();
            let other_report = self
                .simulate_with(cfg, &mut other, &other_opts)
                .map_err(|e| SimError { message: format!("under {order}: {}", e.message), ..e })?;
            let bytes = |a: &KernelArg| a.buffer_type().map(|_| a.to_bytes());
            if let Some(i) = (0..args.len()).find(|&i| bytes(&args[i]) != bytes(&other[i])) {
                let (elem, _) = args[i].buffer_type().expect("a buffer");
                let (a, b) = (args[i].to_bytes(), other[i].to_bytes());
                let size = elem.bytes();
                let at = a.chunks(size).zip(b.chunks(size)).position(|(x, y)| x != y).unwrap_or(0);
                let value = |v: &[u8]| element(elem, &v[at * size..(at + 1) * size]);
                let buffer = self.signature().params[i].name.clone();
                return Err(SimError::plain(format!(
                    "`{buffer}`[{at}] (argument {i}) is {} under {first} and {} under {order}: the kernel's results \
                     depend on how its warps interleave",
                    value(&a),
                    value(&b)
                )));
            }
            let mut other_printed = other_report.printed;
            other_printed.sort();
            if other_printed != printed {
                return Err(SimError::plain(format!("the printf output differs between {first} and {order}")));
            }
        }
        Ok(report)
    }
}

/// An element of a buffer (little-endian `bytes`) as a report shows it.
fn element(ty: DType, bytes: &[u8]) -> String {
    let bits = bytes.iter().rev().fold(0u64, |acc, &b| (acc << 8) | b as u64);
    match ty {
        DType::F32 => format!("{}", f32::from_bits(bits as u32)),
        DType::F64 => format!("{}", f64::from_bits(bits)),
        DType::F16 => format!("{}", crate::value::f16_bits_to_f32(bits as u16)),
        DType::BF16 => format!("{}", crate::value::bf16_bits_to_f32(bits as u16)),
        DType::Bool => format!("{}", bits != 0),
        t if t.is_signed() => {
            let shift = 64 - t.bits();
            format!("{}", ((bits << shift) as i64) >> shift)
        }
        _ => format!("{bits}"),
    }
}

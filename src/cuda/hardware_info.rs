//! Device queries for sizing persistent grids (CuTe DSL's
//! `cutlass.utils.HardwareInfo`), and the cluster occupancy of a real kernel
//! ([`CudaFunction::max_active_clusters`]).

use std::ffi::c_int;

use super::{
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, CudaContext, CudaError, CudaFunction, CudaStream, DeviceAttribute,
    FuncAttribute, LaunchAttribute, LaunchExConfig, check, err,
};

/// An empty kernel to query occupancy with (CuTe DSL compiles one with its
/// own pipeline; a tiny PTX module needs no `mlir-opt` at run time).
const EMPTY_KERNEL_PTX: &str = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry cute_rs_hardware_info_empty()
{
\tret;
}
";
const EMPTY_KERNEL: &str = "cute_rs_hardware_info_empty";

/// The first driver with `cuOccupancyMaxActiveClusters`, CUDA 11.8.
const MIN_DRIVER: u32 = 11080;

/// The device queries CuTe DSL's kernels size persistent grids with (CuTe
/// DSL's `cutlass.utils.HardwareInfo`): how many clusters of a size can be
/// resident at once, the SM count and the L2 size. It holds the device's
/// primary context ([`context`](Self::context)).
pub struct HardwareInfo {
    ctx: CudaContext,
    device_id: usize,
    driver_version: u32,
}

impl HardwareInfo {
    /// The queries of device `device_id`. Fails without a driver, and for a
    /// device id the driver does not have.
    pub fn new(device_id: usize) -> Result<HardwareInfo, CudaError> {
        let count = CudaContext::device_count()?;
        if device_id >= count {
            return Err(err(format!(
                "HardwareInfo: device {device_id} is out of range: the driver reports {count} device(s)"
            )));
        }
        let ctx = CudaContext::new(device_id)?;
        let driver_version = ctx.driver_version()?;
        Ok(HardwareInfo { ctx, device_id, driver_version })
    }

    /// The device's ordinal.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// The device's primary context, to load and launch kernels with.
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// The most clusters of `cluster_size` CTAs (1 to 32) that can be
    /// resident on the device at once, each CTA as large as a block can be
    /// (all the shared memory a block may opt in to), as CuTe DSL's
    /// `get_max_active_clusters` measures it with an empty kernel. `stream`
    /// lets the driver narrow the answer to the SMs the stream may use (a
    /// green context's partition).
    ///
    /// Clusters need sm_90: on an older device a cluster of 1 is a CTA, and
    /// the answer is the CTAs per SM times the SM count; larger clusters are
    /// an error. Fails for a cluster size outside 1 to 32 and on drivers
    /// older than CUDA 11.8.
    pub fn get_max_active_clusters(
        &self,
        cluster_size: u32,
        stream: Option<&CudaStream<'_>>,
    ) -> Result<u32, CudaError> {
        if self.driver_version < MIN_DRIVER {
            return Err(err(format!(
                "HardwareInfo::get_max_active_clusters needs a CUDA 11.8 driver or newer, not {}.{}",
                self.driver_version / 1000,
                self.driver_version % 1000 / 10
            )));
        }
        if !(1..=32).contains(&cluster_size) {
            return Err(err(format!(
                "HardwareInfo::get_max_active_clusters: a cluster has 1 to 32 CTAs, not {cluster_size}"
            )));
        }
        let function = self.ctx.load_ptx(EMPTY_KERNEL_PTX, EMPTY_KERNEL)?;
        let optin = self.ctx.device_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)?;
        function.set_attribute(FuncAttribute::MaxDynamicSharedSizeBytes, optin)?;
        let smem = function.available_dynamic_smem_per_block(1, 1)?;
        let per_sm = function.max_active_blocks_per_multiprocessor(1, smem)?;
        if self.ctx.compute_capability().0 < 9 {
            return self.without_clusters(cluster_size, per_sm);
        }
        function.set_attribute(FuncAttribute::NonPortableClusterSizeAllowed, 1)?;
        function.query_max_active_clusters(
            [cluster_size, per_sm.max(1), 1],
            [128, 1, 1],
            smem,
            [cluster_size, 1, 1],
            stream,
        )
    }

    /// The answer without clusters: CTAs per SM times SMs for a cluster of
    /// one, an error for more.
    fn without_clusters(&self, cluster_size: u32, per_sm: u32) -> Result<u32, CudaError> {
        if cluster_size != 1 {
            let (major, minor) = self.ctx.compute_capability();
            return Err(err(format!(
                "HardwareInfo::get_max_active_clusters: clusters of {cluster_size} CTAs need thread block clusters \
                 (sm_90 or newer), and {} is sm_{major}{minor}; only a cluster size of 1 can be asked for",
                self.ctx.name()
            )));
        }
        Ok(per_sm * self.get_device_multiprocessor_count()?)
    }

    /// The L2 cache's size in bytes (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`).
    pub fn get_l2_cache_size_in_bytes(&self) -> Result<u64, CudaError> {
        Ok(self.ctx.device_attribute(DeviceAttribute::L2CacheSize)?.max(0) as u64)
    }

    /// The number of SMs (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`).
    pub fn get_device_multiprocessor_count(&self) -> Result<u32, CudaError> {
        Ok(self.ctx.device_attribute(DeviceAttribute::MultiprocessorCount)?.max(0) as u32)
    }
}

impl CudaFunction<'_> {
    /// The dynamic shared memory each of `num_blocks` blocks of `block_size`
    /// threads can have on one SM (`cuOccupancyAvailableDynamicSMemPerBlock`).
    pub fn available_dynamic_smem_per_block(&self, num_blocks: u32, block_size: u32) -> Result<usize, CudaError> {
        let query = self
            .ctx
            .api
            .occupancy_available_dynamic_smem
            .ok_or_else(|| err("the CUDA driver has no cuOccupancyAvailableDynamicSMemPerBlock (CUDA 11.0)"))?;
        self.ctx.make_current()?;
        let mut bytes = 0usize;
        // SAFETY: a valid out-pointer and a loaded function.
        let r = unsafe { query(&mut bytes, self.function, num_blocks as c_int, block_size as c_int) };
        check(&self.ctx.api, r, "cuOccupancyAvailableDynamicSMemPerBlock")?;
        Ok(bytes)
    }

    /// The most clusters of `cluster` CTAs (each `block` threads with
    /// `dynamic_smem` bytes of dynamic shared memory) of this kernel that
    /// can be resident at once (`cuOccupancyMaxActiveClusters`), on `stream`
    /// (the legacy stream for `None`): the grid size of a persistent launch
    /// of the kernel itself. Before sm_90 a cluster of one CTA is a block,
    /// and the answer is the blocks per SM times the SM count; larger
    /// clusters are an error there.
    pub fn max_active_clusters(
        &self,
        block: [u32; 3],
        dynamic_smem: usize,
        cluster: [u32; 3],
        stream: Option<&CudaStream<'_>>,
    ) -> Result<u32, CudaError> {
        let cluster_size = cluster.iter().product::<u32>();
        if self.ctx.compute_capability().0 < 9 {
            if cluster_size != 1 {
                return Err(err(format!(
                    "CudaFunction::max_active_clusters: clusters of {cluster:?} need sm_90 or newer, and {} is older",
                    self.ctx.name()
                )));
            }
            let per_sm = self.max_active_blocks_per_multiprocessor(block.iter().product(), dynamic_smem)?;
            let sms = self.ctx.device_attribute(DeviceAttribute::MultiprocessorCount)?.max(0) as u32;
            return Ok(per_sm * sms);
        }
        self.query_max_active_clusters(cluster, block, dynamic_smem, cluster, stream)
    }

    /// `cuOccupancyMaxActiveClusters` for a grid of `grid` in clusters of `cluster`.
    fn query_max_active_clusters(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_smem: usize,
        cluster: [u32; 3],
        stream: Option<&CudaStream<'_>>,
    ) -> Result<u32, CudaError> {
        let query =
            self.ctx.api.occupancy_max_active_clusters.ok_or_else(|| {
                err("the CUDA driver has no cuOccupancyMaxActiveClusters (CUDA 11.8): update the driver")
            })?;
        let smem = u32::try_from(dynamic_smem).map_err(|_| err("the dynamic shared memory does not fit in 32 bits"))?;
        let mut attr = LaunchAttribute { id: CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, pad: 0, value: [0; 16] };
        attr.value[..3].copy_from_slice(&cluster);
        let config = LaunchExConfig {
            grid,
            block,
            shared_mem_bytes: smem,
            stream: stream.map_or(std::ptr::null_mut(), |s| s.raw),
            attrs: &mut attr,
            num_attrs: 1,
        };
        self.ctx.make_current()?;
        let mut n: c_int = 0;
        // SAFETY: a valid out-pointer, a loaded function and a config whose attribute array
        // outlives the call.
        let r = unsafe { query(&mut n, self.function, &config) };
        check(&self.ctx.api, r, "cuOccupancyMaxActiveClusters")?;
        Ok(n.max(0) as u32)
    }
}

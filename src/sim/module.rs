//! A simulated module: the state that outlives one launch of a kernel (its module globals),
//! as a loaded CUDA module keeps its `__device__` and `__constant__` variables between
//! launches.

use std::collections::HashSet;

use super::{ClusterShapes, KernelArg, LaunchConfig, Schedule, SimError, SimOptions, SimReport, exec};
use crate::builder::Kernel;
use crate::ir::Param;

/// A kernel's simulated module ([`Kernel::sim_module`]): launches through it share the
/// module globals ([`module_global`](crate::module_global)), which start at their initial
/// values (or zero), and the host reads and writes them between launches.
/// [`Kernel::simulate`] uses a fresh module each time.
pub struct SimModule<'k> {
    kernel: &'k Kernel,
    /// The bytes of each module global, in [`KernelIr::globals`](crate::ir::KernelIr) order.
    globals: Vec<Vec<u8>>,
    /// The multicast parameters' replicas.
    pub(super) multicast: Vec<exec::McGroup>,
}

impl Kernel {
    /// A fresh simulated module of this kernel, its globals at their initial values.
    pub fn sim_module(&self) -> SimModule<'_> {
        let ir = self.ir();
        let globals = ir
            .globals
            .iter()
            .map(|g| {
                let d = ir.buffer(g.buffer);
                g.init.clone().unwrap_or_else(|| vec![0; d.len.unwrap_or(0) * d.elem.bytes()])
            })
            .collect();
        SimModule { kernel: self, globals, multicast: Vec::new() }
    }
}

/// A launch's global-memory footprint, per buffer (PDL model).
pub(super) type Footprint = Vec<Option<Vec<u8>>>;

/// One run of a launch: the cluster shape, and the clusters (by first block) of that shape,
/// or all of them.
type ClusterRun = ([u32; 3], Option<HashSet<[u32; 3]>>);

/// A deterministic SplitMix64 step.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Add `other`'s counts and output to `report`.
fn merge(report: &mut SimReport, other: SimReport) {
    report.blocks += other.blocks;
    report.warp_instructions += other.warp_instructions;
    report.mma_instructions += other.mma_instructions;
    report.tma_instructions += other.tma_instructions;
    report.tma_bytes += other.tma_bytes;
    report.tma_stores += other.tma_stores;
    report.tma_store_bytes += other.tma_store_bytes;
    report.prefetches += other.prefetches;
    report.tensormap_prefetches += other.tensormap_prefetches;
    report.cache_hints += other.cache_hints;
    report.cluster_copies += other.cluster_copies;
    report.tcgen05_mma_instructions += other.tcgen05_mma_instructions;
    report.barriers += other.barriers;
    report.global_bytes_read += other.global_bytes_read;
    report.global_bytes_written += other.global_bytes_written;
    report.global_sectors += other.global_sectors;
    report.global_sectors_ideal += other.global_sectors_ideal;
    report.shared_wavefronts += other.shared_wavefronts;
    report.shared_wavefronts_ideal += other.shared_wavefronts_ideal;
    report.printed.extend(other.printed);
    report.diagnostics.extend(other.diagnostics);
    report.steps += other.steps;
}

impl SimModule<'_> {
    fn global_index(&self, name: &str) -> Result<usize, String> {
        let ir = self.kernel.ir();
        ir.globals
            .iter()
            .position(|g| ir.buffer(g.buffer).name == name)
            .ok_or_else(|| format!("kernel `{}` has no module global `{name}`", ir.name))
    }

    /// Module global `name`'s current bytes, as a buffer of its element type.
    pub fn read_global(&self, name: &str) -> Result<KernelArg, String> {
        let k = self.global_index(name)?;
        let ir = self.kernel.ir();
        let elem = ir.buffer(ir.globals[k].buffer).elem;
        Ok(KernelArg::Bytes { elem, data: self.globals[k].clone() })
    }

    /// Overwrite module global `name` with `value` (a buffer of its element type and
    /// length), as `CudaFunction::global` does on a GPU. Read-only globals refuse.
    pub fn write_global(&mut self, name: &str, value: &KernelArg) -> Result<(), String> {
        let k = self.global_index(name)?;
        let ir = self.kernel.ir();
        let g = &ir.globals[k];
        let d = ir.buffer(g.buffer);
        if g.readonly {
            return Err(format!(
                "module global `{name}` is read-only: the kernel may have folded its value, so a write would go unseen"
            ));
        }
        match value.buffer_type() {
            Some((t, n)) if t == d.elem && Some(n) == d.len => {}
            _ => {
                return Err(format!(
                    "module global `{name}` holds {} {} elements, got {}",
                    d.len.unwrap_or(0),
                    d.elem,
                    value.describe()
                ));
            }
        }
        self.globals[k] = value.to_bytes();
        Ok(())
    }

    /// Run a launch with the default options.
    pub fn simulate(&mut self, cfg: LaunchConfig, args: &mut [KernelArg]) -> Result<SimReport, SimError> {
        self.simulate_with(cfg, args, &SimOptions::default())
    }

    /// Run a launch: [`Kernel::simulate_with`], with this module's globals.
    pub fn simulate_with(
        &mut self,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        opts: &SimOptions,
    ) -> Result<SimReport, SimError> {
        self.run(cfg, args, opts, None).map(|(report, _)| report)
    }

    /// Run a launch. With `pdl`, record the launch's global-memory footprint (returned per
    /// buffer) and check its pre-wait accesses against the primary's, if any. A cooperative
    /// launch ([`LaunchConfig::cooperative`]) runs the whole grid resident.
    pub(super) fn run(
        &mut self,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        opts: &SimOptions,
        pdl: Option<Option<exec::PdlPrimary>>,
    ) -> Result<(SimReport, Footprint), SimError> {
        // A kernel with a device workspace takes it last.
        crate::experimental::host_runtime::check_launch_args(self.kernel, &cfg, args).map_err(SimError::plain)?;
        let cooperative;
        let opts = if cfg.cooperative {
            cooperative = SimOptions { resident_clusters: super::ResidentClusters::Grid, ..opts.clone() };
            &cooperative
        } else {
            opts
        };
        let mut pdl = pdl.map(|p| p.map(std::rc::Rc::new));
        let mut footprint: Vec<Option<Vec<u8>>> = Vec::new();
        let kernel = self.kernel;
        let ir = kernel.ir();
        let sig = kernel.signature();
        let plan = super::check_launch(&sig, cfg, args).map_err(SimError::plain)?;
        let mut cfg = LaunchConfig { dynamic_smem_bytes: plan.region_bytes as u32, smem_bytes: None, ..cfg };
        // The cluster shapes: one run per shape, the clusters of each named by their first block.
        let mut runs: Vec<ClusterRun> = Vec::new();
        match (cfg.fallback_cluster, opts.cluster_shapes) {
            (None, _) | (Some(_), ClusterShapes::Preferred) => runs.push((cfg.cluster, None)),
            (Some(f), ClusterShapes::Fallback) => runs.push((f, None)),
            (Some(f), ClusterShapes::Mixed { seed }) => {
                let p = cfg.cluster;
                let mut state = seed;
                let (mut preferred, mut fallback) = (HashSet::new(), HashSet::new());
                for z in (0..cfg.grid[2]).step_by(p[2] as usize) {
                    for y in (0..cfg.grid[1]).step_by(p[1] as usize) {
                        for x in (0..cfg.grid[0]).step_by(p[0] as usize) {
                            if splitmix(&mut state) & 1 == 0 {
                                preferred.insert([x, y, z]);
                                continue;
                            }
                            for dz in (0..p[2]).step_by(f[2] as usize) {
                                for dy in (0..p[1]).step_by(f[1] as usize) {
                                    for dx in (0..p[0]).step_by(f[0] as usize) {
                                        fallback.insert([x + dx, y + dy, z + dz]);
                                    }
                                }
                            }
                        }
                    }
                }
                if !preferred.is_empty() {
                    runs.push((p, Some(preferred)));
                }
                if !fallback.is_empty() {
                    runs.push((f, Some(fallback)));
                }
            }
        }
        cfg.fallback_cluster = None;

        let mut globals = vec![None; ir.buffers.len()];
        let mut scalars = Vec::new();
        for (p, a) in ir.args.iter().zip(args.iter()) {
            match p {
                Param::Buffer(b) | Param::ByValue { buffer: b, .. } => globals[b.0 as usize] = Some(a.to_bytes()),
                Param::Scalar { value, .. } => scalars.push((*value, a.scalar_bits().unwrap().1)),
            }
        }
        for (g, bytes) in ir.globals.iter().zip(&self.globals) {
            globals[g.buffer.0 as usize] = Some(bytes.clone());
        }
        // Hidden tensor maps are bound per run (bind_tensor_maps), from the address map.
        for t in ir.tensor_maps.iter().filter(|t| t.source.is_some()) {
            globals[t.buffer.0 as usize] = Some(vec![0; crate::tma::TENSOR_MAP_BYTES]);
        }
        let seed = match opts.schedule {
            Schedule::Random { seed } => Some(seed),
            _ => None,
        };
        let mut report: Option<SimReport> = None;
        let mut multicast = std::mem::take(&mut self.multicast);
        for (shape, filter) in runs {
            let run_cfg = LaunchConfig { cluster: shape, ..cfg };
            let mut machine = exec::Machine::new(ir, opts, run_cfg, globals, scalars.clone());
            machine.set_multicast(std::mem::take(&mut multicast));
            machine.check_param_alignment().map_err(SimError::plain)?;
            machine.bind_tensor_maps(&sig).map_err(SimError::plain)?;
            if let Some(f) = filter {
                machine.set_cluster_filter(f);
            }
            if let Some(primary) = pdl.as_mut() {
                machine.set_pdl(primary.clone());
            }
            if let Err(e) = machine.run() {
                return Err(SimError { seed, steps: machine.steps(), ..e });
            }
            if pdl.is_some() {
                let run = machine.take_footprint();
                if footprint.is_empty() {
                    footprint = run;
                } else {
                    for (f, r) in footprint.iter_mut().zip(run) {
                        if let (Some(f), Some(r)) = (f.as_mut(), r) {
                            f.iter_mut().zip(r).for_each(|(a, b)| *a |= b);
                        }
                    }
                }
            }
            multicast = machine.take_multicast();
            let (out, r) = machine.finish();
            globals = out;
            match &mut report {
                None => report = Some(r),
                Some(total) => merge(total, r),
            }
        }
        self.multicast = multicast;
        let mut report = report.expect("a launch has at least one cluster");
        report.seed = seed;
        for (p, a) in ir.args.iter().zip(args.iter_mut()) {
            if let Param::Buffer(b) = p {
                a.set_from_bytes(globals[b.0 as usize].as_ref().unwrap());
            }
        }
        for (g, bytes) in ir.globals.iter().zip(self.globals.iter_mut()) {
            *bytes = globals[g.buffer.0 as usize].take().unwrap_or_default();
        }
        Ok((report, footprint))
    }
}

/// A simulated [`KernelModule`](crate::KernelModule) ([`KernelModule::sim_module`]): its
/// kernels, launched by name, share the module globals, as the kernels of one loaded CUDA
/// module do (a global one kernel writes, the next reads). The globals start at their initial
/// values (or zero).
///
/// [`KernelModule::sim_module`]: crate::KernelModule::sim_module
pub struct SimKernelModule<'m> {
    module: &'m crate::KernelModule,
    /// Every module global's bytes, by name.
    globals: std::collections::BTreeMap<String, Vec<u8>>,
}

impl crate::KernelModule {
    /// A fresh simulated module, its globals at their initial values.
    pub fn sim_module(&self) -> SimKernelModule<'_> {
        let mut globals = std::collections::BTreeMap::new();
        for k in self.kernels() {
            let m = k.sim_module();
            let ir = k.ir();
            for (g, bytes) in ir.globals.iter().zip(m.globals) {
                globals.entry(ir.buffer(g.buffer).name.clone()).or_insert(bytes);
            }
        }
        SimKernelModule { module: self, globals }
    }
}

impl<'m> SimKernelModule<'m> {
    /// The module simulated.
    pub fn module(&self) -> &'m crate::KernelModule {
        self.module
    }

    fn kernel(&self, name: &str) -> Result<&'m Kernel, String> {
        self.module.kernel(name).ok_or_else(|| {
            let names: Vec<&str> = self.module.kernels().iter().map(Kernel::name).collect();
            format!("module `{}` has no kernel `{name}` (it has {})", self.module.name(), names.join(", "))
        })
    }

    /// Kernel `kernel`'s module, holding the shared globals' current bytes.
    fn view(&self, kernel: &'m Kernel) -> SimModule<'m> {
        let ir = kernel.ir();
        let globals = ir.globals.iter().map(|g| self.globals[&ir.buffer(g.buffer).name].clone()).collect();
        SimModule { kernel, globals, multicast: Vec::new() }
    }

    /// Keep what a kernel's module holds now.
    fn keep(&mut self, m: SimModule<'m>) {
        let ir = m.kernel.ir();
        for (g, bytes) in ir.globals.iter().zip(m.globals) {
            self.globals.insert(ir.buffer(g.buffer).name.clone(), bytes);
        }
    }

    /// The kernel that declares global `name`.
    fn declaring(&self, name: &str) -> Result<&'m Kernel, String> {
        self.module
            .kernels()
            .iter()
            .find(|k| k.ir().globals.iter().any(|g| k.ir().buffer(g.buffer).name == name))
            .ok_or_else(|| format!("module `{}` has no module global `{name}`", self.module.name()))
    }

    /// Launch kernel `kernel` with the default options.
    pub fn simulate(&mut self, kernel: &str, cfg: LaunchConfig, args: &mut [KernelArg]) -> Result<SimReport, SimError> {
        self.simulate_with(kernel, cfg, args, &SimOptions::default())
    }

    /// Launch kernel `kernel` ([`SimModule::simulate_with`]) over the module's globals.
    pub fn simulate_with(
        &mut self,
        kernel: &str,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        opts: &SimOptions,
    ) -> Result<SimReport, SimError> {
        let k = self.kernel(kernel).map_err(SimError::plain)?;
        let mut m = self.view(k);
        let out = m.simulate_with(cfg, args, opts);
        self.keep(m);
        out
    }

    /// Module global `name`'s current bytes, as a buffer of its element type.
    pub fn read_global(&self, name: &str) -> Result<KernelArg, String> {
        self.view(self.declaring(name)?).read_global(name)
    }

    /// Overwrite module global `name` (as [`SimModule::write_global`]): every kernel sees it.
    pub fn write_global(&mut self, name: &str, value: &KernelArg) -> Result<(), String> {
        let mut m = self.view(self.declaring(name)?);
        m.write_global(name, value)?;
        self.keep(m);
        Ok(())
    }
}

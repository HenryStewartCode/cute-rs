//! Simulated multicast memory: the replicas a multicast
//! kernel parameter ([`param_tensor_multicast`](crate::utils::distributed::param_tensor_multicast))
//! stands for, one per simulated GPU.
//!
//! The simulator runs one rank's grid. Its launch argument for the multicast parameter (or
//! for the unicast parameter that aliases it, [`MulticastReplicas::alias`]) is the local
//! replica; [`MulticastReplicas`] holds every rank's replica, so that `multimem.ld_reduce`
//! reads them all and `multimem.st` and `multimem.red` write them all. A multicast parameter
//! without replicas is a group of one: the launch argument alone.

use super::{KernelArg, LaunchConfig, SimError, SimOptions, SimReport, exec};
use crate::builder::Kernel;
use crate::ir::Param;

/// The replicas of one multicast parameter for [`Kernel::simulate_multicast`].
#[derive(Clone, Debug)]
pub struct MulticastReplicas {
    /// The multicast parameter's name.
    pub param: String,
    /// Every rank's replica, by rank: buffers of the parameter's element type and length.
    /// After the launch they hold what the kernel left in each, the local one included.
    pub replicas: Vec<KernelArg>,
    /// The rank of the simulated GPU: its replica is the launch argument.
    pub local: usize,
    /// A unicast buffer parameter over the same memory as the local replica (CuTe DSL's
    /// examples pass `flag` and `flag_mc` over one allocation): its launch argument is the
    /// local replica, and the multicast parameter's own argument receives its final contents.
    pub alias: Option<String>,
}

impl MulticastReplicas {
    /// Replicas `replicas` (one per rank) of multicast parameter `param`, the simulated GPU
    /// being rank `local`.
    pub fn new(param: &str, replicas: Vec<KernelArg>, local: usize) -> MulticastReplicas {
        assert!(local < replicas.len(), "MulticastReplicas: rank {local} of {} replicas", replicas.len());
        MulticastReplicas { param: param.to_string(), replicas, local, alias: None }
    }

    /// The same, with unicast parameter `unicast` aliasing the local replica.
    pub fn alias(mut self, unicast: &str) -> MulticastReplicas {
        self.alias = Some(unicast.to_string());
        self
    }
}

impl Kernel {
    /// [`Kernel::simulate_with`] with the replicas of the kernel's multicast parameters: each
    /// group's launch argument (or its alias's) is the local replica, and the other replicas
    /// take part in every multimem instruction. After the launch, `groups` hold every
    /// replica's contents.
    pub fn simulate_multicast(
        &self,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        groups: &mut [MulticastReplicas],
        opts: &SimOptions,
    ) -> Result<SimReport, SimError> {
        let ir = self.ir();
        let param_buffer = |name: &str| -> Result<(usize, crate::ir::BufferId), SimError> {
            ir.args
                .iter()
                .enumerate()
                .find_map(|(k, p)| match p {
                    Param::Buffer(b) if ir.buffer(*b).name == name => Some((k, *b)),
                    _ => None,
                })
                .ok_or_else(|| SimError::plain(format!("kernel `{}` has no buffer parameter `{name}`", ir.name)))
        };
        let mut bound = Vec::new();
        let mut aliases = Vec::new();
        for g in groups.iter() {
            let (k, buffer) = param_buffer(&g.param)?;
            if !ir.is_multicast(buffer) {
                return Err(SimError::plain(format!("parameter `{}` of `{}` is not multicast", g.param, ir.name)));
            }
            if bound.iter().any(|m: &exec::McGroup| m.buffer == buffer) {
                return Err(SimError::plain(format!("multicast parameter `{}` has two groups of replicas", g.param)));
            }
            let alias = match &g.alias {
                Some(name) => {
                    let (a, b) = param_buffer(name)?;
                    if ir.is_multicast(b) {
                        return Err(SimError::plain(format!(
                            "`{name}` aliases the local replica of `{}`, so it must be a unicast parameter",
                            g.param
                        )));
                    }
                    aliases.push((k, a));
                    Some(b)
                }
                None => None,
            };
            let expected = args.get(alias.map_or(k, |_| aliases.last().expect("pushed").1)).map(KernelArg::to_bytes);
            let replicas: Vec<Vec<u8>> = g.replicas.iter().map(KernelArg::to_bytes).collect();
            if let Some(local) = expected
                && replicas.iter().any(|r| r.len() != local.len())
            {
                return Err(SimError::plain(format!(
                    "the replicas of `{}` must each hold {} bytes, as the launch argument does",
                    g.param,
                    local.len()
                )));
            }
            bound.push(exec::McGroup { buffer, local: g.local, alias, replicas });
        }
        let mut module = self.sim_module();
        module.multicast = bound;
        let report = module.simulate_with(cfg, args, opts)?;
        for (g, m) in groups.iter_mut().zip(std::mem::take(&mut module.multicast)) {
            for (arg, bytes) in g.replicas.iter_mut().zip(m.replicas) {
                arg.set_from_bytes(&bytes);
            }
        }
        for (k, a) in aliases {
            let bytes = args[a].to_bytes();
            args[k].set_from_bytes(&bytes);
        }
        Ok(report)
    }
}

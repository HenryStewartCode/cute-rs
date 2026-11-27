// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's cluster launch control,
//! modelled from the PTX ISA's `clusterlaunchcontrol.try_cancel` and
//! `clusterlaunchcontrol.query_cancel`.
//!
//! * **The queue.** The clusters of the grid that have not started wait in
//!   the launch queue (launch order; [`SimOptions::resident_clusters`] says
//!   how many run at once). A `try_cancel` takes the first one that has not
//!   launched, so each is handed out at most once and a cancelled cluster
//!   never starts. With [`SimOptions::clc_launch_seed`], clusters at the
//!   front of the queue launch at random before each query (they start when
//!   a slot frees up and can no longer be cancelled). An empty queue answers
//!   "not cancelled".
//! * **The response** is 16 bytes, opaque in the PTX ISA. The simulator's
//!   encoding: the cancelled cluster's first CTA's x, y and z as
//!   little-endian u32s, then a word of [`CLC_TAG`] with bit 0 set when a
//!   cluster was cancelled (x, y and z are 0 otherwise). `clc_response`
//!   decodes it and reports 16 bytes that are not a response.
//! * **Delivery.** The response is an asynchronous write through the async
//!   proxy that completes 16 bytes of tx on the mbarrier, as a bulk copy
//!   does (the in-flight model): it lands when a waiter sees the phase's
//!   arrivals in (or at random), and reading it earlier is reported. With
//!   `.multicast::cluster::all` it lands at the same offsets in every CTA of
//!   the cluster, counting down each CTA's mbarrier; PTX leaves that
//!   undefined once a CTA of the cluster has exited, which is reported.
//!
//! [`SimOptions::resident_clusters`]: crate::SimOptions::resident_clusters
//! [`SimOptions::clc_launch_seed`]: crate::SimOptions::clc_launch_seed

use super::{Machine, R, splitmix};
use crate::ir::{Inst, Op};

/// The high bits of the last word of the simulator's try_cancel response.
pub(super) const CLC_TAG: u32 = 0xC1C0_0000;

impl<'a> Machine<'a> {
    /// The next cluster a `try_cancel` cancels, as its response: the first
    /// queued cluster that has not launched, removed from the queue.
    fn clc_cancel_next(&mut self) -> [u8; 16] {
        if let Some(rng) = self.clc_rng.as_mut() {
            let launch = (splitmix(rng) % 3) as usize;
            self.clc_launched = (self.clc_launched + launch).min(self.queued.len());
        }
        let mut words = [0u32, 0, 0, CLC_TAG];
        if self.clc_launched < self.queued.len() {
            let blocks = self.queued.remove(self.clc_launched).expect("a queued cluster");
            let first = blocks.iter().min_by_key(|b| (b[2], b[1], b[0])).expect("a cluster has blocks");
            words = [first[0], first[1], first[2], CLC_TAG | 1];
            self.report.clc_cancelled += 1;
        }
        let mut bytes = [0u8; 16];
        for (k, w) in words.iter().enumerate() {
            bytes[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    /// `clusterlaunchcontrol.try_cancel` by the threads of `active`.
    pub(super) fn clc_try_cancel(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        let Op::ClcTryCancel { buffer, index, elem, mbar, mbar_index, multicast } = &inst.op else {
            unreachable!("a try_cancel")
        };
        let what = if *multicast {
            "clusterlaunchcontrol.try_cancel.multicast::cluster::all (issue_clc_query)"
        } else {
            "clusterlaunchcontrol.try_cancel (issue_clc_query)"
        };
        self.progress();
        let size = elem.bytes().max(1);
        for &t in active {
            let byte = self.checked_range(*buffer, self.index(*index, t), 16 / size, size, 16, t, what)?;
            let mbyte = self.checked_range(*mbar, self.index(*mbar_index, t), 1, 8, 8, t, "try_cancel mbarrier")?;
            let own = self.block_rank(t);
            let targets: Vec<usize> = if *multicast {
                let base = self.cluster_base(own);
                (base..base + self.cluster_size).collect()
            } else {
                vec![own]
            };
            for &block in &targets {
                if self.exited_warps(block).len() == self.warps_per_block {
                    return Err(self.err(
                        format!(
                            "{what} by CTA {} after CTA {} of the cluster has exited: the multicast response is \
                             written to every CTA of the cluster, which the PTX ISA leaves undefined once one has \
                             exited (keep every CTA alive, with a cluster barrier before exiting, until the \
                             scheduler has stopped querying)",
                            own % self.cluster_size,
                            block % self.cluster_size
                        ),
                        Some(t),
                    ));
                }
            }
            for &block in &targets {
                self.check_copy_mbarrier((block, mbar.0, mbyte), t, what)?;
            }
            let response = self.clc_cancel_next();
            for &block in &targets {
                let key = (block, mbar.0, mbyte);
                self.issue_to_mbarrier(
                    *buffer,
                    block,
                    vec![(byte, response.to_vec())],
                    Vec::new(),
                    key,
                    t,
                    what,
                    true,
                )?;
            }
        }
        Ok(())
    }

    /// The 16-byte load and `clusterlaunchcontrol.query_cancel` decode of a
    /// response (`clc_response`) by the threads of `active`.
    pub(super) fn clc_query_cancel(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        let Op::ClcQueryCancel { buffer, index, elem, results } = &inst.op else { unreachable!("a query_cancel") };
        let what = "clc_response (clusterlaunchcontrol.query_cancel)";
        let size = elem.bytes().max(1);
        for &t in active {
            let byte = self.checked_range(*buffer, self.index(*index, t), 16 / size, size, 16, t, what)?;
            let lo = self.read(*buffer, byte, 8, t)?;
            let hi = self.read(*buffer, byte + 8, 8, t)?;
            let words = [lo as u32, (lo >> 32) as u32, hi as u32, (hi >> 32) as u32];
            if words[3] & !1 != CLC_TAG {
                return Err(self.err(
                    format!(
                        "{what}: the 16 bytes at byte {byte} of `{}` are not a try_cancel response (in the \
                         simulator's encoding, the last word is {CLC_TAG:#010x} or {:#010x}, not {:#010x}): read the \
                         response only after waiting on the mbarrier its issue_clc_query completes",
                        self.ir.buffer(*buffer).name,
                        CLC_TAG | 1,
                        words[3]
                    ),
                    Some(t),
                ));
            }
            let valid = words[3] & 1;
            for (r, w) in results.iter().zip([words[0], words[1], words[2], valid]) {
                self.set(*r, t, u64::from(w));
            }
        }
        Ok(())
    }
}

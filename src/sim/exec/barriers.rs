// SPDX-License-Identifier: MIT OR Apache-2.0

//! Barriers: block and cluster barriers, rendezvous of warp collectives, mbarrier extensions, setmaxnreg and warp exit.

use super::*;

impl<'a> Machine<'a> {
    /// Group `k`'s turn at `.sync` warp collective `inst` (member mask
    /// `mask`), with independent thread scheduling: the lanes of the mask
    /// may stand in several convergence groups of the warp. The group
    /// blocks until every lane of the mask that has not exited has reached
    /// the instruction, then it executes once for all of them. Lanes of the
    /// mask that no group holds (they wait at a loop's end or a join) can
    /// never come: executing reports them.
    pub(super) fn rendezvous(&mut self, inst: &'a Inst, k: usize, mask: ValueId) -> R {
        let lanes = self.warp_lanes(k).to_vec();
        let base = lanes[0] - self.lane_of(lanes[0]);
        let bits = |m: &Self, lanes: &[usize]| lanes.iter().fold(0u32, |acc, &t| acc | 1 << m.lane_of(t));
        let live = if self.warp_size(base) == 32 { u32::MAX } else { (1u32 << self.warp_size(base)) - 1 };
        let needed = lanes.iter().fold(0u32, |acc, &t| acc | self.get(mask, t) as u32) & live;
        let mine = bits(self, &lanes);
        if needed & !mine == 0 {
            self.run_inst(inst, &[k])?;
            self.advance(k);
            return Ok(());
        }
        let held = self.parked[k].iter().fold(0u32, |acc, g| acc | bits(self, g.lanes()));
        if needed & !mine & !held != 0 {
            self.run_inst(inst, &[k])?;
            self.advance(k);
            return Ok(());
        }
        let partners: Vec<usize> = (0..self.parked[k].len())
            .filter(|&i| {
                let g = &self.parked[k][i];
                matches!(g.status, Status::Blocked(Wait::Rendezvous { inst: at }) if same_collective(at, inst))
                    && bits(self, g.lanes()) & !needed == 0
            })
            .collect();
        let present = partners.iter().fold(mine, |acc, &i| acc | bits(self, self.parked[k][i].lanes()));
        if needed & !present != 0 {
            self.warps[k].status = Status::Blocked(Wait::Rendezvous { inst });
            return Ok(());
        }
        // Lanes at another instruction of the same kind (sm_70 lets lanes
        // meet at different ones) run this one, with their own operands,
        // and take their results back.
        let mut all = lanes;
        let mut moved: Vec<(&'a Inst, Vec<usize>)> = Vec::new();
        for &i in &partners {
            let g = &self.parked[k][i];
            all.extend_from_slice(g.lanes());
            let Status::Blocked(Wait::Rendezvous { inst: at }) = g.status else { unreachable!("a partner") };
            if !std::ptr::eq(at, inst) {
                moved.push((at, g.lanes().to_vec()));
            }
        }
        // The partners' registers of this instruction's operands may hold
        // values they use later (an operand defined before the `if`): they
        // are put back afterwards.
        let mut saved: Vec<(ValueId, usize, u64)> = Vec::new();
        for (at, ts) in &moved {
            for (mine, theirs) in inst.op.operands().into_iter().zip(at.op.operands()) {
                for &t in ts {
                    saved.push((mine, t, self.get(mine, t)));
                }
                for &t in ts {
                    let bits = self.get(theirs, t);
                    self.set(mine, t, bits);
                }
            }
        }
        all.sort_unstable();
        self.inst(inst, &all)?;
        for &(v, t, bits) in saved.iter().rev() {
            self.set(v, t, bits);
        }
        for (at, ts) in &moved {
            if let (Some(mine), Some(theirs)) = (collective_result(inst), collective_result(at)) {
                for &t in ts {
                    let bits = self.get(mine, t);
                    self.set(theirs, t, bits);
                }
            }
        }
        self.advance(k);
        for i in partners {
            let g = &mut self.parked[k][i];
            g.status = Status::Runnable;
            g.groups[0].frames.last_mut().expect("a blocked group has a frame").next += 1;
            g.settled = false;
        }
        Ok(())
    }

    /// The warps of `block` that have exited.
    pub(super) fn exited_warps(&self, block: usize) -> Vec<usize> {
        let w = self.warps_per_block;
        (0..w).filter(|&warp| matches!(self.warps[block * w + warp].status, Status::Exited)).collect()
    }

    /// Threads in the warps `warps` of a block.
    fn threads_in(&self, warps: &[usize]) -> usize {
        warps.iter().map(|&warp| self.warp_size(warp * 32)).sum()
    }

    /// After a warp of `block` exited: if it was the block's last, nothing
    /// asynchronous may still write its shared memory, which is gone (a
    /// bulk copy or `cp.async` in flight, or an arrival that
    /// `cp.async.mbarrier.arrive` owes one of its mbarriers).
    pub(super) fn check_block_exit(&self, block: usize) -> R {
        let w = self.warps_per_block;
        if !(0..w).all(|warp| matches!(self.warps[block * w + warp].status, Status::Exited)) {
            return Ok(());
        }
        self.tmem_check_exit(block)?;
        let cta = format!("CTA {} ({})", block % self.cluster_size, self.block_name(block));
        // Accesses of other CTAs to its shared memory must be
        // ordered before its exit.
        if let Some(log) = self.dsmem_log.get(&block) {
            let mut entries: Vec<(&usize, &(u32, crate::ir::Loc))> = log.iter().collect();
            entries.sort_by_key(|e| *e.0);
            for (&slot, &(clock, loc)) in entries {
                let known = (0..w).map(|warp| self.clocks[block * w + warp][slot]).max().unwrap_or(0);
                if known < clock {
                    return Err(self.err(
                        format!(
                            "{cta} exits while {} may still access its shared memory (the access at {}:{} is not \
                             ordered before the exit): CUDA requires a cluster barrier (cluster_arrive, cluster_wait) \
                             before a CTA whose shared memory other CTAs use exits",
                            self.slot_name(slot, block),
                            loc.file(),
                            loc.line()
                        ),
                        None,
                    ));
                }
            }
        }
        let mut keys: Vec<&MbarrierKey> = self.mbarriers.keys().filter(|key| key.0 == block).collect();
        keys.sort();
        for key in keys {
            let m = &self.mbarriers[key];
            let name = smem_place(self.ir.buffer(BufferId(key.1)), key.2);
            if !m.copies.is_empty() {
                let n = m.copies.len();
                return Err(self.err(
                    format!(
                        "{cta} exits while {n} bulk cop{} reporting to its mbarrier {name} {} still in flight: they \
                         write shared memory that no longer exists (wait for the mbarrier phase before exiting)",
                        if n == 1 { "y" } else { "ies" },
                        if n == 1 { "is" } else { "are" }
                    ),
                    None,
                ));
            }
            let owed = m.deferred.iter().filter(|d| d.copies.iter().any(|&id| self.ops[id].in_flight())).count();
            if owed > 0 {
                return Err(self.err(
                    format!(
                        "{cta} exits while its mbarrier {name} still waits for {owed} cp.async.mbarrier.arrive \
                         arrival(s) whose copies are in flight (wait for the mbarrier phase before exiting)"
                    ),
                    None,
                ));
            }
        }
        for op in self.ops.iter().filter(|op| op.in_flight()) {
            if let Some(w) = op.writes.iter().find(|w| w.block == block) {
                let place = smem_place(self.ir.buffer(w.dst), w.dst_byte);
                return Err(self.err(
                    format!(
                        "{cta} exits while an asynchronous copy is still writing its shared memory ({place}): wait \
                         for the copy (cp_async_wait_group, or its mbarrier) before exiting"
                    ),
                    None,
                ));
            }
            if let Some(r) = op.reads.iter().find(|r| r.block == block) {
                let place = smem_place(self.ir.buffer(r.buffer), r.bytes.start);
                return Err(self.err(
                    format!(
                        "{cta} exits while an asynchronous operation still reads its shared memory ({place}): wait \
                         for the operation before exiting"
                    ),
                    None,
                ));
            }
        }
        Ok(())
    }

    /// After a warp of `block` exited: an all-thread barrier of the block
    /// that every other warp waits at would complete only because warps
    /// exited instead of reaching it, which is reported.
    pub(super) fn check_exit(&self, block: usize) -> R {
        for state in &self.named[block] {
            if state.all_threads && !state.waiting.is_empty() {
                let exited = self.exited_warps(block);
                if state.arrived.len() + exited.len() == state.expected {
                    let site = state.site.expect("an all-thread barrier with waiters has a site");
                    return Err(self.exit_error(site, block, &state.arrived, &exited));
                }
            }
        }
        Ok(())
    }

    /// The report of all-thread barrier `site` of `block`, reached by warps
    /// `arrived`, that would complete only because warps `exited` have
    /// exited.
    fn exit_error(&self, site: &Inst, block: usize, arrived: &[usize], exited: &[usize]) -> SimError {
        let what = if matches!(site.op, Op::Barrier) { "sync_threads()" } else { "barrier" };
        SimError {
            loc: Some(site.loc),
            block: self.blocks[block],
            ..self.err(
                format!(
                    "{what} reached by {} of {} threads: warps {} exited without reaching it (barriers must not be \
                     inside divergent control flow)",
                    self.threads_in(arrived),
                    self.threads_per_block,
                    warp_list(exited)
                ),
                None,
            )
        }
    }

    /// The `mbarrier.try_wait` that `group` stands at: warps for which a
    /// lane's phase is not over block until it is (or until the wait times
    /// out, when nothing else can run); the others execute it.
    pub(super) fn try_wait(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        let mut proceed: Vec<usize> = Vec::new();
        self.loc = Some(inst.loc);
        for &k in group {
            let mut blocker = if self.warps[k].timed_out { None } else { self.wait_blocker(inst, k)? };
            // A seeded schedule sometimes lets the wait time out at once.
            if blocker.is_some()
                && let Some(rng) = self.sched_rng.as_mut()
                && splitmix(rng).is_multiple_of(8)
            {
                blocker = None;
            }
            match blocker {
                Some((key, phase)) => self.warps[k].status = Status::Blocked(Wait::Mbarrier { key, phase, inst }),
                None => proceed.push(k),
            }
        }
        if proceed.is_empty() {
            return Ok(());
        }
        self.run_inst(inst, &proceed)?;
        for k in proceed {
            self.warps[k].timed_out = false;
            self.advance(k);
        }
        Ok(())
    }

    /// Inline PTX holding `barrier.cluster.arrive` and/or
    /// `barrier.cluster.wait` (`inst`), for the warps of `group`: the
    /// cluster barrier's arrive and wait ([`Self::cluster_arrive`],
    /// [`Self::cluster_wait`]), `.relaxed` and `.aligned` as the template
    /// says. Warps that get past it run the instruction's simulation. Woken
    /// at a combined arrive-and-wait, a warp does not arrive again.
    pub(super) fn inline_cluster_barrier(&mut self, inst: &'a Inst, template: &str, group: &[usize]) -> R {
        self.loc = Some(inst.loc);
        let arrives = template.contains("barrier.cluster.arrive");
        let waits = template.contains("barrier.cluster.wait");
        let relaxed = template.contains("barrier.cluster.arrive.relaxed");
        let aligned = template.contains(".aligned");
        if arrives {
            let arriving: Vec<usize> = group
                .iter()
                .copied()
                .filter(|&k| !waits || self.warp_lanes(k).iter().all(|&t| self.cluster_arrival[t].is_none()))
                .collect();
            if !arriving.is_empty() {
                self.cluster_arrive(&arriving, relaxed, aligned)?;
            }
        }
        let proceed = if waits { self.cluster_wait(inst, group, aligned)? } else { group.to_vec() };
        if proceed.is_empty() {
            return Ok(());
        }
        self.run_inst(inst, &proceed)?;
        for k in proceed {
            self.advance(k);
        }
        Ok(())
    }

    /// Report a cluster barrier instruction (`what`) that only some lanes of
    /// warp `k` execute, if it is `.aligned`.
    fn check_aligned(&self, k: usize, aligned: bool, what: &str) -> R {
        let lanes = self.warp_lanes(k);
        let base = lanes[0] - self.lane_of(lanes[0]);
        if aligned && lanes.len() != self.warp_size(base) {
            return Err(self.err(
                format!(
                    "{what} executed by {} of the {} lanes of warp {}: its .aligned form needs every thread of the \
                     warp to execute the same instruction (the non-aligned form lets threads arrive one by one)",
                    lanes.len(),
                    self.warp_size(base),
                    k % self.warps_per_block
                ),
                Some(lanes[0]),
            ));
        }
        Ok(())
    }

    /// `barrier.cluster.arrive{.relaxed}{.aligned}` by the lanes of the
    /// warps of `group`: each thread arrives in the current phase of its
    /// cluster's barrier (once: arriving again before the phase completes is
    /// reported); a release arrival publishes its warp's clock at cluster
    /// scope, and the warp moves on to a new time of its own.
    pub(super) fn cluster_arrive(&mut self, group: &[usize], relaxed: bool, aligned: bool) -> R {
        let what = if relaxed { "cluster_arrive_relaxed" } else { "cluster_arrive" };
        let mut clusters: Vec<usize> = Vec::new();
        for &k in group {
            self.check_aligned(k, aligned, what)?;
            let (block, slot) = (k / self.warps_per_block, self.slot_of(k));
            let c = block / self.cluster_size;
            let phase = self.cluster_barriers[c].phase;
            for &t in self.warps[k].groups[0].frames.last().expect("a running warp has a frame").lanes.iter() {
                if let Some(earlier) = self.cluster_arrival[t] {
                    // Whether the earlier phase has completed depends on the
                    // schedule; arriving twice without a wait never is right.
                    return Err(self.err(
                        format!(
                            "{what}: the thread arrives at the cluster barrier again without having waited since its \
                             arrival in phase {earlier} (each thread arrives once per phase, then waits with \
                             cluster_wait before arriving again)"
                        ),
                        Some(t),
                    ));
                }
                self.cluster_arrival[t] = Some(phase);
            }
            let clock = std::mem::take(&mut self.clocks[k]);
            let barrier = &mut self.cluster_barriers[c];
            if !relaxed {
                join(&mut barrier.release, &clock);
            } else if let Some(fence) = &self.fence_release[k] {
                // A fence then a relaxed arrival is a release pattern (the
                // PTX ISA's example for barrier.cluster.arrive.relaxed): it
                // publishes the clock at the fence.
                join(&mut barrier.release, fence);
            }
            barrier.init_release[slot] = barrier.init_release[slot].max(clock[slot]);
            self.clocks[k] = clock;
            self.clocks[k][slot] += 1;
            if !clusters.contains(&c) {
                clusters.push(c);
            }
        }
        self.progress();
        for c in clusters {
            self.complete_cluster_barrier(c);
        }
        Ok(())
    }

    /// `barrier.cluster.wait{.aligned}` by the warps of `group`: a warp
    /// whose lanes' phase has completed acquires what its arrivals released
    /// and goes on (it is returned); one whose phase is still running
    /// blocks. A thread that has not arrived would wait for itself.
    pub(super) fn cluster_wait(&mut self, inst: &'a Inst, group: &[usize], aligned: bool) -> R<Vec<usize>> {
        let mut proceed = Vec::new();
        for &k in group {
            self.check_aligned(k, aligned, "cluster_wait")?;
            let c = (k / self.warps_per_block) / self.cluster_size;
            let phase = self.cluster_barriers[c].phase;
            let lanes = self.warp_lanes(k).to_vec();
            if let Some(&t) = lanes.iter().find(|&&t| self.cluster_arrival[t].is_none()) {
                return Err(self.err(
                    "cluster_wait by a thread that has not arrived at the cluster barrier in this phase: it would \
                     wait for its own arrival, so the barrier never completes (cluster_arrive first)",
                    Some(t),
                ));
            }
            if lanes.iter().any(|&t| self.cluster_arrival[t] == Some(phase)) {
                self.warps[k].status = Status::Blocked(Wait::Cluster { inst });
                continue;
            }
            let barrier = &self.cluster_barriers[c];
            join(&mut self.clocks[k], &barrier.completed);
            join(&mut self.init_clocks[k], &barrier.init_completed);
            for t in lanes {
                self.cluster_arrival[t] = None;
            }
            proceed.push(k);
        }
        Ok(proceed)
    }

    /// The threads of cluster `c` that have not exited, by thread index.
    fn live_cluster_threads(&self, c: usize) -> impl Iterator<Item = usize> + '_ {
        let tpb = self.threads_per_block;
        let w = self.warps_per_block;
        let blocks = c * self.cluster_size..(c + 1) * self.cluster_size;
        blocks
            .flat_map(move |b| b * tpb..(b + 1) * tpb)
            .filter(move |&t| !matches!(self.warps[(t / tpb) * w + (t % tpb) / 32].status, Status::Exited))
    }

    /// Complete cluster `c`'s barrier phase if some thread has arrived in it
    /// and every thread of the cluster that has not exited has, waking the
    /// warps that wait for it.
    pub(super) fn complete_cluster_barrier(&mut self, c: usize) {
        let phase = self.cluster_barriers[c].phase;
        let mut any = false;
        for t in self.live_cluster_threads(c) {
            if self.cluster_arrival[t] != Some(phase) {
                return;
            }
            any = true;
        }
        if !any {
            return;
        }
        let barrier = &mut self.cluster_barriers[c];
        barrier.phase += 1;
        barrier.completed = std::mem::replace(&mut barrier.release, vec![0; barrier.completed.len()]);
        barrier.init_completed = std::mem::replace(&mut barrier.init_release, vec![0; barrier.init_completed.len()]);
        self.report.barriers += 1;
        let w = self.warps_per_block;
        let round = self.round;
        let warps = c * self.cluster_size * w..(c + 1) * self.cluster_size * w;
        for state in self.warps[warps.clone()].iter_mut().chain(self.parked[warps].iter_mut().flatten()) {
            if let Status::Blocked(Wait::Cluster { .. }) = state.status {
                state.status = Status::Runnable;
                state.ready_round = round + 1;
            }
        }
    }

    /// The mbarrier phase that warp `k`'s lanes would have to wait for at
    /// `mbarrier.try_wait` `inst`, if any. Lanes whose wait the instruction
    /// reports as a mistake do not block: executing it reports them.
    fn wait_blocker(&mut self, inst: &Inst, k: usize) -> R<Option<(MbarrierKey, u64)>> {
        let Op::MbarrierWait { buffer, index, parity, state: by_state, .. } = &inst.op else {
            unreachable!("an mbarrier wait")
        };
        let w = self.warps_per_block;
        let lanes = self.warp_lanes(k).to_vec();
        let mut blocker = None;
        for t in lanes {
            let Ok(byte) = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, "mbarrier.try_wait") else {
                return Ok(None);
            };
            let key = (self.block_rank(t), buffer.0, byte);
            self.fire_deferred_arrivals(key);
            self.land_bulk_copies(key, t)?;
            let Some(state) = self.mbarriers.get(&key) else { return Ok(None) };
            let raw = self.get(*parity, t);
            let over = if *by_state { state.phase > raw } else { raw & 1 != state.phase & 1 };
            let asked = if over { state.phase.saturating_sub(1) } else { state.phase };
            if !*by_state && state.known_phases(&self.clocks[self.block_rank(t) * w + self.warp_of(t) as usize]) < asked
            {
                return Ok(None);
            }
            if !over && blocker.is_none() {
                blocker = Some((key, state.phase));
            }
        }
        Ok(blocker)
    }

    /// Wake the warps blocked at an `mbarrier.try_wait` whose phase has
    /// completed, landing the bulk copies and making the `cp.async`
    /// arrivals that the phase still waits for first.
    pub(super) fn wake_mbarrier_waiters(&mut self) -> R {
        for (k, g) in self.all_groups() {
            let Status::Blocked(Wait::Mbarrier { key, phase, inst }) = self.group(k, g).status else { continue };
            self.loc = Some(inst.loc);
            let t = self.group(k, g).lanes()[0];
            self.fire_deferred_arrivals(key);
            self.land_bulk_copies(key, t)?;
            if self.mbarriers.get(&key).is_none_or(|m| m.phase != phase) {
                self.group_mut(k, g).status = Status::Runnable;
            }
        }
        Ok(())
    }

    /// `setmaxnreg` by the warps of `group`: each joins its warpgroup's pending one and waits;
    /// once all four warps have, it completes (an increase once the block's pool has the
    /// registers). Checks that whole warps of whole warpgroups execute the same one, that
    /// the warps synchronized since the previous one, and the budget the PTX ISA requires.
    pub(super) fn setmaxnreg(&mut self, inst: &'a Inst, group: &[usize], increase: bool, count: u32) -> R {
        let w = self.warps_per_block;
        let tpb = self.threads_per_block;
        let what = inst.op.name();
        if !tpb.is_multiple_of(128) {
            return Err(self
                .err(format!("{what} needs whole warpgroups of 128 threads, but the block has {tpb} threads"), None));
        }
        let mut full = Vec::new();
        for &k in group {
            let lanes = self.warp_lanes(k).to_vec();
            let (block, warp) = (k / w, k % w);
            if lanes.len() != 32 {
                return Err(self.err(
                    format!(
                        "{what} {count} executed by {} of the 32 lanes of warp {warp}: every thread of the warpgroup \
                         executes it together (.sync.aligned), outside divergent code",
                        lanes.len()
                    ),
                    Some(lanes[0]),
                ));
            }
            let wg = warp / 4;
            let slot = self.slot_of(k);
            let seen = self.clocks[k].clone();
            let groups = tpb / 128;
            let initial = setmaxnreg_initial_budget(tpb);
            let state = self.setmaxnreg.entry(block).or_insert_with(|| BlockRegs {
                free: 0,
                groups: vec![WarpgroupRegs { budget: initial, pending: None, last: Vec::new() }; groups],
            });
            let g = &mut state.groups[wg];
            if let Some(&(o, _, _)) = g.last.iter().find(|&&(o, s, time)| o != k && seen[s] <= time) {
                return Err(self.err(
                    format!(
                        "{what} {count} by warp {warp} of warpgroup {wg} before it has synchronized with warp {} since \
                         their previous setmaxnreg: separate consecutive setmaxnreg of a warpgroup with a barrier \
                         (a named barrier over the warpgroup, say)",
                        o % w
                    ),
                    Some(lanes[0]),
                ));
            }
            let p =
                g.pending.get_or_insert(PendingRegs { increase, count, inst, arrived: Vec::new(), joined: Vec::new() });
            if (p.increase, p.count) != (increase, count) {
                let other = format!("setmaxnreg.{} {}", if p.increase { "inc" } else { "dec" }, p.count);
                let at = Self::at(p.inst);
                let others = warp_list(&p.arrived.iter().map(|a| 4 * wg + a).collect::<Vec<_>>());
                return Err(self.err(
                    format!(
                        "{what} {count} by warp {warp} while warps {others} of warpgroup {wg} execute {other} ({at}): \
                         the warps of a warpgroup execute the same setmaxnreg"
                    ),
                    Some(lanes[0]),
                ));
            }
            if p.arrived.contains(&(warp % 4)) {
                let missing: Vec<usize> = (0..4).filter(|a| !p.arrived.contains(a)).map(|a| 4 * wg + a).collect();
                return Err(self.err(
                    format!(
                        "{what} {count}: warp {warp} executes it again before warps {} of warpgroup {wg} have executed \
                         the previous one",
                        warp_list(&missing)
                    ),
                    Some(lanes[0]),
                ));
            }
            p.arrived.push(warp % 4);
            p.joined.push((k, slot, seen[slot]));
            if p.arrived.len() == 4 {
                full.push((block, wg));
            }
            self.clocks[k][slot] += 1;
            self.advance(k);
            self.warps[k].status = Status::Blocked(Wait::SetMaxNReg { block, group: wg, inst });
        }
        self.progress();
        for (block, wg) in full {
            self.complete_setmaxnreg(block, wg)?;
        }
        Ok(())
    }

    /// Complete warpgroup `wg`'s pending `setmaxnreg` if every warp has executed it and, for
    /// an increase, the pool has the registers; a decrease then lets waiting increases of
    /// the block go ahead.
    fn complete_setmaxnreg(&mut self, block: usize, wg: usize) -> R {
        let state = self.setmaxnreg.get_mut(&block).expect("a setmaxnreg state");
        let g = &mut state.groups[wg];
        let Some(p) = g.pending.as_ref().filter(|p| p.arrived.len() == 4) else { return Ok(()) };
        let (increase, count, budget, inst) = (p.increase, p.count, g.budget, p.inst);
        let what = format!("setmaxnreg.{} {count}", if increase { "inc" } else { "dec" });
        if increase && budget > count {
            return Err(SimError {
                loc: Some(inst.loc),
                ..self.err(
                    format!(
                        "{what} by warpgroup {wg}, whose budget is already {budget} registers per thread: an increase \
                         needs a budget of at most its count"
                    ),
                    None,
                )
            });
        }
        if !increase && budget < count {
            return Err(SimError {
                loc: Some(inst.loc),
                ..self.err(
                    format!(
                        "{what} by warpgroup {wg}, whose budget is only {budget} registers per thread: a decrease \
                         needs a budget of at least its count"
                    ),
                    None,
                )
            });
        }
        let need = (count.saturating_sub(budget)) * 128;
        if increase && state.free < need {
            // It waits for registers.
            return Ok(());
        }
        if increase {
            state.free -= need;
        } else {
            state.free += (budget - count) * 128;
        }
        let g = &mut state.groups[wg];
        let p = g.pending.take().expect("pending");
        g.budget = count;
        g.last = p.joined.clone();
        let round = self.round;
        for &(k, _, _) in &p.joined {
            let s = &mut self.warps[k];
            s.status = Status::Runnable;
            s.ready_round = round + 1;
        }
        self.progress();
        if !increase {
            let groups = self.setmaxnreg[&block].groups.len();
            for other in (0..groups).filter(|&o| o != wg) {
                self.complete_setmaxnreg(block, other)?;
            }
        }
        Ok(())
    }

    /// For hang and deadlock reports: what a warp waiting at a `setmaxnreg` waits for.
    pub(super) fn setmaxnreg_wait(&self, block: usize, wg: usize, ctx: &str) -> String {
        let w = self.warps_per_block;
        let Some(state) = self.setmaxnreg.get(&block) else { return String::new() };
        let g = &state.groups[wg];
        let Some(p) = &g.pending else { return String::new() };
        let site = format!("setmaxnreg.{} {} ({})", if p.increase { "inc" } else { "dec" }, p.count, Self::at(p.inst));
        if p.arrived.len() < 4 {
            let missing: Vec<usize> =
                (0..4).filter(|a| !p.arrived.contains(a)).map(|a| block * w + 4 * wg + a).collect();
            return format!(
                " at {site}{ctx}, which warps {} of warpgroup {wg} have not executed: {}",
                warp_list(&missing.iter().map(|k| k % w).collect::<Vec<_>>()),
                self.whereabouts_of(&missing, false)
            );
        }
        let need = (p.count - g.budget) * 128;
        format!(
            " at {site}{ctx}, an increase from {} to {} registers per thread that needs {need} registers from the \
             block's pool, which has {}: no warpgroup releases them (setmaxnreg.dec), so it can never be served",
            g.budget, p.count, state.free
        )
    }

    /// A barrier instruction (`inst`) executed by the warps of `group`:
    /// `sync_threads` (`id` is `None`: barrier 0, for every thread of the
    /// block), or `bar.sync` / `bar.arrive` on barrier `id`, of `threads`
    /// threads (every thread of the block without a count). Arriving warps
    /// that wait block until the barrier completes; the last arrival
    /// completes it and wakes them.
    pub(super) fn barrier(
        &mut self,
        inst: &'a Inst,
        group: &[usize],
        kind: BarrierKind,
        id: Option<ValueId>,
        threads: Option<ValueId>,
        red: Option<(BarrierRedKind, ValueId, ValueId)>,
    ) -> R {
        let what = match (id, kind) {
            _ if red.is_some() => "barrier_cta_red",
            (None, _) => "sync_threads()",
            (_, BarrierKind::Sync) => "barrier",
            (_, BarrierKind::Arrive) => "barrier_arrive",
            (_, BarrierKind::ArriveUnaligned) => "barrier_arrive (unaligned)",
        };
        let w = self.warps_per_block;
        // (block, barrier id) -> (thread count, arriving warps)
        let mut arrivals: BTreeMap<(usize, usize), (Option<usize>, Vec<usize>)> = BTreeMap::new();
        // A bar.red's true and false predicates, and
        // its arriving warps' result values, by (block, barrier id).
        let mut red_parts: BTreeMap<(usize, usize), RedPart> = BTreeMap::new();
        for &k in group {
            let lanes = self.warp_lanes(k);
            let base = lanes[0] - self.lane_of(lanes[0]);
            let (block, warp) = (k / w, k % w);
            let partial = lanes.len() != self.warp_size(base);
            if partial && kind != BarrierKind::ArriveUnaligned {
                if id.is_none() {
                    let n: usize = group.iter().filter(|&&o| o / w == block).map(|&o| self.warp_lanes(o).len()).sum();
                    return Err(self.err(
                        format!(
                            "sync_threads() reached by {n} of {} threads (barriers must not be inside divergent \
                             control flow)",
                            self.threads_per_block
                        ),
                        None,
                    ));
                }
                return Err(self.err(
                    format!(
                        "{what} executed by {} of the {} lanes of warp {warp} (each participating warp must reach it as a whole)",
                        lanes.len(),
                        self.warp_size(base)
                    ),
                    Some(lanes[0]),
                ));
            }
            let uniform = |v: ValueId, name: &str| -> R<i32> {
                let first = self.get(v, lanes[0]) as u32 as i32;
                match lanes.iter().find(|&&t| self.get(v, t) as u32 as i32 != first) {
                    Some(&t) => Err(self.err(
                        format!(
                            "{what}: the lanes of warp {warp} disagree on the {name} ({first} and {})",
                            self.get(v, t) as u32 as i32
                        ),
                        Some(t),
                    )),
                    None => Ok(first),
                }
            };
            let b = match id {
                Some(id) => uniform(id, "barrier id")?,
                None => 0,
            };
            if !(0..16).contains(&b) {
                return Err(self.err(format!("{what}: barrier id {b} is out of range (0 to 15)"), Some(base)));
            }
            let count = match threads {
                Some(n) => {
                    let n = uniform(n, "thread count")?;
                    let limit = w * 32;
                    if n <= 0 || n % 32 != 0 || n as usize > limit {
                        return Err(self.err(
                            format!("{what}: a thread count of {n} is not a positive multiple of 32 up to the block's {limit}"),
                            Some(base),
                        ));
                    }
                    Some(n as usize)
                }
                None => None,
            };
            if partial {
                // barrier.arrive (unaligned) waits for every non-exited
                // thread of its warp: the warp arrives once all its lanes
                // have executed one for this barrier, from any instruction.
                let bits = lanes.iter().fold(0u32, |acc, &t| acc | 1 << self.lane_of(t));
                let full = if self.warp_size(base) == 32 { u32::MAX } else { (1u32 << self.warp_size(base)) - 1 };
                let first = lanes[0];
                let seen = self.unaligned.get(&(block, b as usize, warp)).copied().unwrap_or(0);
                if seen & bits != 0 {
                    return Err(self.err(
                        format!(
                            "{what}: lanes {} of warp {warp} arrive at barrier {b} again before the rest of their \
                             warp has arrived",
                            warp_list(&(0..32).filter(|l| (seen & bits) >> l & 1 == 1).collect::<Vec<_>>())
                        ),
                        Some(first),
                    ));
                }
                if seen | bits != full {
                    self.unaligned.insert((block, b as usize, warp), seen | bits);
                    continue;
                }
                self.unaligned.remove(&(block, b as usize, warp));
            }
            let entry = arrivals.entry((block, b as usize)).or_insert((count, Vec::new()));
            if entry.0 != count {
                return Err(self.err(
                    format!("{what}: warps arriving at barrier {b} together disagree on its thread count"),
                    Some(base),
                ));
            }
            entry.1.push(warp);
            if let Some((_, pred, result)) = red {
                let lanes = self.warp_lanes(k).to_vec();
                let part = red_parts.entry((block, b as usize)).or_default();
                for t in lanes {
                    if self.get(pred, t) & 1 == 1 {
                        part.0 += 1;
                    } else {
                        part.1 += 1;
                    }
                }
                part.2.push((k, result));
            }
        }
        for &k in group {
            self.advance(k);
        }
        self.progress();
        let waits = kind == BarrierKind::Sync;
        for ((block, b), (count, warps)) in arrivals {
            let expected = count.map_or(w, |n| n / 32);
            let all_threads = count.is_none() && waits;
            let state = &self.named[block][b];
            let pending = |s: &NamedState| -> Vec<usize> {
                s.arrived.iter().copied().filter(|a| !s.waiting.contains(a)).collect()
            };
            if id.is_none() && !pending(state).is_empty() {
                return Err(self.err(
                    format!(
                        "sync_threads() uses barrier 0 while warps {} have arrived at it with barrier_arrive and it has not completed",
                        warp_list(&pending(state))
                    ),
                    None,
                ));
            }
            if id.is_some() && state.site.is_some_and(|s| matches!(s.op, Op::Barrier)) {
                return Err(self.err(
                    format!(
                        "sync_threads() uses barrier 0 while warps {} arrive at it with {what} before it has \
                         completed (warps {} wait at the sync_threads())",
                        warp_list(&warps),
                        warp_list(&state.waiting)
                    ),
                    None,
                ));
            }
            if !state.arrived.is_empty() && state.expected != expected {
                return Err(self.err(
                    format!(
                        "{what}: barrier {b} is used with {} threads while warps {} wait on it with {} (the warps \
                         arriving at it disagree on its thread count)",
                        expected * 32,
                        warp_list(&state.arrived),
                        state.expected * 32
                    ),
                    None,
                ));
            }
            if let Some(&again) = warps.iter().find(|w| state.arrived.contains(w)) {
                return Err(self.err(
                    format!("{what}: warp {again} arrives at barrier {b} again before the barrier has completed"),
                    None,
                ));
            }
            let total = state.arrived.len() + warps.len();
            if total > expected {
                return Err(self.err(
                    format!(
                        "{what}: {total} warps arrive at barrier {b}, which completes after {expected} ({} threads)",
                        expected * 32
                    ),
                    None,
                ));
            }
            // An all-thread barrier must be reached at one instruction, by
            // every warp of the block.
            if all_threads || state.all_threads {
                if let Some(site) = state.site.filter(|s| !std::ptr::eq(*s, inst)) {
                    return Err(self.err(
                        format!(
                            "{what} reached by {} of {} threads here, while warps {} wait at another one ({}:{}) \
                             (barriers must not be inside divergent control flow)",
                            self.threads_in(&warps),
                            self.threads_per_block,
                            warp_list(&state.waiting),
                            site.loc.file(),
                            site.loc.line()
                        ),
                        None,
                    ));
                }
                let exited = self.exited_warps(block);
                if !exited.is_empty() && total + exited.len() == expected {
                    let mut arrived = state.arrived.clone();
                    arrived.extend(&warps);
                    return Err(self.exit_error(inst, block, &arrived, &exited));
                }
            }
            // A bar.red's warps must all reduce the same way.
            let red_kind = red.map(|r| r.0);
            if !state.arrived.is_empty() && state.red != red_kind {
                let name = |k: Option<BarrierRedKind>| match k {
                    Some(k) => format!("barrier_cta_red ({})", k.name()),
                    None => "a plain barrier".to_string(),
                };
                return Err(self.err(
                    format!(
                        "{what}: warps {} reach barrier {b} with {} while warps {} wait at it with {}: every thread \
                         of a barrier's use executes the same reduction",
                        warp_list(&warps),
                        name(red_kind),
                        warp_list(&state.arrived),
                        name(state.red)
                    ),
                    None,
                ));
            }
            let part = red_parts.remove(&(block, b)).unwrap_or_default();
            let state = self.named[block][b].clone();
            let release = if state.release.is_empty() { vec![0; self.slots()] } else { state.release.clone() };
            if total == expected {
                if let Some(kind) = red_kind {
                    // Every participant gets the reduction of all predicates.
                    let (yes, no) = (state.red_counts.0 + part.0, state.red_counts.1 + part.1);
                    let value = match kind {
                        BarrierRedKind::Popc => yes as u64,
                        BarrierRedKind::And => (no == 0) as u64,
                        BarrierRedKind::Or => (yes > 0) as u64,
                    };
                    for &(k, result) in state.red_results.iter().chain(&part.2) {
                        let lanes = self.warp_lanes(k).to_vec();
                        for t in lanes {
                            self.set(result, t, value);
                        }
                    }
                }
                // Complete: waiting warps acquire everything the arrivals released.
                let mut acquiring = state.waiting.clone();
                if waits {
                    acquiring.extend(&warps);
                }
                self.synchronize(block, &warps, &acquiring, &release);
                for &warp in &state.waiting {
                    let woken = &mut self.warps[block * w + warp];
                    woken.status = Status::Runnable;
                    woken.ready_round = self.round + 1;
                }
                self.named[block][b] = NamedState::default();
                self.report.barriers += 1;
            } else {
                // Arrive (and wait, for bar.sync): the barrier keeps what
                // these warps release.
                let mut joined = release;
                for &warp in &warps {
                    join(&mut joined, &self.clocks[block * w + warp]);
                }
                self.synchronize(block, &warps, &[], &[]);
                let state = &mut self.named[block][b];
                state.arrived.extend(&warps);
                state.expected = expected;
                state.release = joined;
                state.red = red_kind;
                state.red_counts.0 += part.0;
                state.red_counts.1 += part.1;
                state.red_results.extend(part.2);
                if waits {
                    state.waiting.extend(&warps);
                    state.all_threads |= all_threads;
                    state.site.get_or_insert(inst);
                    for &warp in &warps {
                        self.warps[block * w + warp].status = Status::Blocked(Wait::Barrier { block, id: b, inst });
                    }
                }
            }
        }
        Ok(())
    }

    /// The report of `what` on the mbarrier keyed by
    /// `key` if `mbarrier.inval` invalidated it.
    pub(super) fn invalidated_error(&self, key: MbarrierKey, what: &str, t: usize) -> Option<SimError> {
        let at = self.invalidated.get(&key)?;
        Some(self.err(
            format!(
                "{what} on an mbarrier that mbarrier_inval invalidated at {}:{}: after an inval only mbarrier_init \
                 may use its memory as an mbarrier again",
                at.file(),
                at.line()
            ),
            Some(t),
        ))
    }

    /// The mbarrier operations of [`MbarrierExtKind`].
    pub(super) fn mbarrier_ext(&mut self, inst: &'a Inst, active: &[usize]) -> R {
        use crate::ir::MbarrierExtKind as K;
        let Op::MbarrierExt { kind, result, buffer, index, value } = &inst.op else { unreachable!("an mbarrier op") };
        let what = kind.ptx();
        self.progress();
        let mut released: Vec<usize> = Vec::new();
        for &t in active {
            let byte = self.checked_range(*buffer, self.index(*index, t), 1, 8, 8, t, what)?;
            let key = (self.block_rank(t), buffer.0, byte);
            if !self.mbarriers.contains_key(&key) {
                if let Some(e) = self.invalidated_error(key, what, t) {
                    return Err(e);
                }
                return Err(self.err(format!("{what} on an mbarrier that was never initialised"), Some(t)));
            }
            if *kind == K::Inval {
                let m = &self.mbarriers[&key];
                if !m.copies.is_empty() || !m.deferred.is_empty() {
                    return Err(self.err(
                        format!(
                            "mbarrier.inval on an mbarrier that {} bulk copies or cp.async arrivals still report to",
                            m.copies.len() + m.deferred.len()
                        ),
                        Some(t),
                    ));
                }
                self.mbarriers.remove(&key);
                self.invalidated.insert(key, inst.loc);
                continue;
            }
            let v = value.map_or(1, |v| self.get(v, t) as u32);
            let k = self.clock_index(t);
            let clock = self.clocks[k].clone();
            let state = self.mbarriers.get_mut(&key).expect("checked above");
            let token = state.phase;
            if *kind == K::CompleteTx {
                state.tx_count -= v as i64;
                state.complete_if_done();
            } else {
                // (arrivals, arrive-drop, expect-tx bytes, must not complete)
                let (n, drop, tx, nocomplete) = match kind {
                    K::ArriveState => (1, false, 0, false),
                    K::ArriveDrop => (1, true, 0, false),
                    K::ArriveDropExpectTx => (1, true, v, false),
                    K::ArriveDropNoComplete => (v, true, 0, true),
                    _ => (v, false, 0, true),
                };
                let (left, owed) = (state.arrivals_left, state.deferred.len() as u32);
                if n == 0 || n > left.saturating_sub(owed) {
                    return Err(self.err(
                        format!(
                            "{what}: {n} arrival(s) at an mbarrier whose phase is waiting for {left} ({owed} of them \
                             owed by asynchronous operations)"
                        ),
                        Some(t),
                    ));
                }
                if nocomplete && n == left && state.tx_count + tx as i64 == 0 {
                    return Err(self.err(
                        format!(
                            "{what} with {n} arrival(s) completes the mbarrier's phase, which a .noComplete arrival \
                             must not do (use mbarrier_arrive for the last arrival)"
                        ),
                        Some(t),
                    ));
                }
                join(&mut state.release, &clock);
                state.tx_count += tx as i64;
                state.arrivals_left -= n;
                if drop {
                    state.arrivals_per_phase -= n.min(state.arrivals_per_phase);
                }
                state.complete_if_done();
                if !released.contains(&k) {
                    released.push(k);
                }
            }
            if let Some(r) = result {
                self.set(*r, t, token);
            }
        }
        for k in released {
            let slot = self.slot_of(k);
            self.clocks[k][slot] += 1;
        }
        Ok(())
    }

    /// The warps of `group` execute `exit` (`inst`)
    /// with all their lanes: each ends as if it had reached the end of the
    /// kernel. An exit by part of a warp is not simulated.
    pub(super) fn exit_warps(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        for &k in group {
            let lanes = self.warp_lanes(k).to_vec();
            let base = lanes[0] - self.lane_of(lanes[0]);
            if lanes.len() != self.warp_size(base) || !self.parked[k].is_empty() {
                return Err(self.err(
                    format!(
                        "exit executed by {} of the {} lanes of warp {}: the simulator runs an exit only by a whole \
                         warp (exit by part of a warp is not simulated)",
                        lanes.len(),
                        self.warp_size(base),
                        k % self.warps_per_block
                    ),
                    Some(lanes[0]),
                ));
            }
            self.exit_sites.insert(k, inst.loc);
            self.report.exits += 1;
            let state = &mut self.warps[k];
            let frames = &mut state.groups[0].frames;
            frames.truncate(1);
            let body = &mut frames[0];
            body.next = body.region.insts.len();
            state.settled = false;
        }
        Ok(())
    }
}

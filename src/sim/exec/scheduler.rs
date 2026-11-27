// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's scheduler: residency and cluster start, the choice of which warp runs next, and the stepping and settling of instructions.

use super::*;

impl<'a> Machine<'a> {
    pub(in crate::sim) fn run(&mut self) -> R {
        let [gx, gy, gz] = self.cfg.grid;
        let [cx, cy, cz] = self.cfg.cluster;
        let mut clusters = VecDeque::new();
        for z in (0..gz).step_by(cz as usize) {
            for y in (0..gy).step_by(cy as usize) {
                for x in (0..gx).step_by(cx as usize) {
                    let mut blocks = Vec::new();
                    for bz in z..z + cz {
                        for by in y..y + cy {
                            for bx in x..x + cx {
                                blocks.push([bx, by, bz]);
                            }
                        }
                    }
                    if self.cluster_filter.as_ref().is_none_or(|f| f.contains(&[x, y, z])) {
                        clusters.push_back(blocks);
                    }
                }
            }
        }
        let resident = match self.opts.resident_clusters {
            super::ResidentClusters::Count(n) => n.max(1) as usize,
            super::ResidentClusters::Grid => clusters.len(),
        }
        .min(clusters.len());
        self.allocate(resident);
        for c in 0..resident {
            let blocks = clusters.pop_front().expect("a cluster per resident slot");
            self.start_cluster(c, blocks);
        }
        self.queued = clusters;
        self.schedule()
    }

    /// Allocate the state of `resident` clusters that run together; each
    /// cluster that starts in a slot ([`Self::start_cluster`]) resets its
    /// part.
    pub(super) fn allocate(&mut self, resident: usize) {
        let count = resident * self.cluster_size;
        let w = self.warps_per_block;
        let slots = self.slots();
        self.blocks = vec![[0; 3]; count];
        self.cluster_ids = vec![0; resident];
        self.nthreads = self.threads_per_block * count;
        self.loc = None;
        self.clocks = vec![vec![0; slots]; count * w];
        self.init_clocks = vec![vec![0; slots]; count * w];
        self.fence_release = vec![None; count * w];
        self.pending_acquire = vec![None; count * w];
        self.bulk_time = vec![0; count];
        self.named = vec![vec![NamedState::default(); 16]; count];
        self.values = vec![Vec::new(); self.ir.value_types.len()];
        let race_bytes = |bytes: usize| if self.opts.detect_races { Reads::new(bytes) } else { Reads::default() };
        self.values_hi = vec![Vec::new(); self.ir.value_types.len()];
        // The dynamic region gets the launch's bytes.
        let dynamic = self.cfg.dynamic_smem_bytes as usize;
        self.shared = self
            .ir
            .buffers
            .iter()
            .enumerate()
            .map(|(k, b)| {
                if b.space != MemSpace::Shared || self.home[k] != k {
                    return Vec::new();
                }
                let bytes = storage_bytes(self.ir, &self.home, k, dynamic);
                (0..count)
                    .map(|_| Shared {
                        data: Vec::new(),
                        meta: Vec::new(),
                        reads: race_bytes(bytes),
                        strong_reads: Reads::default(),
                    })
                    .collect()
            })
            .collect();
        let nthreads = self.nthreads;
        self.locals = self
            .ir
            .buffers
            .iter()
            .map(|b| {
                (b.space == MemSpace::Local).then(|| {
                    let per_thread = b.len.expect("local buffers have a size") * b.elem.bytes();
                    LocalMem {
                        per_thread,
                        data: vec![0xff; per_thread * nthreads],
                        written: vec![false; per_thread * nthreads],
                    }
                })
            })
            .collect();
        self.ops.clear();
        self.pending_results.clear();
        self.undefined.clear();
        self.wgmma.reset(count * w);
        self.tmaps.reset();
        self.proxy_fences = vec![Vec::new(); count * w];
        self.uncommitted = (0..nthreads).map(|_| Vec::new()).collect();
        self.bulk_uncommitted = (0..nthreads).map(|_| Vec::new()).collect();
        self.bulk_groups = (0..nthreads).map(|_| VecDeque::new()).collect();
        self.pending_global = 0;
        self.groups = (0..nthreads).map(|_| VecDeque::new()).collect();
        self.carry = vec![None; nthreads];
        self.in_flight = (0..nthreads).map(|_| Vec::new()).collect();
        self.cluster_arrival = vec![None; nthreads];
        self.warps = Vec::with_capacity(count * w);
        self.cluster_barriers = (0..resident).map(|_| ClusterBarrier::default()).collect();
    }

    /// Start the cluster of `blocks` in resident slot `c`: fresh shared
    /// memory, clocks (which start again with each cluster, so what other
    /// clusters published orders nothing here), barriers and warps.
    pub(super) fn start_cluster(&mut self, c: usize, blocks: Vec<[u32; 3]>) {
        let (w, tpb, cs) = (self.warps_per_block, self.threads_per_block, self.cluster_size);
        let slots = self.slots();
        let block_range = c * cs..(c + 1) * cs;
        let warp_range = c * cs * w..(c + 1) * cs * w;
        let thread_range = c * cs * tpb..(c + 1) * cs * tpb;
        self.next_cluster_id += 1;
        self.cluster_ids[c] = self.next_cluster_id;
        self.blocks[block_range.clone()].copy_from_slice(&blocks);
        for k in warp_range.clone() {
            self.clocks[k].fill(0);
            let slot = self.slot_of(k);
            self.clocks[k][slot] = 1;
            self.init_clocks[k].fill(0);
            self.fence_release[k] = None;
            self.pending_acquire[k] = None;
            self.proxy_fences[k].clear();
        }
        for block in block_range.clone() {
            self.bulk_time[block] = 0;
            self.named[block] = vec![NamedState::default(); 16];
        }
        self.mbarriers.retain(|key, _| !block_range.contains(&key.0));
        self.releases.retain(|key, _| !block_range.contains(&key.1));
        self.dsmem_log.retain(|block, _| !block_range.contains(block));
        // With every other cluster finished (always, with one resident
        // cluster), nothing that runs refers to earlier asynchronous
        // operations or their mbarriers: forget them, so that memory and the
        // scans over operations stay per cluster rather than per launch.
        let others_done =
            self.warps.iter().enumerate().all(|(k, s)| warp_range.contains(&k) || matches!(s.status, Status::Exited));
        if others_done {
            self.ops.clear();
            self.tmaps.forget_ops();
            self.first_live_op = 0;
            self.pending_results.clear();
            self.any_async_reads = false;
            self.mbarriers.clear();
        }
        self.unaligned.retain(|key, _| !block_range.contains(&key.0));
        self.setmaxnreg.retain(|block, _| !block_range.contains(block));
        self.tmem.start_cluster(block_range.clone());
        self.wgmma.start_cluster(warp_range.clone(), thread_range.clone(), block_range.clone(), others_done);
        let dynamic = self.cfg.dynamic_smem_bytes as usize;
        for (b, decl) in self.ir.buffers.iter().enumerate() {
            if decl.space != MemSpace::Shared || self.home[b] != b {
                continue;
            }
            let bytes = storage_bytes(self.ir, &self.home, b, dynamic);
            for block in block_range.clone() {
                let sh = &mut self.shared[b][block];
                sh.data = vec![0xff; bytes];
                sh.meta = vec![Meta::default(); bytes];
                if !sh.reads.is_empty() {
                    sh.reads = Reads::new(bytes);
                }
                sh.strong_reads = Reads::default();
            }
        }
        for mem in self.locals.iter_mut().flatten() {
            let bytes = mem.per_thread * thread_range.start..mem.per_thread * thread_range.end;
            mem.data[bytes.clone()].fill(0xff);
            mem.written[bytes].fill(false);
        }
        for t in thread_range.clone() {
            self.uncommitted[t].clear();
            self.groups[t].clear();
            self.bulk_uncommitted[t].clear();
            self.bulk_groups[t].clear();
            self.in_flight[t].clear();
            self.carry[t] = None;
            self.cluster_arrival[t] = None;
        }
        for (v, bits) in self.scalars.clone() {
            for t in thread_range.clone() {
                self.set128(v, t, bits);
            }
        }
        let body = &self.ir.body;
        for k in warp_range {
            let start = (k / w) * tpb + (k % w) * 32;
            let end = (start + 32).min((k / w + 1) * tpb);
            let frame = Frame { region: body, next: 0, lanes: (start..end).collect(), kind: FrameKind::Body };
            let state = WarpState::new(vec![frame]);
            if k < self.warps.len() {
                self.warps[k] = state;
                self.parked[k].clear();
            } else {
                self.warps.push(state);
                self.parked.push(Vec::new());
            }
        }
        self.cluster_barriers[c] = ClusterBarrier {
            phase: 0,
            release: vec![0; slots],
            completed: vec![0; slots],
            init_release: vec![0; slots],
            init_completed: vec![0; slots],
        };
        self.report.blocks += cs as u64;
    }

    /// Start the next queued cluster in each resident slot whose cluster
    /// has finished. Returns whether one started.
    fn refill(&mut self) -> bool {
        let per_cluster = self.cluster_size * self.warps_per_block;
        let mut started = false;
        for c in 0..self.cluster_ids.len() {
            if self.queued.is_empty() {
                break;
            }
            let done =
                self.warps[c * per_cluster..(c + 1) * per_cluster].iter().all(|s| matches!(s.status, Status::Exited));
            if done {
                let blocks = self.queued.pop_front().expect("a queued cluster");
                self.clc_launched = self.clc_launched.saturating_sub(1);
                self.start_cluster(c, blocks);
                started = true;
            }
        }
        started
    }

    /// Run the cluster's warps until every one has exited, by the schedule
    /// ([`super::Schedule`]). With `RoundRobin`, each round the scheduler
    /// hands every runnable warp one instruction in turn; warps that stand
    /// at the same instruction one after another in that order execute it
    /// together, which is the same thing, but cheaper (and it lets a barrier
    /// see the warps that reach it together). With `RunToBlock`, each round
    /// one warp runs until it blocks, yields or exits.
    pub(super) fn schedule(&mut self) -> R {
        let mut next_warp = 0;
        loop {
            self.round += 1;
            self.wake_mbarrier_waiters()?;
            let progress = match self.opts.schedule {
                super::Schedule::RoundRobin => {
                    let mut progress = false;
                    let mut cursor = 0;
                    while let Some(group) = self.next_group(&mut cursor)? {
                        progress = true;
                        self.step(&group)?;
                        self.check_idle()?;
                    }
                    progress
                }
                super::Schedule::RunToBlock => self.run_to_block(&mut next_warp)?,
                super::Schedule::Random { .. } => self.run_random()?,
            };
            let runnable =
                self.all_groups().into_iter().any(|(k, g)| matches!(self.group(k, g).status, Status::Runnable));
            if progress || runnable {
                self.refill();
                continue;
            }
            if self.refill() {
                continue;
            }
            if self.warps.iter().all(|s| matches!(s.status, Status::Exited)) {
                if self.refill() {
                    continue;
                }
                return Ok(());
            }
            // Nothing can run. An `mbarrier.try_wait` may give up after a
            // while (it then returns false); a barrier never does. One that
            // has given up again and again, with nothing visible happening
            // in between, would only give up again.
            let mut timed_out = false;
            let epoch = self.epoch;
            for state in self.warps.iter_mut().chain(self.parked.iter_mut().flatten()) {
                if let Status::Blocked(Wait::Mbarrier { .. }) = state.status {
                    let give_ups = match state.timeout_epoch {
                        Some((e, n)) if e == epoch => n,
                        _ => 0,
                    };
                    if give_ups >= GIVE_UPS_WITHOUT_PROGRESS {
                        continue;
                    }
                    state.status = Status::Runnable;
                    state.timed_out = true;
                    state.timeout_epoch = Some((epoch, give_ups + 1));
                    timed_out = true;
                }
            }
            if !timed_out {
                return Err(self.deadlock());
            }
        }
    }

    /// One round of [`super::Schedule::RunToBlock`]: the first runnable warp
    /// from `*next` on (cyclically) runs until it blocks, yields or exits,
    /// and the next round starts after it. Returns whether a warp ran.
    fn run_to_block(&mut self, next: &mut usize) -> R<bool> {
        let n = self.warps.len();
        for i in 0..n {
            let k = (*next + i) % n;
            self.take_turn(k);
            if self.current_inst(k)?.is_none() {
                continue;
            }
            self.warps[k].yielded = false;
            loop {
                self.step(&[k])?;
                self.check_idle()?;
                let at = self.current_inst(k)?;
                if at.is_none() || self.warps[k].yielded {
                    break;
                }
            }
            *next = k + 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// One step of [`super::Schedule::Random`]: a convergence group drawn
    /// from those that can run runs up to its next scheduling point
    /// ([`Self::scheduling_point`]), its next yield, or until it blocks or
    /// ends. Returns whether one ran.
    fn run_random(&mut self) -> R<bool> {
        let round = self.round;
        let ready: Vec<(usize, usize)> =
            self.all_groups().into_iter().filter(|&(k, g)| self.group(k, g).ready(round)).collect();
        if ready.is_empty() {
            return Ok(false);
        }
        let rng = self.sched_rng.as_mut().expect("a seeded schedule");
        let (k, g) = ready[(splitmix(rng) % ready.len() as u64) as usize];
        if g > 0 {
            self.swap_in(k, g);
        }
        self.warps[k].yielded = false;
        let mut first = true;
        while let Some(inst) = self.front_inst(k)? {
            if !first && (self.warps[k].yielded || self.scheduling_point(&inst.op)) {
                break;
            }
            first = false;
            self.step(&[k])?;
            self.check_idle()?;
        }
        Ok(true)
    }

    /// Whether the seeded schedule may switch groups before `op`: anything
    /// another warp could observe or be affected by (memory other than
    /// local, barriers, asynchronous operations, opaque statements).
    fn scheduling_point(&self, op: &Op) -> bool {
        match op {
            Op::Load { buffer, .. } | Op::Store { buffer, .. } => self.ir.buffer(*buffer).space != MemSpace::Local,
            Op::LdMatrix { .. }
            | Op::StMatrix { .. }
            | Op::SetMaxNReg { .. }
            | Op::Tcgen05 { .. }
            | Op::Tcgen05Ld { .. }
            | Op::Tcgen05St { .. }
            | Op::Tcgen05Mma { .. }
            | Op::Tcgen05Commit { .. }
            | Op::Tcgen05Cp { .. }
            | Op::ClcTryCancel { .. }
            | Op::ClcQueryCancel { .. }
            | Op::Atomic { .. }
            | Op::Red { .. }
            | Op::Barrier
            | Op::NamedBarrier { .. }
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
            | Op::MbarrierInit { .. }
            | Op::MbarrierArrive { .. }
            | Op::MbarrierExpectTx { .. }
            | Op::MbarrierWait { .. }
            | Op::FenceMbarrierInit
            | Op::GridDepControl { .. }
            | Op::CpAsync { .. }
            | Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::CpAsyncBulk { .. }
            | Op::CpAsyncBulkCommitGroup
            | Op::CpAsyncBulkWaitGroup { .. }
            | Op::TmaLoad { .. }
            | Op::TmaStore { .. }
            | Op::TmaPrefetch { .. }
            | Op::PrefetchTensormap { .. }
            | Op::TensormapReplace { .. }
            | Op::FenceTensormapRelease
            | Op::FenceTensormapAcquire { .. }
            | Op::TensormapCpFenceproxy { .. }
            | Op::BulkPrefetch { .. }
            | Op::StAsync { .. }
            | Op::FenceProxy { .. }
            | Op::CpAsyncMbarrierArrive { .. }
            | Op::Fence { .. }
            | Op::InlinePtx { .. }
            | Op::MlirOp { .. }
            | Op::Assert { .. }
            | Op::Printf { .. }
            | Op::Wgmma { .. }
            | Op::WgmmaFence { .. }
            | Op::WgmmaCommitGroup
            | Op::WgmmaWait { .. }
            | Op::BarrierRed { .. }
            | Op::MbarrierExt { .. }
            | Op::PrimMarker { .. }
            | Op::Exit
            | Op::StBulk { .. }
            | Op::Tcgen05Shift { .. } => true,
            _ => false,
        }
    }

    /// At a scheduling point, with [`super::AsyncCompletion::Random`]:
    /// complete an operation in flight, drawn at random, one time in four.
    fn complete_at_random(&mut self) {
        // Arrive-ons owed for copies that have all landed (by now, or at
        // their thread's cp_async_wait_group) fire.
        let owed: Vec<MbarrierKey> =
            self.mbarriers.iter().filter(|(_, m)| !m.deferred.is_empty()).map(|(&key, _)| key).collect();
        for key in owed {
            self.fire_ready_arrivals(key);
        }
        let Some(rng) = self.async_rng.as_mut() else { return };
        if !splitmix(rng).is_multiple_of(4) {
            return;
        }
        while self.first_live_op < self.ops.len() && !self.ops[self.first_live_op].in_flight() {
            self.first_live_op += 1;
        }
        let live: Vec<usize> = (self.first_live_op..self.ops.len()).filter(|&id| self.ops[id].in_flight()).collect();
        if live.is_empty() {
            return;
        }
        let rng = self.async_rng.as_mut().expect("checked above");
        let id = live[(splitmix(rng) % live.len() as u64) as usize];
        match self.ops[id].trigger {
            Trigger::CpAsync { .. } => {
                // Owed arrive-ons fire at the next scheduling point.
                self.land_tracked_copy(id);
            }
            Trigger::CompleteTx(key) => self.complete_bulk_copy(id, key),
            Trigger::BulkGroup => self.land_bulk_store(id),
            // Its results land at the warp's wait, so that reading them
            // earlier is always reported.
            Trigger::Wgmma | Trigger::TmemLd => {}
            Trigger::Tcgen05Mma => {
                self.umma_land(id);
            }
            #[cfg(test)]
            Trigger::Test => {}
        }
    }

    /// At the start of warp `k`'s turn (round-robin or run-to-block): if the
    /// group it runs cannot run, or yielded since its last turn, while
    /// another of its groups can, the next of those takes over (the others
    /// keep their order), so groups of one warp that wait for each other
    /// (a spin on a flag the other branch sets) all make progress.
    fn take_turn(&mut self, k: usize) {
        if self.parked[k].is_empty() {
            return;
        }
        let round = self.round;
        if self.warps[k].ready(round) && !self.warps[k].yielded {
            return;
        }
        let Some(i) = self.parked[k].iter().position(|g| g.ready(round)) else { return };
        let next = self.parked[k].remove(i);
        let mut old = std::mem::replace(&mut self.warps[k], next);
        old.yielded = false;
        self.parked[k].push(old);
    }

    /// Something visible happened (see [`SimOptions::max_idle_steps`]).
    pub(super) fn progress(&mut self) {
        self.epoch += 1;
        self.idle = 0;
    }

    /// Warp `k` reached a yield point.
    pub(super) fn yield_warp(&mut self, k: usize) {
        let state = &mut self.warps[k];
        state.yielded = true;
        state.spin_epoch = self.epoch;
    }

    /// Whether a runnable warp spins: it has yielded and polled memory
    /// since anything visible last happened.
    pub(super) fn spins(&self, s: &WarpState) -> bool {
        s.spin_epoch == self.epoch && s.poll_epoch == self.epoch
    }

    /// Report a hang once nothing visible has happened for
    /// [`SimOptions::max_idle_steps`] steps and every warp that has not
    /// exited is blocked or spinning.
    fn check_idle(&mut self) -> R {
        if self.idle <= self.opts.max_idle_steps {
            return Ok(());
        }
        let hung = self.warps.iter().chain(self.parked.iter().flatten()).all(|s| match s.status {
            Status::Runnable => self.spins(s),
            Status::Blocked(_) | Status::Exited => true,
        });
        if hung {
            return Err(self.hang());
        }
        // Some warp is still busy: look again later.
        self.idle = self.opts.max_idle_steps / 2;
        Ok(())
    }

    /// The next warps to step, from warp `*cursor` on, by the schedule
    /// ([`Schedule::RoundRobin`](super::Schedule::RoundRobin): the next
    /// runnable warp, with the runnable warps right after it that stand at
    /// the same instruction), and moves the cursor past them.
    fn next_group(&mut self, cursor: &mut usize) -> R<Option<Vec<usize>>> {
        let mut group: Vec<usize> = Vec::new();
        let mut at: Option<&'a Inst> = None;
        let mut k = *cursor;
        while k < self.warps.len() {
            self.take_turn(k);
            if let Some(inst) = self.current_inst(k)? {
                match at {
                    None => at = Some(inst),
                    Some(i) if std::ptr::eq(i, inst) => {}
                    Some(_) => break,
                }
                group.push(k);
            }
            k += 1;
        }
        *cursor = group.last().map_or(k, |&last| last + 1);
        Ok((!group.is_empty()).then_some(group))
    }

    /// The instruction runnable warp `k` executes next (after handling the
    /// region ends before it), or `None` if it is blocked or has exited.
    /// If the group it runs cannot run, another of its groups that can
    /// takes over.
    fn current_inst(&mut self, k: usize) -> R<Option<&'a Inst>> {
        loop {
            if let Some(inst) = self.front_inst(k)? {
                return Ok(Some(inst));
            }
            let round = self.round;
            match self.parked[k].iter().position(|g| g.ready(round)) {
                Some(i) => self.swap_in(k, i + 1),
                None => return Ok(None),
            }
        }
    }

    /// The instruction the group warp `k` runs executes next, if it can run
    /// (after handling the region ends before it; a group that ends there
    /// hands over to the one waiting for it).
    fn front_inst(&mut self, k: usize) -> R<Option<&'a Inst>> {
        loop {
            if !self.warps[k].ready(self.round) {
                return Ok(None);
            }
            if self.warps[k].settled {
                break;
            }
            self.settle(k)?;
        }
        let frame = self.warps[k].groups[0].frames.last().expect("a running warp has a frame");
        let region: &'a Region = frame.region;
        Ok(Some(&region.insts[frame.next]))
    }

    /// The lanes of warp `k` that run its next instruction.
    pub(super) fn warp_lanes(&self, k: usize) -> &[usize] {
        &self.warps[k].groups[0].frames.last().expect("a running warp has a frame").lanes
    }

    /// [`Self::inst`] for the lanes of the warps of `group`, in order.
    pub(super) fn run_inst(&mut self, inst: &'a Inst, group: &[usize]) -> R {
        if let [k] = *group {
            // Lend the warp's lanes to the instruction.
            let frame = self.warps[k].groups[0].frames.last_mut().expect("a running warp has a frame");
            let lanes = std::mem::take(&mut frame.lanes);
            let result = self.inst(inst, &lanes);
            if self.wgmma.enabled && result.is_ok() {
                self.wgmma_note_access(&inst.op, &lanes);
            }
            self.warps[k].groups[0].frames.last_mut().expect("a running warp has a frame").lanes = lanes;
            return result;
        }
        let mut active = std::mem::take(&mut self.active);
        active.clear();
        for &k in group {
            active.extend_from_slice(self.warp_lanes(k));
        }
        let result = self.inst(inst, &active);
        if self.wgmma.enabled && result.is_ok() {
            self.wgmma_note_access(&inst.op, &active);
        }
        self.active = active;
        result
    }

    /// Move warp `k` past the instruction it executed.
    pub(super) fn advance(&mut self, k: usize) {
        let state = &mut self.warps[k];
        state.groups[0].frames.last_mut().expect("a running warp has a frame").next += 1;
        state.settled = false;
    }

    /// Execute one instruction for the warps of `group`, which all stand
    /// at it.
    pub(super) fn step(&mut self, group: &[usize]) -> R {
        let frame = self.warps[group[0]].groups[0].frames.last().expect("a running warp has a frame");
        let region: &'a Region = frame.region;
        let inst: &'a Inst = &region.insts[frame.next];
        self.idle += group.len() as u64;
        self.steps += 1;
        if self.async_rng.is_some() && self.scheduling_point(&inst.op) {
            self.complete_at_random();
        }
        let count = |m: &mut Self| {
            m.loc = Some(inst.loc);
            m.report.warp_instructions += group.len() as u64;
            m.clock += 1;
        };
        match &inst.op {
            Op::If { .. } | Op::For { .. } | Op::While { .. } | Op::Call { body: Some(_), .. } => {
                count(self);
                for &k in group {
                    self.enter(k, inst)?;
                }
            }
            Op::Barrier => {
                count(self);
                self.barrier(inst, group, BarrierKind::Sync, None, None, None)?;
            }
            Op::NamedBarrier { kind, id, threads } => {
                count(self);
                self.barrier(inst, group, *kind, Some(*id), *threads, None)?;
            }
            Op::BarrierRed { result, kind, pred, id, threads } => {
                count(self);
                self.barrier(inst, group, BarrierKind::Sync, Some(*id), *threads, Some((*kind, *pred, *result)))?;
            }
            Op::Exit => {
                count(self);
                self.exit_warps(inst, group)?;
            }
            Op::SetMaxNReg { increase, count: n } => {
                count(self);
                self.setmaxnreg(inst, group, *increase, *n)?;
            }
            Op::Wgmma { .. } => {
                count(self);
                self.wgmma_issue(inst, group)?;
            }
            Op::Tcgen05 { inst: which, .. } if which.is_warp_collective() => {
                count(self);
                self.tcgen05_issue(inst, group)?;
            }
            Op::ClusterArrive { relaxed, aligned } => {
                count(self);
                self.cluster_arrive(group, *relaxed, *aligned)?;
                for &k in group {
                    self.advance(k);
                }
            }
            Op::ClusterWait { aligned } => {
                count(self);
                let proceed = self.cluster_wait(inst, group, *aligned)?;
                for k in proceed {
                    self.advance(k);
                }
            }
            Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => self.try_wait(inst, group)?,
            Op::Shuffle { mask, .. }
            | Op::Vote { mask, .. }
            | Op::Match { mask, .. }
            | Op::Redux { mask, .. }
            | Op::WarpBarrier { mask }
                if group.iter().any(|&k| !self.parked[k].is_empty()) =>
            {
                for &k in group {
                    self.rendezvous(inst, k, *mask)?;
                }
            }
            Op::InlinePtx { template, predicated: false, .. } if template.contains("barrier.cluster.") => {
                self.inline_cluster_barrier(inst, template, group)?
            }
            _ => {
                self.run_inst(inst, group)?;
                for &k in group {
                    self.advance(k);
                }
            }
        }
        Ok(())
    }

    /// Warp `k` executes structured instruction `inst` (`If`, `For` or
    /// `While`): it enters the instruction's first region.
    fn enter(&mut self, k: usize, inst: &'a Inst) -> R {
        self.advance(k);
        let lanes = self.warp_lanes(k).to_vec();
        let frame = match &inst.op {
            Op::If { cond, then_region, else_region, .. } => {
                let (taken, not_taken): (Vec<usize>, Vec<usize>) =
                    lanes.iter().partition(|&&t| self.get(*cond, t) != 0);
                if taken.is_empty() || not_taken.is_empty() {
                    Frame {
                        region: then_region,
                        next: 0,
                        lanes: taken,
                        kind: FrameKind::IfThen { inst, else_lanes: not_taken, split: None },
                    }
                } else {
                    // Divergent: the else lanes become a convergence group
                    // of their own (independent thread scheduling), which
                    // the then group waits for at the end of the If. The
                    // then group runs first, as a SIMT stack would.
                    self.next_join += 1;
                    let id = self.next_join;
                    self.reset_carry(&not_taken);
                    let kind = FrameKind::IfElse { inst, then_lanes: Vec::new(), join: Some(id) };
                    let mut group =
                        WarpState::new(vec![Frame { region: else_region, next: 0, lanes: not_taken, kind }]);
                    group.ready_round = self.warps[k].ready_round;
                    self.parked[k].push(group);
                    Frame {
                        region: then_region,
                        next: 0,
                        lanes: taken,
                        kind: FrameKind::IfThen { inst, else_lanes: Vec::new(), split: Some(id) },
                    }
                }
            }
            // The unroll and vectorize hints do not change what a loop
            // computes; `at_least_once` is a promise the program makes.
            Op::For { lower, upper, step, init, args, body, opts, .. } => {
                let as_int = |m: &Self, v: ValueId, t: usize| m.get(v, t) as u32 as i32 as i64;
                let mut ivs = Box::new([0i64; 32]);
                for &t in &lanes {
                    let s = self.get(*step, t) as u32 as i32;
                    if s <= 0 {
                        return Err(self.err(format!("loop step must be positive, got {s}"), Some(t)));
                    }
                    let (lo, hi) = (as_int(self, *lower, t), as_int(self, *upper, t));
                    if opts.at_least_once && lo >= hi {
                        return Err(self.err(
                            format!(
                                "the loop is marked at_least_once, but runs zero times in this thread \
                                 ({lo}..{hi}); on the GPU the compiler may assume it runs"
                            ),
                            Some(t),
                        ));
                    }
                    for (a, x) in args.iter().zip(init) {
                        let bits = self.get128(*x, t);
                        self.set128(*a, t, bits);
                        self.carry_pending(*x, *a, t);
                    }
                    ivs[self.lane_of(t)] = lo;
                }
                let running: Vec<usize> =
                    lanes.iter().copied().filter(|&t| ivs[self.lane_of(t)] < as_int(self, *upper, t)).collect();
                self.start_iteration(inst, &running, &ivs);
                Frame { region: body, next: 0, lanes: running, kind: FrameKind::ForBody { inst, all: lanes, ivs } }
            }
            // A call simulated by its traced body: the body runs in the calling lanes.
            Op::Call { body: Some(body), .. } => {
                Frame { region: body, next: 0, lanes: lanes.clone(), kind: FrameKind::Call { inst } }
            }
            Op::While { init, cond_args, cond, .. } => {
                for &t in &lanes {
                    for (a, x) in cond_args.iter().zip(init) {
                        let bits = self.get128(*x, t);
                        self.set128(*a, t, bits);
                        self.carry_pending(*x, *a, t);
                    }
                }
                self.check_iterations(1, &lanes)?;
                Frame {
                    region: cond,
                    next: 0,
                    lanes: lanes.clone(),
                    kind: FrameKind::WhileCond { inst, all: lanes, iterations: 1 },
                }
            }
            _ => unreachable!("not a structured instruction"),
        };
        self.reset_carry(&frame.lanes);
        self.warps[k].groups[0].frames.push(frame);
        Ok(())
    }

    /// Set the induction variable of the `running` lanes of `For`
    /// instruction `inst` for their next iteration.
    fn start_iteration(&mut self, inst: &Inst, running: &[usize], ivs: &[i64; 32]) {
        let Op::For { iv, .. } = &inst.op else { unreachable!("a For") };
        for &t in running {
            self.set(*iv, t, ivs[self.lane_of(t)] as i32 as u32 as u64);
        }
    }

    /// Report a `while_carry` loop that has run more rounds than allowed.
    fn check_iterations(&self, iterations: u64, running: &[usize]) -> R {
        if iterations > self.opts.max_while_iterations {
            return Err(self.err("while_carry did not terminate", running.first().copied()));
        }
        Ok(())
    }

    /// A region is a new block of straight-line code, and the end of a
    /// structured instruction ends one: no carry flag.
    fn reset_carry(&mut self, lanes: &[usize]) {
        for &t in lanes {
            self.carry[t] = None;
        }
    }

    /// Handle the region ends that warp `k` stands at, until it stands at
    /// an instruction or has exited.
    fn settle(&mut self, k: usize) -> R {
        loop {
            let frames = &mut self.warps[k].groups[0].frames;
            if !frames.last().expect("a running warp has a frame").finished() {
                break;
            }
            let mut frame = frames.pop().expect("a running warp has a frame");
            let continued = match &mut frame.kind {
                FrameKind::Body => {
                    self.warps[k].status = Status::Exited;
                    self.warps[k].settled = true;
                    self.progress();
                    let block = k / self.warps_per_block;
                    self.complete_cluster_barrier(block / self.cluster_size);
                    self.land_bulk_stores_at_exit(block);
                    self.check_block_exit(block)?;
                    return self.check_exit(block);
                }
                FrameKind::IfThen { inst, split: Some(id), .. } => {
                    let (inst, id): (&'a Inst, u64) = (inst, *id);
                    if self.parked[k].iter().any(|g| g.join() == Some(id)) {
                        // Wait for the else group at the join.
                        self.warps[k].groups[0].frames.push(frame);
                        self.warps[k].status = Status::Blocked(Wait::Join { id, inst });
                        return Ok(());
                    }
                    let Op::If { then_region, results, .. } = &inst.op else { unreachable!("an If") };
                    for &t in &frame.lanes {
                        for (r, y) in results.iter().zip(&then_region.yields) {
                            let bits = self.get(*y, t);
                            self.set(*r, t, bits);
                            self.carry_pending(*y, *r, t);
                        }
                    }
                    self.reset_carry(&frame.lanes);
                    None
                }
                FrameKind::IfThen { inst, else_lanes, split: None } => {
                    let inst: &'a Inst = inst;
                    let Op::If { else_region: region, .. } = &inst.op else { unreachable!("an If") };
                    let lanes = std::mem::take(else_lanes);
                    self.reset_carry(&lanes);
                    let then_lanes = std::mem::take(&mut frame.lanes);
                    Some(Frame { region, next: 0, lanes, kind: FrameKind::IfElse { inst, then_lanes, join: None } })
                }
                FrameKind::IfElse { inst, then_lanes, join } => {
                    let Op::If { then_region, else_region, results, .. } = &inst.op else { unreachable!("an If") };
                    for (lanes, region) in [(&*then_lanes, then_region), (&frame.lanes, else_region)] {
                        for &t in lanes {
                            for (r, y) in results.iter().zip(&region.yields) {
                                let bits = self.get128(*y, t);
                                self.set128(*r, t, bits);
                                self.carry_pending(*y, *r, t);
                            }
                        }
                    }
                    self.reset_carry(then_lanes);
                    self.reset_carry(&frame.lanes);
                    if let Some(id) = *join {
                        self.end_group(k, id);
                        return Ok(());
                    }
                    None
                }
                FrameKind::Call { inst } => {
                    let Op::Call { body: Some(body), results, .. } = &inst.op else { unreachable!("a Call") };
                    for &t in &frame.lanes {
                        for (r, y) in results.iter().zip(&body.yields) {
                            let bits = self.get128(*y, t);
                            self.set128(*r, t, bits);
                            self.carry_pending(*y, *r, t);
                        }
                    }
                    self.reset_carry(&frame.lanes);
                    None
                }
                FrameKind::ForBody { inst, all, ivs } => {
                    let inst: &'a Inst = inst;
                    let Op::For { upper, step, args, body, results, .. } = &inst.op else { unreachable!("a For") };
                    self.loc = Some(inst.loc);
                    let as_int = |m: &Self, v: ValueId, t: usize| m.get(v, t) as u32 as i32 as i64;
                    let mut running = std::mem::take(&mut frame.lanes);
                    for &t in &running {
                        let next: Vec<u128> = body.yields.iter().map(|y| self.get128(*y, t)).collect();
                        for (a, bits) in args.iter().zip(next) {
                            self.set128(*a, t, bits);
                        }
                        for (a, y) in args.iter().zip(&body.yields) {
                            self.carry_pending(*y, *a, t);
                        }
                        ivs[self.lane_of(t)] += as_int(self, *step, t);
                    }
                    running.retain(|&t| ivs[self.lane_of(t)] < as_int(self, *upper, t));
                    if running.is_empty() {
                        for &t in all.iter() {
                            for (r, a) in results.iter().zip(args) {
                                let bits = self.get128(*a, t);
                                self.set128(*r, t, bits);
                                self.carry_pending(*a, *r, t);
                            }
                        }
                        self.reset_carry(all);
                        None
                    } else {
                        self.start_iteration(inst, &running, ivs);
                        self.reset_carry(&running);
                        frame.lanes = running;
                        frame.next = 0;
                        Some(frame)
                    }
                }
                FrameKind::WhileCond { inst, all, iterations } => {
                    let inst: &'a Inst = inst;
                    let Op::While { cond_args, cond, body_args, body, results, .. } = &inst.op else {
                        unreachable!("a While")
                    };
                    self.loc = Some(inst.loc);
                    let c = cond.yields[0];
                    let (go, stop): (Vec<usize>, Vec<usize>) = frame.lanes.iter().partition(|&&t| self.get(c, t) != 0);
                    for &t in &stop {
                        for (r, a) in results.iter().zip(cond_args) {
                            let bits = self.get128(*a, t);
                            self.set128(*r, t, bits);
                            self.carry_pending(*a, *r, t);
                        }
                    }
                    for &t in &go {
                        for (b, a) in body_args.iter().zip(cond_args) {
                            let bits = self.get128(*a, t);
                            self.set128(*b, t, bits);
                            self.carry_pending(*a, *b, t);
                        }
                    }
                    if go.is_empty() {
                        self.reset_carry(all);
                        None
                    } else {
                        self.reset_carry(&go);
                        let (all, iterations) = (std::mem::take(all), *iterations);
                        Some(Frame {
                            region: body,
                            next: 0,
                            lanes: go,
                            kind: FrameKind::WhileBody { inst, all, iterations },
                        })
                    }
                }
                FrameKind::WhileBody { inst, all, iterations } => {
                    let inst: &'a Inst = inst;
                    let Op::While { cond_args, cond, body, .. } = &inst.op else { unreachable!("a While") };
                    self.loc = Some(inst.loc);
                    for &t in &frame.lanes {
                        let next: Vec<u128> = body.yields.iter().map(|y| self.get128(*y, t)).collect();
                        for (a, bits) in cond_args.iter().zip(next) {
                            self.set128(*a, t, bits);
                        }
                        for (a, y) in cond_args.iter().zip(&body.yields) {
                            self.carry_pending(*y, *a, t);
                        }
                    }
                    let iterations = *iterations + 1;
                    self.check_iterations(iterations, &frame.lanes)?;
                    // A back-edge: spin loops let the other warps run.
                    self.yield_warp(k);
                    self.reset_carry(&frame.lanes);
                    let all = std::mem::take(all);
                    let lanes = std::mem::take(&mut frame.lanes);
                    Some(Frame { region: cond, next: 0, lanes, kind: FrameKind::WhileCond { inst, all, iterations } })
                }
            };
            if let Some(frame) = continued {
                self.warps[k].groups[0].frames.push(frame);
            }
        }
        self.warps[k].settled = true;
        Ok(())
    }

    /// The else group that warp `k` runs, of join record `id`, has finished:
    /// it ends, and the then group waiting for it at the join (or, if that
    /// one has not got there yet, another group) takes over.
    fn end_group(&mut self, k: usize, id: u64) {
        let waiting = self.parked[k]
            .iter()
            .position(|g| matches!(g.status, Status::Blocked(Wait::Join { id: j, .. }) if j == id));
        let i = match waiting {
            Some(i) => {
                let g = &mut self.parked[k][i];
                g.status = Status::Runnable;
                g.settled = false;
                i
            }
            None => self.parked[k].iter().position(|g| g.ready(self.round)).unwrap_or(0),
        };
        self.warps[k] = self.parked[k].remove(i);
    }
}

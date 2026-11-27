// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reports on blocked and exited threads: where each warp is, and the deadlock and hang errors built from them.

use super::*;

impl<'a> Machine<'a> {
    /// `file:line` of an instruction.
    pub(super) fn at(inst: &Inst) -> String {
        format!("{}:{}", inst.loc.file(), inst.loc.line())
    }

    /// `block (x,y,z)` for block rank `block`.
    pub(super) fn block_name(&self, block: usize) -> String {
        let [x, y, z] = self.blocks[block];
        format!("block ({x},{y},{z})")
    }

    /// The constructs warp `k` stands in, innermost first, from its frame
    /// stack: `, in for_range at k.rs:40, iteration 3; in the then-branch of
    /// if_ at k.rs:38` (empty at the top level of the kernel).
    fn context(&self, k: usize) -> String {
        self.group_context(k, &self.warps[k])
    }

    /// The frames group `group` of warp `k` stands in, outermost first: its
    /// own and, for an else group, those of the then group it split from
    /// below the divergent `If`.
    fn frames_of<'s>(&'s self, k: usize, group: &'s WarpState<'a>) -> Vec<&'s Frame<'a>> {
        let own = &group.groups[0].frames;
        let Some(id) = group.join() else { return own.iter().collect() };
        let split_at = |g: &'s WarpState<'a>| {
            g.groups[0]
                .frames
                .iter()
                .position(|f| matches!(f.kind, FrameKind::IfThen { split: Some(s), .. } if s == id))
        };
        for g in std::iter::once(&self.warps[k]).chain(&self.parked[k]) {
            if let Some(i) = split_at(g) {
                let mut frames = self.frames_of(k, g);
                // The then group's frames below the If, then its own.
                let below = frames.len() - (g.groups[0].frames.len() - i);
                frames.truncate(below);
                frames.extend(own.iter());
                return frames;
            }
        }
        own.iter().collect()
    }

    /// [`Self::context`] for group `group` of warp `k`.
    fn group_context(&self, k: usize, group: &WarpState<'a>) -> String {
        let mut parts: Vec<String> = Vec::new();
        for frame in self.frames_of(k, group).into_iter().rev() {
            let t = frame.lanes.first().copied();
            match &frame.kind {
                FrameKind::Body => {}
                FrameKind::IfThen { inst, .. } => {
                    parts.push(format!("in the then-branch of if_ at {}", Self::at(inst)))
                }
                FrameKind::IfElse { inst, .. } => {
                    parts.push(format!("in the else-branch of if_ at {}", Self::at(inst)))
                }
                FrameKind::ForBody { inst, ivs, .. } => {
                    let Op::For { lower, step, .. } = &inst.op else { unreachable!("a For") };
                    let iteration = t.map(|t| {
                        let lo = self.get(*lower, t) as u32 as i32 as i64;
                        let step = (self.get(*step, t) as u32 as i32 as i64).max(1);
                        (ivs[self.lane_of(t)] - lo) / step
                    });
                    parts.push(match iteration {
                        Some(i) => format!("in for_range at {}, iteration {i}", Self::at(inst)),
                        None => format!("in for_range at {}", Self::at(inst)),
                    });
                }
                FrameKind::WhileCond { inst, iterations, .. } | FrameKind::WhileBody { inst, iterations, .. } => {
                    parts.push(format!("in while_carry at {}, iteration {}", Self::at(inst), iterations - 1))
                }
                FrameKind::Call { inst } => {
                    let Op::Call { callee, .. } = &inst.op else { unreachable!("a Call") };
                    let symbol = &self.ir.extern_decl(*callee).symbol;
                    parts.push(format!("in the call of `{symbol}` at {}", Self::at(inst)))
                }
            }
        }
        if parts.is_empty() { String::new() } else { format!(", {}", parts.join("; ")) }
    }

    /// What blocked warp `k` waits at: `barrier 1 (k.rs:12)`.
    fn wait_site(&self, wait: &Wait) -> String {
        match wait {
            Wait::Barrier { inst, .. } if matches!(inst.op, Op::Barrier) => {
                format!("sync_threads() ({})", Self::at(inst))
            }
            Wait::Barrier { id, inst, .. } => format!("barrier {id} ({})", Self::at(inst)),
            Wait::Mbarrier { inst, .. } => format!("mbarrier.try_wait ({})", Self::at(inst)),
            Wait::Cluster { inst } => format!("the cluster barrier ({})", Self::at(inst)),
            Wait::Join { inst, .. } => format!("the end of if_ ({})", Self::at(inst)),
            Wait::SetMaxNReg { inst, .. } | Wait::Wgmma { inst, .. } | Wait::TmemAlloc { inst, .. } => {
                format!("{} ({})", inst.op.name(), Self::at(inst))
            }
            Wait::Rendezvous { inst } => {
                let name = match &inst.op {
                    Op::Shuffle { .. } => "shuffle",
                    Op::Vote { kind: VoteKind::All, .. } => "vote_all_sync",
                    Op::Vote { kind: VoteKind::Any, .. } => "vote_any_sync",
                    Op::Vote { kind: VoteKind::Uni, .. } => "vote_uni_sync",
                    Op::Vote { kind: VoteKind::Ballot, .. } => "vote_ballot_sync",
                    Op::Match { .. } => "match_sync",
                    Op::Redux { .. } => "warp_redux_sync",
                    _ => "sync_warp",
                };
                format!("{name} ({})", Self::at(inst))
            }
        }
    }

    /// ` (lanes 0..=15)` for a group that holds part of its warp (whose
    /// thread `t` is), else nothing.
    fn partial_lanes(&self, group: &WarpState, t: usize) -> String {
        let lanes = self.lane_set(group);
        if lanes.len() == self.warp_size(t - self.lane_of(t)) {
            String::new()
        } else {
            format!(" ({} {})", if lanes.len() == 1 { "lane" } else { "lanes" }, warp_list(&lanes))
        }
    }

    /// The lanes (0 to 31) of group `group`.
    fn lane_set(&self, group: &WarpState) -> Vec<usize> {
        group.lanes().iter().map(|&t| self.lane_of(t)).collect()
    }

    /// Where `lanes` of warp `k` are: `lanes 1..=31 wait at mbarrier.try_wait
    /// (k.rs:30), in the else-branch of if_ at k.rs:20`, by the groups that
    /// hold them; lanes no group holds wait to reconverge.
    fn lanes_whereabouts(&self, k: usize, lanes: &[usize]) -> String {
        let mut parts = Vec::new();
        let mut left: Vec<usize> = lanes.to_vec();
        for g in std::iter::once(&self.warps[k]).chain(&self.parked[k]) {
            let mine: Vec<usize> = self.lane_set(g).into_iter().filter(|l| lanes.contains(l)).collect();
            if mine.is_empty() {
                continue;
            }
            left.retain(|l| !mine.contains(l));
            let what = match &g.status {
                Status::Blocked(wait) => format!("wait at {}{}", self.wait_site(wait), self.group_context(k, g)),
                Status::Runnable if self.spins(g) => format!("spin{}", self.group_context(k, g)),
                Status::Runnable => format!("are still running{}", self.group_context(k, g)),
                Status::Exited => "have exited".into(),
            };
            let what =
                if mine.len() == 1 { what.replacen("wait ", "waits ", 1).replacen("are ", "is ", 1) } else { what };
            let what =
                if mine.len() == 1 && what.starts_with("spin") { what.replacen("spin", "spins", 1) } else { what };
            parts.push(format!("{} {} {what}", if mine.len() == 1 { "lane" } else { "lanes" }, warp_list(&mine)));
        }
        if !left.is_empty() {
            parts.push(format!(
                "{} {} took another path and {} at the end of a branch or loop for the others to reconverge, so they \
                 never get there",
                if left.len() == 1 { "lane" } else { "lanes" },
                warp_list(&left),
                if left.len() == 1 { "waits" } else { "wait" }
            ));
        }
        parts.join("; ")
    }

    /// Where warp `k` is, as the singular and plural predicate of a
    /// sentence about it (`has exited` / `have exited`).
    fn whereabouts(&self, k: usize) -> (String, String) {
        if !self.parked[k].is_empty() {
            let t = self.warps[k].lanes()[0];
            let all: Vec<usize> = (0..self.warp_size(t - self.lane_of(t))).collect();
            let site = format!("is divergent ({})", self.lanes_whereabouts(k, &all));
            return (site.clone(), site.replacen("is", "are", 1));
        }
        match &self.warps[k].status {
            Status::Exited => match self.exit_sites.get(&k) {
                Some(at) => {
                    let site = format!("at the exit at {}:{}", at.file(), at.line());
                    (format!("has exited {site}"), format!("have exited {site}"))
                }
                None => ("has exited".into(), "have exited".into()),
            },
            Status::Blocked(wait) => {
                let site = format!("at {}{}", self.wait_site(wait), self.context(k));
                (format!("waits {site}"), format!("wait {site}"))
            }
            Status::Runnable if self.spins(&self.warps[k]) => ("spins".into(), "spin".into()),
            Status::Runnable => ("is still running".into(), "are still running".into()),
        }
    }

    /// `warp 2 waits at ...; warps 0..=3 have exited`: where each of
    /// `warps` (by key) is, warps in the same place together. With
    /// `blocks`, each is named with its block.
    pub(super) fn whereabouts_of(&self, warps: &[usize], blocks: bool) -> String {
        let w = self.warps_per_block;
        let mut groups: Vec<((String, String, usize), Vec<usize>)> = Vec::new();
        for &k in warps {
            let (one, many) = self.whereabouts(k);
            let block = if blocks { k / w } else { 0 };
            match groups.iter_mut().find(|g| g.0.0 == one && g.0.2 == block) {
                Some(g) => g.1.push(k % w),
                None => groups.push(((one, many, block), vec![k % w])),
            }
        }
        groups
            .into_iter()
            .map(|((one, many, block), list)| {
                let of = if blocks { format!(" of {}", self.block_name(block)) } else { String::new() };
                if list.len() == 1 {
                    format!("warp {}{of} {one}", list[0])
                } else {
                    format!("warps {}{of} {many}", warp_list(&list))
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// How a report names the mbarrier keyed by `key`.
    fn mbarrier_name(&self, key: MbarrierKey) -> String {
        let place = smem_place(self.ir.buffer(BufferId(key.1)), key.2);
        if self.blocks.len() > 1 { format!("{place} of {}", self.block_name(key.0)) } else { place }
    }

    /// `the mbarrier m[0] for phase 2 (parity 0), which still waits for 1
    /// of 2 arrivals, ...`: the state of a phase that warps wait for.
    fn mbarrier_phase(&self, key: MbarrierKey, phase: u64) -> String {
        let name = self.mbarrier_name(key);
        let Some(m) = self.mbarriers.get(&key) else {
            return format!("the mbarrier {name}, which is not initialised");
        };
        if m.phase != phase {
            return format!("the mbarrier {name} for phase {phase} (parity {}), which has completed", phase & 1);
        }
        let mut what = format!(
            "the mbarrier {name} for phase {phase} (parity {}), which still waits for {} of its {} arrival(s)",
            phase & 1,
            m.arrivals_left,
            m.arrivals_per_phase
        );
        if !m.deferred.is_empty() {
            what += &format!(" ({} owed by cp.async.mbarrier.arrive)", m.deferred.len());
        }
        what += &format!(
            ", with a tx-count of {} byte(s) and {} bulk cop{} in flight",
            m.tx_count,
            m.copies.len(),
            if m.copies.len() == 1 { "y" } else { "ies" }
        );
        what
    }

    /// One line per blocked warp, warps in the same state together, and
    /// the location and thread of the first.
    fn blocked_report(&self) -> (Vec<String>, Option<(Loc, usize)>) {
        let w = self.warps_per_block;
        // (tag, the groups (warp, lanes) in that state, the rest of the line)
        type Line = (String, Vec<(usize, String)>, String);
        let mut lines: Vec<Line> = Vec::new();
        let mut first: Option<(Loc, usize)> = None;
        for (k, g) in self.all_groups() {
            let state = self.group(k, g);
            let Status::Blocked(wait) = state.status else { continue };
            let t = state.lanes().first().copied().unwrap_or((k / w) * self.threads_per_block + (k % w) * 32);
            let inst = match wait {
                Wait::Barrier { inst, .. }
                | Wait::Mbarrier { inst, .. }
                | Wait::Cluster { inst }
                | Wait::Join { inst, .. }
                | Wait::SetMaxNReg { inst, .. }
                | Wait::Wgmma { inst, .. }
                | Wait::TmemAlloc { inst, .. }
                | Wait::Rendezvous { inst } => inst,
            };
            first.get_or_insert((inst.loc, t));
            let ctx = self.group_context(k, state);
            let lanes = self.partial_lanes(state, t);
            let (block, rest) = match wait {
                Wait::Barrier { block, id, .. } => {
                    let s = &self.named[block][id];
                    let others: Vec<usize> = s.arrived.iter().copied().filter(|a| !s.waiting.contains(a)).collect();
                    let also = if others.is_empty() {
                        String::new()
                    } else {
                        format!(", and warps {} have arrived without waiting", warp_list(&others))
                    };
                    let missing: Vec<usize> =
                        (0..w).filter(|o| !s.arrived.contains(o)).map(|o| block * w + o).collect();
                    let exited = self.exited_warps(block);
                    let live = missing.len() - exited.len();
                    let mut rest = format!(
                        " at {}{ctx}, which needs {} warps{also}; {}",
                        self.wait_site(&wait),
                        s.expected,
                        self.whereabouts_of(&missing, false)
                    );
                    if !exited.is_empty() && s.arrived.len() + live < s.expected {
                        rest += &format!(
                            ". It can only complete through warps that have exited ({}): PTX may release a barrier \
                             for threads that exited, but that is almost always a bug",
                            warp_list(&exited)
                        );
                    }
                    (block, rest)
                }
                Wait::Mbarrier { key, phase, .. } => {
                    (k / w, format!(" at {}{ctx} on {}", self.wait_site(&wait), self.mbarrier_phase(key, phase)))
                }
                Wait::Cluster { .. } => {
                    let c = (k / w) / self.cluster_size;
                    let phase = self.cluster_barriers[c].phase;
                    let mut missing: Vec<usize> = Vec::new();
                    let mut names: Vec<String> = Vec::new();
                    for o in c * self.cluster_size * w..(c + 1) * self.cluster_size * w {
                        if matches!(self.warps[o].status, Status::Exited) {
                            continue;
                        }
                        let start = (o / w) * self.threads_per_block + (o % w) * 32;
                        let lanes: Vec<usize> = (0..self.warp_size(start))
                            .filter(|&l| self.cluster_arrival[start + l] != Some(phase))
                            .collect();
                        if lanes.is_empty() {
                            continue;
                        }
                        missing.push(o);
                        let partial = if lanes.len() == self.warp_size(start) {
                            String::new()
                        } else {
                            format!(" (lanes {})", warp_list(&lanes))
                        };
                        names.push(format!("warp {}{partial} of {}", o % w, self.block_name(o / w)));
                    }
                    (
                        usize::MAX,
                        format!(
                            " at {}{ctx} for phase {phase}, which {} of the cluster have not arrived at ({})",
                            self.wait_site(&wait),
                            names.join(", "),
                            self.whereabouts_of(&missing, true)
                        ),
                    )
                }
                Wait::SetMaxNReg { block, group, .. } => (block, self.setmaxnreg_wait(block, group, &ctx)),
                Wait::Wgmma { block, group, .. } => (block, self.wgmma_wait_report(block, group, &ctx)),
                Wait::TmemAlloc { block, .. } => (block, self.tmem_wait_report(block, &ctx)),
                Wait::Join { .. } => {
                    let mine = self.lane_set(state);
                    let others: Vec<usize> =
                        (0..self.warp_size(t - self.lane_of(t))).filter(|l| !mine.contains(l)).collect();
                    let rest = format!(
                        " at {}{ctx} for the lanes of its else-branch: {}",
                        self.wait_site(&wait),
                        self.lanes_whereabouts(k, &others)
                    );
                    (k / w, rest)
                }
                Wait::Rendezvous { .. } => {
                    let (Op::Shuffle { mask, .. }
                    | Op::Vote { mask, .. }
                    | Op::Match { mask, .. }
                    | Op::Redux { mask, .. }
                    | Op::WarpBarrier { mask }) = &inst.op
                    else {
                        unreachable!("a .sync warp collective")
                    };
                    let mine = self.lane_set(state);
                    let needed = state.lanes().iter().fold(0u32, |acc, &t| acc | self.get(*mask, t) as u32);
                    let missing: Vec<usize> = (0..self.warp_size(t - self.lane_of(t)))
                        .filter(|&l| needed >> l & 1 == 1 && !mine.contains(&l))
                        .filter(|&l| {
                            !self.parked[k].iter().chain(std::iter::once(&self.warps[k])).any(|o| {
                                matches!(o.status, Status::Blocked(Wait::Rendezvous { inst: at }) if same_collective(at, inst))
                                    && self.lane_set(o).contains(&l)
                            })
                        })
                        .collect();
                    let rest = format!(
                        " at {}{ctx} with member mask {needed:#010x}, for lanes {} of the mask to reach it (every lane \
                         of the mask must execute it; lanes of other branches may, with independent thread \
                         scheduling): {}",
                        self.wait_site(&wait),
                        warp_list(&missing),
                        self.lanes_whereabouts(k, &missing)
                    );
                    (k / w, rest)
                }
            };
            let tag = format!("{block}{rest}");
            match lines.iter_mut().find(|l| l.0 == tag) {
                Some(l) => l.1.push((k, lanes)),
                None => lines.push((tag, vec![(k, lanes)], rest)),
            }
        }
        let out = lines
            .into_iter()
            .map(|(_, warps, rest)| {
                let block = warps[0].0 / w;
                let cluster_wide = warps.iter().any(|&(k, _)| k / w != block);
                let partial = warps.iter().any(|(_, lanes)| !lanes.is_empty());
                let subject = if cluster_wide || partial {
                    let of =
                        |k: usize| if cluster_wide { format!(" of {}", self.block_name(k / w)) } else { String::new() };
                    let list = warps
                        .iter()
                        .map(|(k, lanes)| format!("warp {}{lanes}{}", k % w, of(*k)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if cluster_wide { list } else { format!("{list} of {}", self.block_name(block)) }
                } else if warps.len() == 1 {
                    format!("warp {} of {}", warps[0].0 % w, self.block_name(block))
                } else {
                    let list: Vec<usize> = warps.iter().map(|(k, _)| k % w).collect();
                    format!("warps {} of {}", warp_list(&list), self.block_name(block))
                };
                let verb = if warps.len() == 1 { "waits" } else { "wait" };
                format!("{subject} {verb}{rest}")
            })
            .collect();
        (out, first)
    }

    /// For hang and deadlock reports, the warps that
    /// executed `exit`, and where.
    fn exit_report(&self) -> Vec<String> {
        let w = self.warps_per_block;
        let mut sites: Vec<(&usize, &Loc)> = self.exit_sites.iter().collect();
        sites.sort_by_key(|e| *e.0);
        sites
            .into_iter()
            .map(|(&k, at)| {
                format!(
                    "warp {} of {} exited at the exit at {}:{}: the arrivals and stores after it never happen",
                    k % w,
                    self.block_name(k / w),
                    at.file(),
                    at.line()
                )
            })
            .collect()
    }

    /// The report when no warp can run and some are blocked: each blocked
    /// warp, where it waits and what it waits for, and where the warps it
    /// waits for are.
    pub(super) fn deadlock(&self) -> SimError {
        let (mut lines, first) = self.blocked_report();
        lines.extend(self.exit_report());
        let (loc, t) = first.expect("a deadlock has a blocked warp");
        SimError {
            loc: Some(loc),
            ..self.err(
                format!(
                    "the kernel did not terminate: every warp that has not exited is blocked (a deadlock):\n  - {}",
                    lines.join("\n  - ")
                ),
                Some(t),
            )
        }
    }

    /// The report when nothing visible has happened for
    /// [`SimOptions::max_idle_steps`] steps and every warp that has not
    /// exited is blocked or spinning: the spinning warps, their loops and
    /// what they poll, then the blocked warps.
    pub(super) fn hang(&self) -> SimError {
        let w = self.warps_per_block;
        let mut lines: Vec<String> = Vec::new();
        let mut first: Option<(Loc, usize)> = None;
        let mut polls_global = false;
        for (k, g) in self.all_groups() {
            let state = self.group(k, g);
            if !matches!(state.status, Status::Runnable) {
                continue;
            }
            let t = state.lanes().first().copied().unwrap_or((k / w) * self.threads_per_block + (k % w) * 32);
            let polled = match state.poll {
                Some(Poll::Mbarrier { key, phase, inst }) => {
                    first.get_or_insert((inst.loc, t));
                    format!(
                        ", polling with mbarrier.test_wait ({}) {}",
                        Self::at(inst),
                        self.mbarrier_phase(key, phase)
                    )
                }
                Some(Poll::Load { buffer, block, byte, inst }) => {
                    first.get_or_insert((inst.loc, t));
                    let d = self.ir.buffer(buffer);
                    polls_global |= block == GLOBAL;
                    let place = if block == GLOBAL {
                        format!("{}[{}]", d.name, byte / d.elem.bytes())
                    } else {
                        smem_place(d, byte)
                    };
                    format!(", polling {place} with an ordered load ({})", Self::at(inst))
                }
                None => ", without polling memory".to_string(),
            };
            let lanes = self.partial_lanes(state, t);
            lines.push(format!(
                "warp {}{lanes} of {} spins{}{polled}",
                k % w,
                self.block_name(k / w),
                self.group_context(k, state)
            ));
        }
        let (blocked, first_blocked) = self.blocked_report();
        lines.extend(blocked);
        lines.extend(self.exit_report());
        let (loc, t) = first.or(first_blocked).unwrap_or((self.loc.expect("a step ran"), 0));
        let mut message = format!(
            "the kernel did not terminate: nothing visible has happened for {} warp steps and every warp that has not \
             exited is blocked or spinning (a hang):\n  - {}",
            self.opts.max_idle_steps,
            lines.join("\n  - ")
        );
        if polls_global && let Some(hint) = self.residency_hint() {
            message += &hint;
        }
        SimError { loc: Some(loc), ..self.err(message, Some(t)) }
    }

    /// For hang reports: the clusters of the grid that have not started, if
    /// any, which a spin on global memory may be waiting for.
    fn residency_hint(&self) -> Option<String> {
        if self.queued.is_empty() {
            return None;
        }
        Some(format!(
            "\n{} cluster(s) of the grid have not started: the simulator runs {} cluster(s) at a time \
             (SimOptions::resident_clusters), and a cluster that waits for one that has not started waits forever. \
             If the kernel needs its clusters to run together (a grid-wide barrier), launch it as a cooperative \
             launch (LaunchConfig::with_cooperative(true)), which the GPU runs only with every block resident, or \
             set resident_clusters to ResidentClusters::Grid or a count",
            self.queued.len(),
            self.cluster_ids.len()
        ))
    }
}

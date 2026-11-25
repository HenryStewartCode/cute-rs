// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's tensor maps in memory:
//! `tensormap.replace`, `tensormap.cp_fenceproxy` and the tensormap-proxy
//! fences, modelled from the PTX ISA.
//!
//! * **Encoding.** A descriptor in memory is the simulator's own 128-byte
//!   encoding ([`TmaDesc::to_sim_bytes`]), as in a descriptor parameter, so a
//!   copy (plain loads and stores), a replace and a TMA read round-trip.
//!   `tensormap.replace` decodes the object, changes the field and writes all
//!   128 bytes back (the PTX ISA treats it as a weak write of the whole
//!   object, in the generic proxy). Bytes that do not decode are reported
//!   where they are read as a tensor map.
//! * **The tensormap proxy.** TMA reads a descriptor through the tensormap
//!   proxy, a third proxy beside the generic and async ones. Each generic
//!   write to the bytes of a descriptor in global memory during the launch
//!   is recorded per writing warp (stores, atomics, `tensormap.replace`,
//!   `tensormap.cp_fenceproxy`'s copy). A TMA copy through the descriptor
//!   then needs, for every such write, a release by the writing warp after
//!   it (`fence.proxy.tensormap::generic.release`, or the release of the
//!   `tensormap.cp_fenceproxy` that made it) and an acquire of the
//!   descriptor's address by the issuing thread
//!   (`fence.proxy.tensormap::generic.acquire`) that the release happens
//!   before (in the issuing warp's program order, or through the vector
//!   clocks of the cluster). Writes by another cluster cannot be ordered by
//!   the simulator's clocks: for them a release and an acquire suffice.
//!   Prefetches are hints and are not checked. A descriptor written during
//!   the launch is checked as `cuTensorMapEncodeTiled` would check it when a
//!   copy uses it.
//! * **Descriptors in use.** A TMA store reads its tensor map until its
//!   bulk async-group's reads complete (`cp.async.bulk.wait_group.read`, the
//!   wait `TensorMapManager::update_tensormap` makes before it rewrites a
//!   descriptor in global memory): a generic write to the tensor map before
//!   then is reported. The writes are recorded per warp, so a release must
//!   come from the warp that wrote (the PTX ISA's release is cumulative and
//!   would also cover writes ordered before it from other warps: the
//!   simulator reports those, conservatively). Descriptors written before
//!   the launch (by the host) are not checked for an acquire, although the
//!   PTX memory model asks for one.

use std::collections::HashMap;
use std::rc::Rc;

use super::{Machine, R};
use crate::ir::{BufferId, KernelIr, MemSpace, Op, ValueId};
use crate::nvgpu::tensormap::TensormapField;
use crate::tensor::AddressSpace;
use crate::tma::{TENSOR_MAP_ALIGN, TENSOR_MAP_BYTES, TmaDesc};

/// The granularity of the write records: half a tensor map, the alignment
/// TMA requires of one.
const CHUNK: usize = TENSOR_MAP_ALIGN;

/// A warp's generic writes to a chunk of global memory.
#[derive(Clone)]
struct Writer {
    /// The id of the writer's cluster (clocks are per cluster).
    cluster: u64,
    /// The warp's key and its block and warp index, for messages.
    k: usize,
    block: [u32; 3],
    warp: usize,
    /// The warp's own time at its last write.
    at: u32,
    /// Its own time at the release after that write, if any.
    released: Option<u32>,
}

/// An acquire of a chunk by a thread: its cluster and its warp's clock then.
struct Acquire {
    cluster: u64,
    chunk: (u32, usize),
    clock: Rc<[u32]>,
}

pub(super) struct State {
    /// Whether the kernel has TMA instructions or tensor-map operations;
    /// without them nothing is recorded.
    pub(super) enabled: bool,
    writes: HashMap<(u32, usize), Vec<Writer>>,
    /// By thread.
    acquires: HashMap<usize, Vec<Acquire>>,
    /// TMA stores issued through a tensor map in global memory: the async
    /// operation, and the tensor map's buffer and byte. The store reads the
    /// tensor map until its bulk async-group's reads complete
    /// (`cp.async.bulk.wait_group.read`), so a generic write to it before
    /// then is a race.
    store_maps: Vec<(usize, BufferId, usize)>,
}

impl State {
    pub(super) fn new(ir: &KernelIr) -> State {
        let enabled = ir.count_ops(|op| {
            matches!(
                op,
                Op::TmaLoad { .. }
                    | Op::TmaStore { .. }
                    | Op::TensormapReplace { .. }
                    | Op::FenceTensormapRelease
                    | Op::FenceTensormapAcquire { .. }
                    | Op::TensormapCpFenceproxy { .. }
            )
        }) > 0;
        State { enabled, writes: HashMap::new(), acquires: HashMap::new(), store_maps: Vec::new() }
    }

    pub(super) fn reset(&mut self) {
        self.writes.clear();
        self.acquires.clear();
        self.store_maps.clear();
    }

    /// The simulator forgot its asynchronous operations (between clusters).
    pub(super) fn forget_ops(&mut self) {
        self.store_maps.clear();
    }
}

impl<'a> Machine<'a> {
    fn tmap_cluster(&self, t: usize) -> u64 {
        self.cluster_ids[self.block_rank(t) / self.cluster_size]
    }

    /// TMA store `id` by thread `t` read the tensor map at `desc` (a generic
    /// address): if it is in global memory, remember it until the store's
    /// reads complete.
    pub(super) fn tmap_note_store(&mut self, id: usize, desc: ValueId, t: usize) {
        if let Ok((b, byte)) = self.resolve(AddressSpace::Generic, self.get(desc, t))
            && self.ir.buffer(b).space == MemSpace::Global
        {
            self.tmaps.store_maps.push((id, b, byte));
        }
    }

    /// A generic write by thread `t` of `size` bytes at `byte` of global
    /// buffer `b`: an error if a TMA store still reads a tensor map there.
    pub(super) fn tmap_note_write(&mut self, b: BufferId, byte: usize, size: usize, t: usize) -> R {
        if !self.tmaps.store_maps.is_empty() {
            let ops = &self.ops;
            self.tmaps
                .store_maps
                .retain(|&(id, ..)| ops.get(id).is_some_and(|op| op.in_flight() && !op.reads.is_empty()));
            if let Some(&(id, _, at)) = self
                .tmaps
                .store_maps
                .iter()
                .find(|&&(_, mb, at)| mb == b && at < byte + size && byte < at + TENSOR_MAP_BYTES)
            {
                let op = &self.ops[id];
                return Err(self.err(
                    format!(
                        "a generic write to the tensor map at byte {at} of `{}`, which a TMA store issued by thread {} \
                         of {} still reads: wait for the store's bulk async-group to read its source \
                         (cp_async_bulk_commit_group, then cp_async_bulk_wait_group(0, true) by the issuing thread, \
                         then sync_warp or a barrier) before rewriting a descriptor it uses, as \
                         TensorMapManager::update_tensormap does",
                        self.ir.buffer(b).name,
                        op.thread % self.threads_per_block,
                        self.block_name(self.block_rank(op.thread))
                    ),
                    Some(t),
                ));
            }
        }
        let cluster = self.tmap_cluster(t);
        let k = self.clock_index(t);
        let at = self.clocks[k][self.slot_of(k)];
        let (block, warp) = (self.block_of(t), k % self.warps_per_block);
        for c in byte / CHUNK..=(byte + size.max(1) - 1) / CHUNK {
            let list = self.tmaps.writes.entry((b.0, c)).or_default();
            match list.iter_mut().find(|w| w.cluster == cluster && w.k == k) {
                Some(w) => {
                    w.at = w.at.max(at);
                    w.released = None;
                }
                None => list.push(Writer { cluster, k, block, warp, at, released: None }),
            }
        }
        Ok(())
    }

    /// Release warp `k`'s writes (to `chunks` only, if given) to the
    /// tensormap proxy at its current time, which then advances.
    fn tmap_release_warp(&mut self, k: usize, cluster: u64, chunks: Option<&[(u32, usize)]>) {
        let slot = self.slot_of(k);
        let now = self.clocks[k][slot];
        for (key, list) in self.tmaps.writes.iter_mut() {
            if chunks.is_some_and(|c| !c.contains(key)) {
                continue;
            }
            for w in list.iter_mut().filter(|w| w.cluster == cluster && w.k == k && w.released.is_none()) {
                w.released = Some(now);
            }
        }
        self.clocks[k][slot] += 1;
    }

    /// `fence.proxy.tensormap::generic.release.gpu` by the lanes `active`.
    pub(super) fn tmap_release(&mut self, active: &[usize]) {
        let mut done = Vec::new();
        for &t in active {
            let k = self.clock_index(t);
            if !done.contains(&k) {
                done.push(k);
                let cluster = self.tmap_cluster(t);
                self.tmap_release_warp(k, cluster, None);
            }
        }
    }

    /// The chunks of the tensor map at byte `byte` of buffer `b`.
    fn tmap_chunks(b: BufferId, byte: usize) -> Vec<(u32, usize)> {
        (byte / CHUNK..=(byte + TENSOR_MAP_BYTES - 1) / CHUNK).map(|c| (b.0, c)).collect()
    }

    /// `fence.proxy.tensormap::generic.acquire.gpu [addr], 128` by the lanes `active`.
    pub(super) fn tmap_acquire(&mut self, addr: ValueId, active: &[usize]) -> R {
        let what = "fence.proxy.tensormap::generic.acquire (fence_tma_desc_acquire)";
        for &t in active {
            let a = self.get(addr, t);
            let (b, byte) = self
                .resolve(AddressSpace::Generic, a)
                .map_err(|why| self.err(format!("{what}: the tensor map's address: {why}"), Some(t)))?;
            if !matches!(self.ir.buffer(b).space, MemSpace::Global | MemSpace::Constant) {
                return Err(self.err(
                    format!("{what}: {a:#x} is not a global address; tensor maps that TMA reads are in global memory"),
                    Some(t),
                ));
            }
            let k = self.clock_index(t);
            let clock: Rc<[u32]> = Rc::from(self.clocks[k].as_slice());
            let cluster = self.tmap_cluster(t);
            let list = self.tmaps.acquires.entry(t).or_default();
            list.retain(|q| q.cluster == cluster);
            for chunk in Self::tmap_chunks(b, byte) {
                list.push(Acquire { cluster, chunk, clock: clock.clone() });
            }
        }
        Ok(())
    }

    /// Before thread `t` reads the tensor map at byte `byte` of global
    /// buffer `b` through the tensormap proxy (`what`): every generic write
    /// to it during the launch was released and acquired, in that order.
    /// Returns whether any was written during the launch.
    pub(super) fn tmap_check_read(&self, b: BufferId, byte: usize, t: usize, what: &str) -> R<bool> {
        let cluster = self.tmap_cluster(t);
        let none = Vec::new();
        let acquires = self.tmaps.acquires.get(&t).unwrap_or(&none);
        let name = &self.ir.buffer(b).name;
        let mut written = false;
        for chunk in Self::tmap_chunks(b, byte) {
            let Some(list) = self.tmaps.writes.get(&chunk) else { continue };
            for w in list {
                written = true;
                let who = if w.cluster == cluster && self.block_of(t) == w.block {
                    format!("warp {}", w.warp)
                } else {
                    format!("warp {} of block {:?}", w.warp, w.block)
                };
                let Some(f) = w.released else {
                    return Err(self.err(
                        format!(
                            "{what} reads the tensor map at byte {byte} of `{name}` through the tensormap proxy, but \
                             {who} wrote it through the generic proxy without releasing it since: call \
                             fence_tma_desc_release() (fence.proxy.tensormap::generic.release) after the writes, or \
                             publish it from shared memory with cp_fence_tma_desc_release"
                        ),
                        Some(t),
                    ));
                };
                let mine: Vec<&Acquire> =
                    acquires.iter().filter(|q| q.cluster == cluster && q.chunk == chunk).collect();
                if mine.is_empty() {
                    return Err(self.err(
                        format!(
                            "{what} reads the tensor map at byte {byte} of `{name}` through the tensormap proxy, but \
                             this thread has not acquired it since {who} wrote and released it: call \
                             fence_tma_desc_acquire(ptr) (fence.proxy.tensormap::generic.acquire) on its address \
                             before the copy"
                        ),
                        Some(t),
                    ));
                }
                let ordered = w.cluster != cluster || mine.iter().any(|q| q.clock[self.slot_of(w.k)] > f);
                if !ordered {
                    return Err(self.err(
                        format!(
                            "{what} reads the tensor map at byte {byte} of `{name}`, which {who} wrote and released \
                             (fence.proxy.tensormap::generic.release), but this thread's acquire \
                             (fence_tma_desc_acquire) is not ordered after that release: order the release before \
                             the acquire (a barrier between them, or both in this warp, release first)"
                        ),
                        Some(t),
                    ));
                }
            }
        }
        Ok(written)
    }

    /// The 128 bytes at byte `byte` of buffer `b`, read by thread `t`.
    fn tmap_read_bytes(&mut self, b: BufferId, byte: usize, t: usize) -> R<Vec<u8>> {
        let mut bytes = Vec::with_capacity(TENSOR_MAP_BYTES);
        for i in 0..TENSOR_MAP_BYTES / 8 {
            bytes.extend_from_slice(&self.read_as(b, byte + 8 * i, 8, t, false)?.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Write the 128 bytes of a tensor map at byte `byte` of buffer `b`, by thread `t`.
    fn tmap_write_bytes(&mut self, b: BufferId, byte: usize, bytes: &[u8], t: usize) -> R {
        for (i, w) in bytes.chunks(8).enumerate() {
            self.write_as(b, byte + 8 * i, 8, t, u64::from_le_bytes(w.try_into().unwrap()), false)?;
        }
        Ok(())
    }

    /// The buffer and byte of a tensor map at `addr` in `space` (`what`):
    /// in global memory, or in shared memory if `shared`, aligned and whole.
    fn tmap_locate(&self, space: AddressSpace, addr: u64, t: usize, what: &str, shared: bool) -> R<(BufferId, usize)> {
        let (b, byte) = self
            .resolve(space, addr)
            .map_err(|why| self.err(format!("{what}: the tensor map's address: {why}"), Some(t)))?;
        let ok = match self.ir.buffer(b).space {
            MemSpace::Global => true,
            MemSpace::Shared => shared,
            _ => false,
        };
        if !ok {
            return Err(self.err(
                format!(
                    "{what}: the address {addr:#x} is in {} memory; the tensor map must be in global{} memory",
                    AddressSpace::from(self.ir.buffer(b).space),
                    if shared { " or shared::cta" } else { "" }
                ),
                Some(t),
            ));
        }
        if !addr.is_multiple_of(TENSOR_MAP_ALIGN as u64) {
            return Err(self
                .err(format!("{what}: the tensor map at {addr:#x} is not {TENSOR_MAP_ALIGN}-byte aligned"), Some(t)));
        }
        if byte + TENSOR_MAP_BYTES > self.buffer_bytes(b) {
            return Err(self.err(
                format!(
                    "{what}: the tensor map at byte {byte} of `{}` runs past its end ({} bytes)",
                    self.ir.buffer(b).name,
                    self.buffer_bytes(b)
                ),
                Some(t),
            ));
        }
        Ok((b, byte))
    }

    /// `tensormap.replace` by the lanes `active`.
    pub(super) fn tmap_replace(
        &mut self,
        addr: ValueId,
        space: AddressSpace,
        field: TensormapField,
        value: ValueId,
        active: &[usize],
    ) -> R {
        let what = "tensormap.replace (update_tma_descriptor, tensormap_replace)";
        for &t in active {
            let a = self.get(addr, t);
            let a = if space == AddressSpace::Smem { a as u32 as u64 } else { a };
            let (b, byte) = self.tmap_locate(space, a, t, what, true)?;
            let bytes = self.tmap_read_bytes(b, byte, t)?;
            let mut d = TmaDesc::from_sim_bytes(&bytes).map_err(|e| {
                self.err(
                    format!(
                        "{what}: the 128 bytes at byte {byte} of `{}` do not hold a tensor map ({e}); copy one there \
                         first (copy_tensormap)",
                        self.ir.buffer(b).name
                    ),
                    Some(t),
                )
            })?;
            let v = self.get(value, t);
            match field {
                TensormapField::GlobalAddress => d.global_address = v,
                TensormapField::GlobalDim(o) => d.global_dims[o as usize] = u64::from(v as u32),
                TensormapField::GlobalStride(o) => d.global_strides[o as usize] = v,
            }
            self.tmap_write_bytes(b, byte, &d.to_sim_bytes(), t)?;
        }
        Ok(())
    }

    /// `tensormap.cp_fenceproxy...sync.aligned [dst], [src], 128` by the lanes
    /// `active`: whole warps, each copying once.
    pub(super) fn tmap_cp_fenceproxy(&mut self, dst: ValueId, src: ValueId, active: &[usize]) -> R {
        let what = "tensormap.cp_fenceproxy (cp_fence_tma_desc_release)";
        let bases = self.full_warps(active, &format!("{what}, which is .sync.aligned,"))?;
        for base in bases {
            let lanes = base..base + 32;
            let (g, s) = (self.get(dst, base), self.get(src, base));
            if let Some(t) = lanes.clone().find(|&t| self.get(dst, t) != g || self.get(src, t) != s) {
                return Err(self.err(
                    format!("{what}: the lanes of the warp give different addresses; .aligned needs the same ones"),
                    Some(t),
                ));
            }
            let (sb, sbyte) = self.tmap_locate(AddressSpace::Smem, s as u32 as u64, base, what, true)?;
            if self.ir.buffer(sb).space != MemSpace::Shared {
                return Err(self.err(format!("{what}: the source is not in shared memory"), Some(base)));
            }
            let (gb, gbyte) = self.tmap_locate(AddressSpace::Gmem, g, base, what, false)?;
            let bytes = self.tmap_read_bytes(sb, sbyte, base)?;
            self.tmap_write_bytes(gb, gbyte, &bytes, base)?;
            let k = self.clock_index(base);
            let cluster = self.tmap_cluster(base);
            self.tmap_release_warp(k, cluster, Some(&Self::tmap_chunks(gb, gbyte)));
        }
        Ok(())
    }
}

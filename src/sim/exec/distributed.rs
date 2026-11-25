//! The simulator's external calls and multimem instructions.
//!
//! A multicast buffer stands for a group of replicas, one per simulated GPU
//! ([`MulticastReplicas`](crate::sim::MulticastReplicas)): the local replica is the launch's
//! own storage of the buffer, or of the unicast parameter that aliases it, and the other
//! replicas are held here. `multimem.ld_reduce` reads every replica and adds them in replica
//! order (the hardware's order is unspecified, so tests use sums that are exact in any order),
//! `multimem.st` writes every replica, and `multimem.red` adds atomically to every replica,
//! its release ordering applying to the local one (the only one other simulated threads see).

use super::{Machine, Poll, Publish, R, read_le, write_le};
use crate::ir::{AsmInput, BufferId, DType, ExternId, Inst, MemoryOrder, MultimemOp, MultimemType, Op, ValueId};
use crate::numeric::{NarrowRounding, decode, encode};
use crate::value::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits};

/// A multicast buffer's replicas during a launch.
pub(in crate::sim) struct McGroup {
    pub(in crate::sim) buffer: BufferId,
    /// The rank of the local replica.
    pub(in crate::sim) local: usize,
    /// The unicast buffer whose storage is the local replica, if one aliases it.
    pub(in crate::sim) alias: Option<BufferId>,
    /// Every replica, by rank; the local one's entry is unused during the launch.
    pub(in crate::sim) replicas: Vec<Vec<u8>>,
}

/// One 32-bit word of `multimem.ld_reduce .add` of type `ty` over `words` (one per replica,
/// in replica order): each element is accumulated at the `.acc` precision (Float32 for the
/// halves, Float16 for FP8, Float32 for Float32), rounding to nearest even after every
/// addition, then rounded to the element type (nearest even; FP8 without saturation).
pub fn ld_reduce_word(ty: MultimemType, words: &[u32]) -> u32 {
    match ty {
        MultimemType::F32 => {
            let sum = words.iter().skip(1).fold(f32::from_bits(words[0]), |a, &w| a + f32::from_bits(w));
            sum.to_bits()
        }
        MultimemType::F16x2 | MultimemType::Bf16x2 => {
            let f16 = ty == MultimemType::F16x2;
            let dec = |h: u16| if f16 { f16_bits_to_f32(h) } else { bf16_bits_to_f32(h) };
            let enc = |x: f32| if f16 { f32_to_f16_bits(x) } else { f32_to_bf16_bits(x) };
            let half = |shift: u32| {
                let part = |w: u32| dec((w >> shift) as u16);
                let sum = words.iter().skip(1).fold(part(words[0]), |a, &w| a + part(w));
                (enc(sum) as u32) << shift
            };
            half(0) | half(16)
        }
        MultimemType::E4m3x4 | MultimemType::E5m2x4 => {
            let (elem, _) = ty.element();
            let f8 = elem.float_format().expect("an FP8 format");
            let f16 = DType::F16.float_format().expect("Float16");
            let byte = |shift: u32| {
                let part = |w: u32| decode(f8, (w >> shift) & 0xff);
                // Float16 accumulation: each partial sum rounded to Float16 (exact in f64).
                let mut acc = part(words[0]);
                for &w in &words[1..] {
                    acc = decode(f16, encode(f16, acc + part(w), NarrowRounding::NearestEven, false, false));
                }
                encode(f8, acc, NarrowRounding::NearestEven, false, false) << shift
            };
            byte(0) | byte(8) | byte(16) | byte(24)
        }
    }
}

impl<'a> Machine<'a> {
    /// Bind the multicast groups of a launch.
    pub(in crate::sim) fn set_multicast(&mut self, groups: Vec<McGroup>) {
        self.multicast = groups;
    }

    /// The multicast groups after the launch, with each local replica copied in.
    pub(in crate::sim) fn take_multicast(&mut self) -> Vec<McGroup> {
        let mut groups = std::mem::take(&mut self.multicast);
        for g in &mut groups {
            let home = g.alias.unwrap_or(g.buffer);
            if let Some(bytes) = self.globals[home.0 as usize].as_ref()
                && let Some(slot) = g.replicas.get_mut(g.local)
            {
                *slot = bytes.clone();
            }
        }
        groups
    }

    /// The number of replicas of multicast buffer `b` (1 without a group), the local one's
    /// rank, and the buffer holding the local replica.
    fn replicas_of(&self, b: BufferId) -> (usize, usize, BufferId) {
        match self.multicast.iter().find(|g| g.buffer == b) {
            Some(g) => (g.replicas.len(), g.local, g.alias.unwrap_or(b)),
            None => (1, 0, b),
        }
    }

    fn remote(&mut self, b: BufferId, rank: usize) -> &mut Vec<u8> {
        let g = self.multicast.iter_mut().find(|g| g.buffer == b).expect("a remote replica belongs to a group");
        &mut g.replicas[rank]
    }

    /// Execute a multimem instruction for the `active` threads, one after another.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn multimem(
        &mut self,
        active: &[usize],
        op: &MultimemOp,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        values: &[ValueId],
        results: &[ValueId],
    ) -> R {
        let bytes = op.bytes();
        let (n, local, home) = self.replicas_of(buffer);
        let what = match op {
            MultimemOp::LdReduce { .. } => "multimem.ld_reduce",
            MultimemOp::St { .. } => "multimem.st",
            MultimemOp::RedAdd1 { .. } => "multimem.red",
        };
        let release = matches!(op, MultimemOp::RedAdd1 { order: MemoryOrder::Release, .. });
        let mut published = Vec::new();
        if release {
            for &t in active {
                let k = self.clock_index(t);
                if !published.iter().any(|p: &(usize, std::rc::Rc<[u32]>)| p.0 == k) {
                    published.push((k, self.clocks[k].as_slice().into()));
                }
            }
        }
        for &t in active {
            let idx = self.index(index, t);
            // Bounds and alignment: the access must be aligned to its whole size.
            let start = self.checked_range(buffer, idx, 1, elem.bytes(), elem.bytes(), t, what)?;
            if !start.is_multiple_of(bytes) {
                return Err(self.err(
                    format!(
                        "misaligned {what}: element {idx} of `{}` (byte {start}) is not {bytes}-byte aligned, as \
                         a {bytes}-byte multimem access must be",
                        self.ir.buffer(buffer).name
                    ),
                    Some(t),
                ));
            }
            if start + bytes > self.buffer_bytes(buffer) {
                return Err(self.err(
                    format!(
                        "{what} of {bytes} bytes at byte {start} runs past the end of `{}`",
                        self.ir.buffer(buffer).name
                    ),
                    Some(t),
                ));
            }
            let size = self.buffer_bytes(home);
            for rank in 0..n {
                if rank != local && self.remote(buffer, rank).len() < size {
                    return Err(self.err(
                        format!(
                            "replica {rank} of multicast buffer `{}` holds fewer bytes than the local one",
                            self.ir.buffer(buffer).name
                        ),
                        Some(t),
                    ));
                }
            }
            match *op {
                MultimemOp::LdReduce { ty, regs } => {
                    for (k, r) in results.iter().enumerate().take(regs as usize) {
                        let at = start + 4 * k;
                        let mut words = Vec::with_capacity(n);
                        for rank in 0..n {
                            words.push(if rank == local {
                                self.read_as(home, at, 4, t, false)? as u32
                            } else {
                                read_le(&self.remote(buffer, rank)[at..at + 4]) as u32
                            });
                        }
                        self.set(*r, t, ld_reduce_word(ty, &words) as u64);
                    }
                }
                MultimemOp::St { .. } => {
                    for (k, v) in values.iter().enumerate() {
                        let at = start + 4 * k;
                        let bits = self.get(*v, t) & 0xffff_ffff;
                        for rank in 0..n {
                            if rank == local {
                                self.write_as(home, at, 4, t, bits, false)?;
                            } else {
                                write_le(&mut self.remote(buffer, rank)[at..at + 4], bits);
                            }
                        }
                    }
                }
                MultimemOp::RedAdd1 { .. } => {
                    for rank in 0..n {
                        if rank == local {
                            self.atomic_update(home, start, 4, t, |old| (old as u32).wrapping_add(1) as u64)?;
                            let how = if release {
                                let k = self.clock_index(t);
                                Publish::ReleaseRmw(published.iter().find(|p| p.0 == k).expect("published").1.clone())
                            } else {
                                Publish::Rmw
                            };
                            self.publish(home, start, 4, 4, t, how);
                        } else {
                            let slot = &mut self.remote(buffer, rank)[start..start + 4];
                            let old = read_le(slot) as u32;
                            write_le(slot, old.wrapping_add(1) as u64);
                        }
                    }
                }
            }
        }
        for (k, _) in published {
            let slot = self.slot_of(k);
            self.clocks[k][slot] += 1;
        }
        Ok(())
    }

    /// After a compare-and-swap by the `active` threads: those whose comparison failed polled
    /// the location (a spin on a failing CAS counts toward a hang, as a spin on an ordered load
    /// does). The CAS's result already holds the old value.
    pub(super) fn note_failed_cas(
        &mut self,
        inst: &'a Inst,
        active: &[usize],
        (buffer, index, peer): (BufferId, ValueId, Option<ValueId>),
        value: ValueId,
        compare: Option<ValueId>,
    ) -> R {
        let (Some(compare), Op::Atomic { result, .. }) = (compare, &inst.op) else { return Ok(()) };
        let size = self.ir.dtype(value).bytes();
        for &t in active {
            if self.get(*result, t) == self.get(compare, t) {
                continue;
            }
            let (b, idx) = self.target(buffer, index, size, t, "atomic")?;
            let byte = idx as usize * size;
            // A compare-and-swap on another CTA's shared memory polls that CTA's copy.
            self.smem_target = match peer {
                Some(p) => Some(self.dsmem_target(p, t, "an atomic (atom.shared::cluster)")?),
                None => None,
            };
            let (_, block, byte) = self.release_key(b, byte, t);
            self.smem_target = None;
            let k = self.clock_index(t);
            self.warps[k].poll = Some(Poll::Load { buffer: b, block, byte, inst });
            self.warps[k].poll_epoch = self.epoch;
        }
        Ok(())
    }

    /// A call of an external function without a traced body: its Rust simulation on the
    /// value arguments' bits (0 for a pointer), or an error.
    pub(super) fn extern_call(
        &mut self,
        active: &[usize],
        callee: ExternId,
        args: &[AsmInput],
        results: &[ValueId],
    ) -> R {
        let decl = self.ir.extern_decl(callee);
        let Some(f) = decl.sim.clone() else {
            return Err(self.err(
                format!(
                    "extern function `{}` has no simulator implementation: give it one (ExternFn::sim for raw \
                     bits, ExternFn::sim_body for DSL code), or call a DeviceFunction",
                    decl.symbol
                ),
                active.first().copied(),
            ));
        };
        let symbol = decl.symbol.clone();
        self.progress();
        for &t in active {
            let bits: Vec<u64> = args
                .iter()
                .map(|a| match a {
                    AsmInput::Value(v) => self.get(*v, t),
                    AsmInput::Address { .. } => 0,
                })
                .collect();
            let out = f.call(&bits, &mut []);
            if out.len() != results.len() {
                return Err(self.err(
                    format!("the simulation of `{symbol}` returned {} values, not {}", out.len(), results.len()),
                    Some(t),
                ));
            }
            for (r, b) in results.iter().zip(out) {
                let width = self.ir.dtype(*r).bits();
                let b = if width >= 64 { b } else { b & ((1u64 << width) - 1) };
                self.set(*r, t, b);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ld_reduce_words() {
        let f = |x: f32| x.to_bits();
        assert_eq!(f32::from_bits(ld_reduce_word(MultimemType::F32, &[f(1.5), f(2.0), f(-0.5)])), 3.0);
        // Float16 pairs: 1 + 2 = 3 (low), 0.5 + 0.25 = 0.75 (high).
        let h = |lo: f32, hi: f32| f32_to_f16_bits(lo) as u32 | (f32_to_f16_bits(hi) as u32) << 16;
        assert_eq!(ld_reduce_word(MultimemType::F16x2, &[h(1.0, 0.5), h(2.0, 0.25)]), h(3.0, 0.75));
        let b = |lo: f32, hi: f32| f32_to_bf16_bits(lo) as u32 | (f32_to_bf16_bits(hi) as u32) << 16;
        assert_eq!(ld_reduce_word(MultimemType::Bf16x2, &[b(1.0, -4.0), b(1.0, 2.0)]), b(2.0, -2.0));
        // E4M3: 1.0 is 0x38, 2.0 is 0x40, 3.0 is 0x44.
        let w = |a: u32| a | a << 8 | a << 16 | a << 24;
        assert_eq!(ld_reduce_word(MultimemType::E4m3x4, &[w(0x38), w(0x40)]), w(0x44));
        // E5M2: 1.0 is 0x3c, 2.0 is 0x40, 3.0 is 0x42.
        assert_eq!(ld_reduce_word(MultimemType::E5m2x4, &[w(0x3c), w(0x40)]), w(0x42));
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cluster launch control (CuTe DSL's `cute.arch.issue_clc_query` and
//! `cute.arch.clc_response`, roadmap part m10-clc-scheduling; the PTX ISA's
//! `clusterlaunchcontrol.try_cancel` and `clusterlaunchcontrol.query_cancel`,
//! sm_100 and PTX 8.6).

use std::panic::Location;

use crate::builder;
use crate::ir::{DType, Op};
use crate::tensor::{AddressSpace, Ptr, index_operand};
use crate::value::{Int32, Numeric, Value};

/// Check that `ptr` is a 16-byte-aligned shared-memory pointer to a byte-sized
/// type, and give its buffer, index operand and element type.
#[track_caller]
fn response_slot<T: Numeric>(ptr: &Ptr<T>, what: &str) -> (crate::ir::BufferId, crate::ir::ValueId, DType) {
    assert!(
        ptr.memspace() == AddressSpace::Smem,
        "{what}: the response pointer is a {} pointer; the try_cancel response lives in shared memory",
        ptr.memspace()
    );
    assert!(
        !T::DTYPE.is_sub_byte() && 16 % T::DTYPE.bytes() == 0,
        "{what}: the response pointer's element type {} does not divide the 16-byte response",
        T::DTYPE.cute_name()
    );
    let (buffer, offset) = ptr.parts();
    if let Some(o) = (offset.clone() * T::DTYPE.bytes()).as_static() {
        assert!(o % 16 == 0, "{what}: the response is at byte {o} of its buffer, which is not 16-byte aligned");
    }
    (buffer, index_operand(offset), T::DTYPE)
}

/// Take a still-queued cluster of the grid off the launch queue, so that it never runs
/// (CuTe DSL's `cute.arch.issue_clc_query(mbar_ptr, clc_response_ptr,
/// multicast=True)`; `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128`).
/// The opaque 16-byte response is written asynchronously at
/// `clc_response_ptr` (16-byte aligned shared memory) and completes 16 bytes
/// of tx on the mbarrier at `mbar_ptr`, whose phase must expect them; decode
/// it with [`clc_response`] after waiting on that phase. With `multicast`
/// (sm_100a) the response lands at the same offsets in every CTA of the
/// cluster, counting down each CTA's mbarrier: every CTA must still be
/// running. Per thread: issue it from one thread
/// ([`elect_one`](crate::arch::elect_one)).
#[track_caller]
pub fn issue_clc_query<T: Numeric>(mbar_ptr: &Ptr<u64>, clc_response_ptr: &Ptr<T>, multicast: bool) {
    let loc = Location::caller();
    let what = "issue_clc_query";
    assert!(
        mbar_ptr.memspace() == AddressSpace::Smem,
        "{what}: the mbarrier pointer is a {} pointer, not a shared-memory one",
        mbar_ptr.memspace()
    );
    let (buffer, index, elem) = response_slot(clc_response_ptr, what);
    let (mbar, mbar_offset) = mbar_ptr.parts();
    let mbar_index = index_operand(mbar_offset);
    builder::with(|b| b.push(Op::ClcTryCancel { buffer, index, elem, mbar, mbar_index, multicast }, loc));
}

/// Decode the try_cancel response at `result_addr` (CuTe DSL's
/// `cute.arch.clc_response(result_addr)`): a 16-byte shared-memory load,
/// then `clusterlaunchcontrol.query_cancel.is_canceled` and
/// `get_first_ctaid::{x,y,z}`. Returns the cancelled cluster's first CTA's
/// `(x, y, z)` and whether a cluster was cancelled (1 or 0) as `Int32`s;
/// the coordinates are meaningless when it was not.
#[track_caller]
pub fn clc_response<T: Numeric>(result_addr: &Ptr<T>) -> (Int32, Int32, Int32, Int32) {
    let loc = Location::caller();
    let (buffer, index, elem) = response_slot(result_addr, "clc_response");
    let results = builder::with(|b| {
        let results = [0; 4].map(|_| b.new_value(DType::I32));
        b.push(Op::ClcQueryCancel { buffer, index, elem, results }, loc);
        results
    });
    let [x, y, z, valid] = results.map(Value::<i32>::from_id);
    (x, y, z, valid)
}

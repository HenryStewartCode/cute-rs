// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's PdlWaitBarrier and
// PdlLaunchBarrier (cutlass.experimental.task_scheduling.resources, NVIDIA
// EULA); the code is written independently.

//! Programmatic dependent launch as barrier-only resources.

use crate::arch::{griddepcontrol_launch_dependents, griddepcontrol_wait};

use super::resource::{MemoryResource, Resource, Work, WorkResults};

/// Which side of a programmatic dependent launch a barrier resource is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PdlRole {
    /// [`PdlWaitBarrier`]: `griddepcontrol.wait`.
    Wait,
    /// [`PdlLaunchBarrier`]: `griddepcontrol.launch_dependents`.
    Launch,
}

/// The wait for the previous grid of a programmatic dependent launch (CuTe
/// DSL's `PdlWaitBarrier`): a barrier-only resource without a pipeline,
/// whose consumer work `wait_griddep` is `griddepcontrol.wait` (sm_90).
///
/// A resource whose data the previous grid writes lists the wait among its
/// upstream resources in the TaskManager's dependency graph (an
/// ordering-only edge); the TaskManager then checks that every task
/// producing that resource has waited before its producer steps. The wait
/// itself may sit in any phase.
#[derive(Debug)]
pub struct PdlWaitBarrier {
    mem: MemoryResource,
}

impl PdlWaitBarrier {
    /// The wait barrier called `name`.
    pub fn new(name: &str) -> PdlWaitBarrier {
        let mem = MemoryResource::new(name).work(Work::consumer("wait_griddep").run(|_, _| {
            griddepcontrol_wait();
            WorkResults::none()
        }));
        PdlWaitBarrier { mem }
    }
}

impl Resource for PdlWaitBarrier {
    fn memory(&self) -> &MemoryResource {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut MemoryResource {
        &mut self.mem
    }

    fn pdl_role(&self) -> Option<PdlRole> {
        Some(PdlRole::Wait)
    }
}

/// The early launch of the dependent grid (CuTe DSL's `PdlLaunchBarrier`):
/// a barrier-only resource without a pipeline, whose producer work
/// `launch_griddep` is `griddepcontrol.launch_dependents` (sm_90).
///
/// It may sit in any task and phase, but the TaskManager checks that a
/// `wait_griddep` has run before it on every path (in the same task, or in
/// a task it waits on through a pipeline). It is never a downstream
/// resource of the dependency graph, and a kernel need not launch early at
/// all.
#[derive(Debug)]
pub struct PdlLaunchBarrier {
    mem: MemoryResource,
}

impl PdlLaunchBarrier {
    /// The launch barrier called `name`.
    pub fn new(name: &str) -> PdlLaunchBarrier {
        let mem = MemoryResource::new(name).work(Work::producer("launch_griddep").run(|_, _| {
            griddepcontrol_launch_dependents();
            WorkResults::none()
        }));
        PdlLaunchBarrier { mem }
    }
}

impl Resource for PdlLaunchBarrier {
    fn memory(&self) -> &MemoryResource {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut MemoryResource {
        &mut self.mem
    }

    fn pdl_role(&self) -> Option<PdlRole> {
        Some(PdlRole::Launch)
    }
}

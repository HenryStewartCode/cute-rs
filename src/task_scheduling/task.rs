// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.task.Task (NVIDIA EULA); the code is
// written independently.

//! Tasks: a captured schedule bound to a range of warps.

use super::resource::Resource;
use super::schedule::Schedule;

/// A change of the per-thread register budget a task makes when it starts
/// (`setmaxnreg`, sm_90a and newer): the task's warps must be whole,
/// aligned warpgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterBudget {
    /// [`setmaxregister_increase`](crate::setmaxregister_increase) to this count.
    Increase(u32),
    /// [`setmaxregister_decrease`](crate::setmaxregister_decrease) to this count.
    Decrease(u32),
}

impl RegisterBudget {
    /// The register count.
    pub fn count(self) -> u32 {
        match self {
            RegisterBudget::Increase(n) | RegisterBudget::Decrease(n) => n,
        }
    }
}

/// A schedule bound to warps `warp_idx .. warp_idx + num_warps` of the
/// block (CuTe DSL's `Task(name, src_resources, dst_resources, warp_idx,
/// num_warps, schedule)`). The task consumes its source resources (their
/// consumer-side steps) and produces its destination resources (their
/// producer-side steps); the [`TaskManager`](super::TaskManager) checks
/// that the schedule agrees and runs it on those warps. A task with no
/// resources and an empty schedule pads warps that have nothing to do.
#[derive(Clone, Debug)]
pub struct Task {
    name: String,
    src: Vec<String>,
    dst: Vec<String>,
    warp_idx: u32,
    num_warps: u32,
    schedule: Schedule,
    registers: Option<RegisterBudget>,
}

impl Task {
    /// The task `name` on warps `warp_idx .. warp_idx + num_warps`.
    pub fn new(
        name: &str,
        src_resources: &[&dyn Resource],
        dst_resources: &[&dyn Resource],
        warp_idx: u32,
        num_warps: u32,
        schedule: Schedule,
    ) -> Task {
        Task {
            name: name.to_string(),
            src: src_resources.iter().map(|r| r.name().to_string()).collect(),
            dst: dst_resources.iter().map(|r| r.name().to_string()).collect(),
            warp_idx,
            num_warps,
            schedule,
            registers: None,
        }
    }

    /// Raising the register budget to `count` when it starts.
    pub fn setmaxnreg_increase(mut self, count: u32) -> Task {
        self.registers = Some(RegisterBudget::Increase(count));
        self
    }

    /// Lowering the register budget to `count` when it starts.
    pub fn setmaxnreg_decrease(mut self, count: u32) -> Task {
        self.registers = Some(RegisterBudget::Decrease(count));
        self
    }

    /// The task's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The names of the resources it consumes.
    pub fn src_resources(&self) -> &[String] {
        &self.src
    }

    /// The names of the resources it produces.
    pub fn dst_resources(&self) -> &[String] {
        &self.dst
    }

    /// Its first warp.
    pub fn warp_idx(&self) -> u32 {
        self.warp_idx
    }

    /// Its number of warps.
    pub fn num_warps(&self) -> u32 {
        self.num_warps
    }

    /// Its threads (`32 * num_warps`).
    pub fn num_threads(&self) -> u32 {
        32 * self.num_warps
    }

    /// Its schedule.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Its register budget, if it sets one.
    pub fn registers(&self) -> Option<RegisterBudget> {
        self.registers
    }

    /// Whether `resource` is one of its sources.
    pub fn consumes(&self, resource: &str) -> bool {
        self.src.iter().any(|r| r == resource)
    }

    /// Whether `resource` is one of its destinations.
    pub fn produces(&self, resource: &str) -> bool {
        self.dst.iter().any(|r| r == resource)
    }
}

//! Helpers shared by test binaries: the printf kernels the simulator and
//! the GPU both run, and the child-process runner that captures what a
//! kernel's device code writes to stdout.

// Shared by several test binaries; each uses part of it.
#![allow(dead_code)]

pub mod child;
pub mod printf_cases;

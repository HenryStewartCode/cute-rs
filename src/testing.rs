//! CuTe DSL's `cute.testing` and `cutlass.testing`: device assertions ([`assert_`], a check
//! that stops the kernel, and [`RuntimeAssertion`], predicates a kernel records for the host
//! to verify after the run), benchmarking ([`benchmark`], [`get_workspace_count`]; feature
//! `cuda`) and autotuning ([`tune`], [`Autotuner`]).

mod autotune;
mod benchmark;
mod convert;

#[cfg(feature = "cuda")]
pub use autotune::tune;
pub use autotune::{Autotuner, CantImplement, Trial, TuneError, TuneKey, Tuned, tune_by};
#[cfg(feature = "cuda")]
pub use benchmark::{BenchError, benchmark, benchmark_with_workspaces, get_workspace_count};
pub use benchmark::{BenchmarkOptions, workspace_count_for_l2};
#[cfg(feature = "cuda")]
pub use convert::convert;
pub use convert::{CONVERT_DTYPES, ConvertError, convert_kernels_built, convert_sim, elem_per_copy};

use std::panic::Location;

use crate::builder::{self, Kernel};
use crate::diagnostics::{AssertionFailure, SourceLoc};
use crate::ir::Op;
use crate::layout::Layout;
use crate::params::{ArgError, ArgPack, KernelParam, leaf_name};
use crate::sim::KernelArg;
use crate::tensor::Tensor;
use crate::value::Boolean;

/// Assert `cond` in device code (CuTe DSL's `cute.testing.assert_(cond,
/// msg)`): where it fails, the thread stops.
///
/// * The simulator checks it unless [`SimOptions::assertions`](crate::SimOptions::assertions)
///   is off: a failure is a [`SimError`](crate::SimError) naming this line,
///   the block, the thread and `msg`.
/// * Compiled code includes it only when assertions are enabled
///   ([`CompileOptions::enable_assertions`](crate::CompileOptions::enable_assertions),
///   `--enable-assertions`, `CUTE_RS_ENABLE_ASSERTIONS=1`), as `cf.assert`: a
///   call to `__assertfail`, which prints `msg` with this file and line on
///   stderr, and the launch fails with `CUDA_ERROR_ASSERT`, which is sticky:
///   the process can no longer use the GPU, so run kernels that may assert
///   in a child process.
#[track_caller]
pub fn assert_(cond: impl Into<Boolean>, msg: &str) {
    let loc = Location::caller();
    let cond = cond.into().checked();
    builder::with(|b| b.push(Op::Assert { cond, message: msg.to_string() }, loc));
}

/// One [`DeviceAssertions::store`] of a kernel: the build-time table entry
/// [`RuntimeAssertion::verify`] reports from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssertionRecord {
    /// The parameter's name.
    pub param: String,
    pub index: usize,
    pub message: String,
    pub loc: SourceLoc,
}

/// What a kernel is built for: [`RuntimeAssertion::spec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssertionSpec {
    pub num_assertions: usize,
    pub init_value: bool,
    /// No parameter at all, and stores do nothing.
    pub disable: bool,
}

/// The device side of a [`RuntimeAssertion`]: a Boolean buffer of
/// `num_assertions` flags.
#[derive(Clone)]
pub struct DeviceAssertions {
    param: String,
    spec: AssertionSpec,
    flags: Option<Tensor<bool>>,
}

impl DeviceAssertions {
    /// Declare the parameter `name` (none when `spec.disable`): the
    /// [`KernelParam`] declaration, for kernels built with
    /// [`Kernel::build`](crate::Kernel::build). Pass
    /// [`RuntimeAssertion::kernel_arg`] for it.
    #[track_caller]
    pub fn param(name: &str, spec: AssertionSpec) -> DeviceAssertions {
        let flags = (!spec.disable).then(|| crate::param_tensor::<bool>(name, Layout::col_major(spec.num_assertions)));
        DeviceAssertions { param: name.to_string(), spec, flags }
    }

    /// Record `pred` at `idx` (CuTe DSL's `RuntimeAssertion.store(idx,
    /// pred, msg)`): the flag becomes `pred` (`pred` and the flag, with an
    /// initial value of true). The message and this line go into the
    /// kernel's table for [`RuntimeAssertion::verify`]. Every thread that
    /// runs it writes the flag, so store from one thread (or the same value
    /// from all).
    #[track_caller]
    pub fn store(&self, idx: usize, pred: impl Into<Boolean>, msg: &str) {
        let loc = Location::caller();
        assert!(
            idx < self.spec.num_assertions,
            "RuntimeAssertion::store: index {idx} is past the {} assertions; make more",
            self.spec.num_assertions
        );
        let Some(flags) = &self.flags else { return };
        let pred = pred.into();
        let value = if self.spec.init_value { pred & flags.load(idx) } else { pred };
        flags.store(idx, value);
        let record = AssertionRecord {
            param: self.param.clone(),
            index: idx,
            message: msg.to_string(),
            loc: SourceLoc::from(loc),
        };
        builder::with(|b| {
            b.assertion_records.retain(|r| !(r.param == record.param && r.index == idx));
            b.assertion_records.push(record);
        });
    }
}

/// Predicates a kernel records for the host to check (CuTe DSL's
/// `cute.testing.RuntimeAssertion(num_assertions, init_value=False,
/// disable=False)`): the host owns a Boolean buffer of `num_assertions`
/// flags, all `init_value`; the kernel's [`DeviceAssertions::store`]
/// writes them; [`verify`](RuntimeAssertion::verify) reports the first
/// stored flag that is false, with its message and line.
///
/// It is a [`KernelParam`] (with [`DeviceAssertions`] on the device), so a
/// [`Kernel::jit`](crate::KernelOptions::jit) kernel takes it as an
/// argument; with [`Kernel::build`](crate::Kernel::build), declare it with
/// [`DeviceAssertions::param`], pass [`kernel_arg`](RuntimeAssertion::kernel_arg)
/// and [`read_back`](RuntimeAssertion::read_back) after the run.
#[derive(Clone, Debug)]
pub struct RuntimeAssertion {
    spec: AssertionSpec,
    flags: Vec<bool>,
}

impl RuntimeAssertion {
    /// `num_assertions` flags, initially false.
    pub fn new(num_assertions: usize) -> RuntimeAssertion {
        RuntimeAssertion {
            spec: AssertionSpec { num_assertions, init_value: false, disable: false },
            flags: vec![false; num_assertions],
        }
    }

    /// The flags' initial value.
    pub fn init_value(mut self, value: bool) -> RuntimeAssertion {
        self.spec.init_value = value;
        self.flags = vec![value; self.spec.num_assertions];
        self
    }

    /// Turn the assertions off: no parameter, stores do nothing, `verify`
    /// passes.
    pub fn disable(mut self, disable: bool) -> RuntimeAssertion {
        self.spec.disable = disable;
        self
    }

    /// The spec a kernel is built for.
    pub fn spec(&self) -> AssertionSpec {
        self.spec
    }

    /// The buffer argument (`None` when disabled).
    pub fn kernel_arg(&self) -> Option<KernelArg> {
        (!self.spec.disable).then(|| KernelArg::buffer(&self.flags))
    }

    /// Take the flags back from the run's argument.
    pub fn read_back(&mut self, arg: &KernelArg) {
        self.flags = arg.to_vec::<bool>();
    }

    /// The flags as the last run left them.
    pub fn flags(&self) -> &[bool] {
        &self.flags
    }

    /// Check the flags the kernel stored: the first false one (by index)
    /// fails, with its message and the line of its store. `kernel` holds
    /// the table; it must have one `RuntimeAssertion` parameter (see
    /// [`verify_param`](RuntimeAssertion::verify_param) for several).
    pub fn verify(&self, kernel: &Kernel) -> Result<(), AssertionFailure> {
        let mut params: Vec<&str> = kernel.assertion_records().iter().map(|r| r.param.as_str()).collect();
        params.sort_unstable();
        params.dedup();
        assert!(
            params.len() <= 1,
            "RuntimeAssertion::verify: kernel `{}` has several assertion parameters ({}); use verify_param",
            kernel.name(),
            params.join(", ")
        );
        match params.first() {
            Some(p) => self.verify_param(kernel, p),
            None => Ok(()),
        }
    }

    /// [`verify`](RuntimeAssertion::verify) against the stores of the
    /// parameter `param`.
    pub fn verify_param(&self, kernel: &Kernel, param: &str) -> Result<(), AssertionFailure> {
        if self.spec.disable {
            return Ok(());
        }
        let mut used: Vec<&AssertionRecord> = kernel.assertion_records().iter().filter(|r| r.param == param).collect();
        used.sort_by_key(|r| r.index);
        match used.into_iter().find(|r| !self.flags.get(r.index).copied().unwrap_or(true)) {
            Some(r) => Err(AssertionFailure { message: r.message.clone(), loc: r.loc.clone(), index: r.index }),
            None => Ok(()),
        }
    }
}

impl KernelParam for DeviceAssertions {
    type Spec = AssertionSpec;
    type Host = RuntimeAssertion;

    #[track_caller]
    fn declare(name: &str, spec: &AssertionSpec) -> Self {
        DeviceAssertions::param(&leaf_name(name), *spec)
    }

    fn spec_of(host: &RuntimeAssertion) -> AssertionSpec {
        host.spec
    }

    fn pack(name: &str, spec: &AssertionSpec, host: &RuntimeAssertion, out: &mut ArgPack) -> Result<(), ArgError> {
        let name = leaf_name(name);
        if host.spec != *spec {
            return Err(ArgError::new(
                &name,
                format!("the kernel was built for {spec:?}, but the RuntimeAssertion is {:?}", host.spec),
            ));
        }
        if let Some(arg) = host.kernel_arg() {
            out.arg(&name, arg);
        }
        Ok(())
    }

    fn write_back(spec: &AssertionSpec, host: &mut RuntimeAssertion, buffers: &mut dyn Iterator<Item = KernelArg>) {
        if !spec.disable {
            host.read_back(&buffers.next().expect("cute: write_back: too few buffers"));
        }
    }
}

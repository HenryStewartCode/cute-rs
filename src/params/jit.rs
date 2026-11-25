//! Typed kernel entry points: a kernel whose parameters follow the
//! signature of the closure that builds it ([`Kernel::jit`]), launched with
//! host values of the matching types.

use std::marker::PhantomData;

use super::{ArgError, ArgPack, KernelParam, pack_args, pack_args_full, write_back_args};
use crate::builder::{Kernel, KernelOptions};
use crate::sim::{KernelArg, LaunchConfig, SimError, SimOptions, SimReport};

/// A kernel body taking the parameters `P`: a closure of one argument (`P`
/// itself, a tuple or struct of parameters) or of several (`P` is then
/// the tuple of their types). `M` tells the forms apart and is inferred.
pub trait JitBody<P, M> {
    fn call(self, params: P);
}

/// The marker of a closure of one argument.
pub struct OneArg;

impl<F: FnOnce(P), P> JitBody<P, OneArg> for F {
    fn call(self, params: P) {
        self(params)
    }
}

macro_rules! jit_body {
    ($marker:ident; $($name:ident $idx:tt),+) => {
        /// The marker of a closure of several arguments.
        pub struct $marker;

        impl<Func: FnOnce($($name),+), $($name),+> JitBody<($($name,)+), $marker> for Func {
            fn call(self, params: ($($name,)+)) {
                self($(params.$idx),+)
            }
        }
    };
}
jit_body!(TwoArgs; A 0, B 1);
jit_body!(ThreeArgs; A 0, B 1, C 2);
jit_body!(FourArgs; A 0, B 1, C 2, D 3);
jit_body!(FiveArgs; A 0, B 1, C 2, D 3, E 4);
jit_body!(SixArgs; A 0, B 1, C 2, D 3, E 4, F 5);

/// A kernel built from a typed signature `P` (CuTe DSL's `@cute.jit` /
/// `@cute.kernel` function, whose parameters are runtime arguments unless
/// `Constexpr`), with the [`Spec`](KernelParam::Spec) it was built for. It
/// is a [`Kernel`] (through `Deref`), and its launches take a
/// [`P::Host`](KernelParam::Host) value: checked against the spec, packed
/// into the flat argument list, and given the outputs back.
pub struct TypedKernel<P: KernelParam> {
    kernel: Kernel,
    spec: P::Spec,
    _p: PhantomData<fn() -> P>,
}

impl<P: KernelParam> Clone for TypedKernel<P> {
    fn clone(&self) -> Self {
        TypedKernel { kernel: self.kernel.clone(), spec: self.spec.clone(), _p: PhantomData }
    }
}

impl<P: KernelParam> std::ops::Deref for TypedKernel<P> {
    type Target = Kernel;
    fn deref(&self) -> &Kernel {
        &self.kernel
    }
}

impl<P: KernelParam> TypedKernel<P> {
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn into_kernel(self) -> Kernel {
        self.kernel
    }

    pub fn spec(&self) -> &P::Spec {
        &self.spec
    }

    /// The flat argument list for `host`, checked against the spec and the
    /// kernel's parameters.
    pub fn pack(&self, host: &P::Host) -> Result<Vec<KernelArg>, ArgError> {
        pack_args::<P>(&self.kernel, "", &self.spec, host)
    }

    /// [`pack`](Self::pack), keeping what backs each argument (external memory) and
    /// refusing a read-only external tensor for a buffer the kernel writes.
    pub fn pack_full(&self, host: &P::Host) -> Result<ArgPack, ArgError> {
        pack_args_full::<P>(&self.kernel, "", &self.spec, host)
    }

    /// Take the buffers of `args` (as [`pack`](Self::pack) made them, after
    /// a run) back into `host`.
    pub fn write_back(&self, host: &mut P::Host, args: Vec<KernelArg>) {
        write_back_args::<P>(&self.spec, host, args)
    }

    /// Run on the simulator with `host`'s values ([`Kernel::simulate`]),
    /// taking the outputs back into it.
    pub fn simulate(&self, cfg: LaunchConfig, host: &mut P::Host) -> Result<SimReport, SimError> {
        self.simulate_with(cfg, host, &SimOptions::default())
    }

    /// [`simulate`](Self::simulate) with options.
    pub fn simulate_with(
        &self,
        cfg: LaunchConfig,
        host: &mut P::Host,
        opts: &SimOptions,
    ) -> Result<SimReport, SimError> {
        let pack = self.pack_full(host).map_err(|e| SimError::plain(e.to_string()))?;
        if let Some((i, d)) =
            pack.externals().iter().enumerate().find_map(|(i, e)| e.and_then(|e| e.device).map(|d| (i, d)))
        {
            return Err(SimError::plain(format!(
                "argument `{}` is CUDA device memory (device {}): the simulator runs on host memory (kDLCPU) \
                 DLPack tensors, which it copies in and writes back",
                pack.names()[i],
                d.2
            )));
        }
        let mut args = pack.finish();
        let report = self.kernel.simulate_with(cfg, &mut args, opts)?;
        self.write_back(host, args);
        Ok(report)
    }

    /// Compile for `ctx`'s device and run there with `host`'s values
    /// ([`Kernel::run_cuda`]), taking the outputs back into it.
    #[cfg(feature = "cuda")]
    pub fn run_cuda(
        &self,
        ctx: &crate::cuda::CudaContext,
        toolchain: &crate::Toolchain,
        cfg: LaunchConfig,
        host: &mut P::Host,
    ) -> Result<crate::cuda::LaunchStats, Box<dyn std::error::Error>> {
        self.run_cuda_with(ctx, toolchain, &crate::CompileOptions::new().with_env()?, cfg, host)
    }

    /// [`run_cuda`](Self::run_cuda) with compile options.
    #[cfg(feature = "cuda")]
    pub fn run_cuda_with(
        &self,
        ctx: &crate::cuda::CudaContext,
        toolchain: &crate::Toolchain,
        options: &crate::CompileOptions,
        cfg: LaunchConfig,
        host: &mut P::Host,
    ) -> Result<crate::cuda::LaunchStats, Box<dyn std::error::Error>> {
        let pack = self.pack_full(host)?;
        let externals = pack.externals().to_vec();
        let mut args = pack.finish();
        let stats = self.kernel.run_cuda_external(ctx, toolchain, options, cfg, &mut args, &externals)?;
        self.write_back(host, args);
        Ok(stats)
    }

    /// Launch an already loaded `function` (compiled from this kernel) with
    /// `host`'s values ([`CudaContext::launch`](crate::cuda::CudaContext::launch)).
    #[cfg(feature = "cuda")]
    pub fn launch(
        &self,
        ctx: &crate::cuda::CudaContext,
        function: &crate::cuda::CudaFunction<'_>,
        cfg: LaunchConfig,
        host: &mut P::Host,
    ) -> Result<crate::cuda::LaunchStats, Box<dyn std::error::Error>> {
        let pack = self.pack_full(host)?;
        let externals = pack.externals().to_vec();
        let mut args = pack.finish();
        let stats = ctx.launch_external(&self.kernel, function, cfg, &mut args, &externals)?;
        self.write_back(host, args);
        Ok(stats)
    }
}

impl KernelOptions {
    /// Build a kernel whose parameters are `body`'s arguments, declared from
    /// `spec` (see [`Kernel::jit`]).
    #[track_caller]
    pub fn jit<P: KernelParam, M>(self, spec: P::Spec, body: impl JitBody<P, M>) -> TypedKernel<P> {
        let s = spec.clone();
        let kernel = self.build(move || body.call(P::declare("", &s)));
        TypedKernel { kernel, spec, _p: PhantomData }
    }

    /// Build a kernel whose parameters are `body`'s arguments, with the
    /// spec the host values imply (see [`Kernel::jit_for`]).
    #[track_caller]
    pub fn jit_for<P: KernelParam, M>(self, host: &P::Host, body: impl JitBody<P, M>) -> TypedKernel<P> {
        self.jit(P::spec_of(host), body)
    }
}

impl Kernel {
    /// Build a typed kernel entry point (CuTe DSL's `@cute.jit` function
    /// with annotated parameters): the closure's arguments are the kernel's
    /// parameters ([`KernelParam`]s: `Int32`, `Tensor<f32>`, `Ptr<f16>`,
    /// layouts and shapes, `Option`s, tuples, arrays, [`kernel_params!`](crate::kernel_params)
    /// structs, and [`Constexpr`](crate::Constexpr)s and plain Rust values,
    /// which are build-time constants), declared from `spec` in order and
    /// named `arg0`, `arg1`, ... (a struct's fields by their names). Every
    /// runtime type is a runtime argument, as in CuTe DSL.
    ///
    /// ```
    /// use cute::prelude::*;
    ///
    /// let spec = (ScalarSpec::Plain, make_fake_compact_tensor::<f32>((sym_int32(1, None),), None, false));
    /// let k = Kernel::builder("scale").block_dim([32, 1, 1]).jit(spec, |s: Float32, x: Tensor<f32>| {
    ///     let i = thread_idx().x;
    ///     if_(i.lt(size(&x, &[0])), || x.store(i, x.load(i) * s));
    /// });
    /// let x = HostTensor::from_slice(&[1.0f32, 2.0, 3.0], Layout::col_major((3,)));
    /// let mut host = (2.0f32, x.mark_layout_dynamic(None).unwrap());
    /// k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut host).unwrap();
    /// assert_eq!(host.1.to_vec::<f32>(), [2.0, 4.0, 6.0]);
    /// ```
    ///
    /// Host values are checked by type when the program is compiled: a
    /// launch with the wrong number or types of values does not compile.
    ///
    /// ```compile_fail
    /// use cute::prelude::*;
    /// let k = Kernel::jit("k", (ScalarSpec::Plain, ScalarSpec::Plain), |_: Int32, _: Float32| {});
    /// let mut host = (1i32,); // one value for two parameters
    /// k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut host).unwrap();
    /// ```
    ///
    /// ```compile_fail
    /// use cute::prelude::*;
    /// let k = Kernel::jit("k", (ScalarSpec::Plain, ScalarSpec::Plain), |_: Int32, _: Float32| {});
    /// let mut host = (1i32, 2i32); // an i32 for the Float32
    /// k.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut host).unwrap();
    /// ```
    #[track_caller]
    pub fn jit<P: KernelParam, M>(name: impl Into<String>, spec: P::Spec, body: impl JitBody<P, M>) -> TypedKernel<P> {
        Kernel::builder(name).jit(spec, body)
    }

    /// [`Kernel::jit`] with the spec the host values imply (CuTe DSL's
    /// call-site typing): a host tensor's own spec (static unless marked
    /// dynamic), a plain scalar for each number, a dynamic `Int32` leaf for
    /// each integer of a shape, the value of each constant.
    #[track_caller]
    pub fn jit_for<P: KernelParam, M>(
        name: impl Into<String>,
        host: &P::Host,
        body: impl JitBody<P, M>,
    ) -> TypedKernel<P> {
        Kernel::builder(name).jit_for(host, body)
    }
}

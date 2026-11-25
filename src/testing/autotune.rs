//! Autotuning (CuTe DSL's `cutlass.testing.tune` and `cute.testing.autotune_jit`): time every
//! configuration of a kernel, skip the ones that cannot be built or launched, keep the
//! fastest, and remember it per problem.
//!
//! [`tune`] (feature `cuda`) takes the configurations (CuTe DSL's `params_dict` is their
//! product: build it with iterators) and a closure that builds the kernel for one and returns
//! its launching closure, which [`benchmark`](super::benchmark) times. A configuration whose
//! build panics (a builder's refusal: a tile that does not divide, a vector too wide for the
//! layout) or returns an error ([`CantImplement`], CuTe DSL's `CantImplementError`, or any
//! other), or whose launches fail, is skipped and recorded in [`Tuned::trials`]. [`tune_by`]
//! is the same over any measure, so the selection runs without a GPU too. A caught panic is
//! still printed by the panic hook. A launch that faults (an illegal address, a device-side
//! assertion) leaves a sticky CUDA error, and every later configuration then fails too: the
//! trials name the first, and the build closure should refuse what it cannot run correctly
//! ([`CantImplement`]) rather than launch it.
//!
//! [`Autotuner`] keeps the winner per [`TuneKey`]: the problem's cache key (what the compile
//! cache keys the kernel on, such as [`TensorSpec::cache_keys`](crate::TensorSpec::cache_keys)
//! of its tensors, so problems that compile to the same kernel share a winner) plus the named
//! parts its [`update_on_change`](Autotuner::update_on_change) lists, such as the runtime
//! extents `M` and `N`, a change in which tunes again. The kernels a tuning builds go through
//! the compile cache ([`KernelCache`](crate::KernelCache) or `Kernel::compile_with`'s own), so
//! revisiting a configuration does not recompile it.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

/// A configuration that cannot be implemented for the problem (CuTe DSL's
/// `CantImplementError`): return it from a tuning's build closure to skip the configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CantImplement(pub String);

impl CantImplement {
    pub fn new(message: impl Into<String>) -> CantImplement {
        CantImplement(message.into())
    }
}

impl fmt::Display for CantImplement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() { f.write_str("the configuration cannot be implemented") } else { f.write_str(&self.0) }
    }
}

impl Error for CantImplement {}

/// One configuration's outcome: its time in microseconds, or why it was skipped.
#[derive(Clone, Debug, PartialEq)]
pub struct Trial<C> {
    pub config: C,
    pub outcome: Result<f64, String>,
}

/// A tuning's result: the fastest configuration, its time in microseconds, and every trial
/// in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuned<C> {
    pub config: C,
    pub time_us: f64,
    pub trials: Vec<Trial<C>>,
}

impl<C> Tuned<C> {
    /// The configurations that were skipped, with the reasons.
    pub fn skipped(&self) -> impl Iterator<Item = (&C, &str)> {
        self.trials.iter().filter_map(|t| t.outcome.as_ref().err().map(|e| (&t.config, e.as_str())))
    }
}

/// No configuration could be timed: each trial's configuration (as `Debug` prints it) and
/// reason, or none at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuneError {
    pub trials: Vec<(String, String)>,
}

impl fmt::Display for TuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.trials.is_empty() {
            return f.write_str("tune: no configurations to try");
        }
        write!(f, "tune: no configuration could be timed")?;
        for (c, why) in &self.trials {
            write!(f, "; {c}: {why}")?;
        }
        Ok(())
    }
}

impl Error for TuneError {}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    let m = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
    format!("panicked: {}", m.unwrap_or_default())
}

/// Measure every configuration with `measure` (microseconds) and return the fastest
/// ([`tune`] with any measure). A configuration whose measure panics, fails or gives a time
/// that is not a finite non-negative number is skipped; ties go to the earlier one.
pub fn tune_by<C: Clone + fmt::Debug>(
    configs: impl IntoIterator<Item = C>,
    mut measure: impl FnMut(&C) -> Result<f64, Box<dyn Error>>,
) -> Result<Tuned<C>, TuneError> {
    let mut trials = Vec::new();
    let mut best: Option<(usize, f64)> = None;
    for config in configs {
        let outcome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| measure(&config))) {
            Ok(Ok(t)) if t.is_finite() && t >= 0.0 => Ok(t),
            Ok(Ok(t)) => Err(format!("measured {t} microseconds")),
            Ok(Err(e)) => Err(e.to_string()),
            Err(p) => Err(panic_text(&*p)),
        };
        if let Ok(t) = outcome
            && best.is_none_or(|(_, b)| t < b)
        {
            best = Some((trials.len(), t));
        }
        trials.push(Trial { config, outcome });
    }
    match best {
        Some((i, time_us)) => Ok(Tuned { config: trials[i].config.clone(), time_us, trials }),
        None => Err(TuneError {
            trials: trials.into_iter().map(|t| (format!("{:?}", t.config), t.outcome.unwrap_err())).collect(),
        }),
    }
}

/// What an [`Autotuner`] keeps a winner for: the problem's cache key and named parts (runtime
/// values such as extents), of which only those the tuner's
/// [`update_on_change`](Autotuner::update_on_change) names count.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuneKey {
    cache_key: u64,
    parts: Vec<(String, String)>,
}

impl TuneKey {
    /// A key of the problem's cache key (anything hashable: a kernel's spec keys, its name).
    pub fn new(cache_key: &impl Hash) -> TuneKey {
        let mut h = DefaultHasher::new();
        cache_key.hash(&mut h);
        TuneKey { cache_key: h.finish(), parts: Vec::new() }
    }

    /// Add the part `name` with `value` (compared as `Debug` prints it).
    pub fn part(mut self, name: &str, value: impl fmt::Debug) -> Self {
        self.parts.retain(|(n, _)| n != name);
        self.parts.push((name.to_string(), format!("{value:?}")));
        self
    }
}

/// A tuner that keeps the winning configuration per problem (CuTe DSL's `autotune_jit`): the
/// first call for a [`TuneKey`] tunes, later ones return the winner. See the
/// module documentation.
#[derive(Clone, Debug)]
pub struct Autotuner<C> {
    configs: Vec<C>,
    update_on_change: Vec<String>,
    options: super::BenchmarkOptions,
    winners: HashMap<TuneKey, Tuned<C>>,
    tunings: usize,
}

impl<C: Clone + fmt::Debug> Autotuner<C> {
    /// A tuner over `configs`, benchmarking with the default [`BenchmarkOptions`](super::BenchmarkOptions).
    pub fn new(configs: impl IntoIterator<Item = C>) -> Autotuner<C> {
        Autotuner {
            configs: configs.into_iter().collect(),
            update_on_change: Vec::new(),
            options: super::BenchmarkOptions::default(),
            winners: HashMap::new(),
            tunings: 0,
        }
    }

    /// The key parts whose change tunes again (CuTe DSL's `update_on_change=["M", "N"]`); the
    /// others are ignored.
    pub fn update_on_change(mut self, names: &[&str]) -> Self {
        self.update_on_change = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// How [`get_or_tune`](Self::get_or_tune) benchmarks.
    pub fn options(mut self, options: super::BenchmarkOptions) -> Self {
        self.options = options;
        self
    }

    pub fn configs(&self) -> &[C] {
        &self.configs
    }

    /// The key a winner is kept under: the cache key and the watched parts, in name order.
    fn effective(&self, key: &TuneKey) -> TuneKey {
        let mut parts: Vec<(String, String)> =
            key.parts.iter().filter(|(n, _)| self.update_on_change.contains(n)).cloned().collect();
        parts.sort();
        TuneKey { cache_key: key.cache_key, parts }
    }

    /// The winner kept for `key`, if it was tuned.
    pub fn get(&self, key: &TuneKey) -> Option<&Tuned<C>> {
        self.winners.get(&self.effective(key))
    }

    /// The winner for `key`, tuning with `measure` ([`tune_by`]) if there is none. A failed
    /// tuning keeps nothing.
    pub fn get_or_tune_by(
        &mut self,
        key: &TuneKey,
        measure: impl FnMut(&C) -> Result<f64, Box<dyn Error>>,
    ) -> Result<&Tuned<C>, TuneError> {
        let key = self.effective(key);
        if !self.winners.contains_key(&key) {
            let tuned = tune_by(self.configs.iter().cloned(), measure)?;
            self.tunings += 1;
            self.winners.insert(key.clone(), tuned);
        }
        Ok(&self.winners[&key])
    }

    /// How many tunings ran.
    pub fn tunings(&self) -> usize {
        self.tunings
    }

    /// How many winners are kept.
    pub fn len(&self) -> usize {
        self.winners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.winners.is_empty()
    }

    /// Forget every winner.
    pub fn clear(&mut self) {
        self.winners.clear();
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::{Autotuner, TuneError, TuneKey, Tuned, tune_by};
    use crate::cuda::CudaStream;
    use crate::testing::{BenchError, BenchmarkOptions, benchmark};

    /// Tune on the GPU (CuTe DSL's `cutlass.testing.tune`): for each configuration, `build`
    /// makes the kernel and returns its launch closure (`|stream, workspace|`, as
    /// [`benchmark`] takes it), which is timed with `options` on `stream`; configurations
    /// whose build panics or fails ([`CantImplement`](super::CantImplement)) or whose launches
    /// fail are skipped. Returns the fastest with its time in microseconds.
    pub fn tune<'c, C, L>(
        stream: &CudaStream<'c>,
        configs: impl IntoIterator<Item = C>,
        options: &BenchmarkOptions,
        mut build: impl FnMut(&C) -> Result<L, BenchError>,
    ) -> Result<Tuned<C>, TuneError>
    where
        C: Clone + std::fmt::Debug,
        L: FnMut(&CudaStream<'c>, usize) -> Result<(), BenchError>,
    {
        tune_by(configs, |c| {
            let launch = build(c)?;
            benchmark(stream, options, launch)
        })
    }

    impl<C: Clone + std::fmt::Debug> Autotuner<C> {
        /// The winner for `key`, tuning on the GPU with [`tune`] if there is none.
        pub fn get_or_tune<'c, L>(
            &mut self,
            stream: &CudaStream<'c>,
            key: &TuneKey,
            mut build: impl FnMut(&C) -> Result<L, BenchError>,
        ) -> Result<&Tuned<C>, TuneError>
        where
            L: FnMut(&CudaStream<'c>, usize) -> Result<(), BenchError>,
        {
            let options = self.options.clone();
            self.get_or_tune_by(key, |c| {
                let launch = build(c)?;
                benchmark(stream, &options, launch)
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu::tune;

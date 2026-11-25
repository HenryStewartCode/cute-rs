//! The benchmark and autotune helpers without a GPU: the workspace count, the benchmark
//! options, `tune_by`'s selection (skipping configurations that panic, fail or cannot be
//! implemented) and the `Autotuner`'s cache, which tunes again only when a watched part of
//! the key changes. `tests/gpu_benchmark.rs` times real kernels.

use std::cell::Cell;

use cute::prelude::*;
use cute::testing::{Autotuner, BenchmarkOptions, CantImplement, TuneKey, tune_by, workspace_count_for_l2};

#[test]
fn workspace_count_covers_three_l2_caches() {
    // min(warmup + iterations, 3 * L2 / workspace + 1)
    let l2 = 32 << 20; // the SM89 test GPU's 32 MiB
    assert_eq!(workspace_count_for_l2(l2, 12 << 20, 10, 100), 9);
    assert_eq!(workspace_count_for_l2(l2, 96 << 20, 10, 100), 2);
    assert_eq!(workspace_count_for_l2(l2, 4 << 10, 10, 100), 110);
    assert_eq!(workspace_count_for_l2(l2, 1 << 20, 2, 3), 5);
    assert_eq!(workspace_count_for_l2(l2, 0, 10, 100), 1);
    assert_eq!(workspace_count_for_l2(0, 1 << 20, 10, 100), 1);
}

#[test]
fn benchmark_options_default_to_cute_dsls() {
    let o = BenchmarkOptions::default();
    assert_eq!((o.warmup_iterations, o.iterations, o.workspace_count, o.use_cuda_graphs), (10, 100, 1, false));
    let o = BenchmarkOptions::new().warmup_iterations(2).iterations(5).workspace_count(3).use_cuda_graphs(true);
    assert_eq!((o.warmup_iterations, o.iterations, o.workspace_count, o.use_cuda_graphs), (2, 5, 3, true));
}

/// The fastest configuration wins; a panic, an error, `CantImplement` and a NaN time are
/// skipped and recorded; ties go to the earlier configuration.
#[test]
fn tune_by_skips_what_cannot_run_and_keeps_the_fastest() {
    let tuned = tune_by([1usize, 2, 3, 4, 5, 6], |&c| match c {
        1 => panic!("tile 1 does not divide the layout"),
        2 => Err(CantImplement::new("2 is not supported").into()),
        3 => Ok(9.0),
        4 => Ok(f64::NAN),
        5 => Ok(4.5),
        _ => Ok(4.5),
    })
    .unwrap();
    assert_eq!((tuned.config, tuned.time_us), (5, 4.5));
    let skipped: Vec<(usize, &str)> = tuned.skipped().map(|(c, why)| (*c, why)).collect();
    assert_eq!(skipped.len(), 3, "{skipped:?}");
    assert!(skipped[0].1.contains("panicked: tile 1 does not divide"), "{skipped:?}");
    assert_eq!(skipped[1], (2, "2 is not supported"));
    assert!(skipped[2].1.contains("NaN"), "{skipped:?}");
    assert_eq!(tuned.trials.len(), 6);
    assert_eq!(CantImplement::new("").to_string(), "the configuration cannot be implemented");
}

#[test]
fn tune_by_fails_when_nothing_runs() {
    let e = tune_by(["a", "b"], |c| Err(CantImplement::new(format!("no {c}")).into())).unwrap_err();
    assert_eq!(e.trials, [("\"a\"".to_string(), "no a".to_string()), ("\"b\"".to_string(), "no b".to_string())]);
    assert!(e.to_string().contains("no configuration could be timed"), "{e}");
    let e = tune_by(Vec::<u8>::new(), |_| Ok(1.0)).unwrap_err();
    assert_eq!(e.to_string(), "tune: no configurations to try");
}

/// The winner is kept per cache key and watched part; an unwatched part does not tune again.
#[test]
fn the_autotuner_caches_the_winner_and_retunes_on_change() {
    let mut tuner = Autotuner::new([64usize, 128]).update_on_change(&["M", "N"]);
    let measured = Cell::new(0);
    // A measure that favours 128-bit copies for wide rows and 64-bit ones otherwise.
    let tune = |tuner: &mut Autotuner<usize>, key: &TuneKey, n: usize| {
        tuner
            .get_or_tune_by(key, |&bits| {
                measured.set(measured.get() + 1);
                Ok(if (bits == 128) == (n >= 512) { 1.0 } else { 2.0 })
            })
            .unwrap()
            .config
    };
    let spec = make_fake_tensor::<f32>((sym_int32(1, None), sym_int32(4, None)), (sym_int32(4, None), 1));
    let key =
        |m: usize, n: usize, run: usize| TuneKey::new(&spec.cache_key()).part("M", m).part("N", n).part("run", run);
    assert_eq!(tune(&mut tuner, &key(1024, 1024, 0), 1024), 128);
    assert_eq!((tuner.tunings(), measured.get()), (1, 2));
    // The same problem, and an unwatched part changed: the cached winner.
    assert_eq!(tune(&mut tuner, &key(1024, 1024, 1), 1024), 128);
    assert_eq!((tuner.tunings(), measured.get()), (1, 2));
    // N changed: tuned again.
    assert_eq!(tune(&mut tuner, &key(1024, 256, 2), 256), 64);
    assert_eq!((tuner.tunings(), tuner.len()), (2, 2));
    // Back to the first size: cached.
    assert_eq!(tune(&mut tuner, &key(1024, 1024, 3), 1024), 128);
    assert_eq!(tuner.tunings(), 2);
    assert_eq!(tuner.get(&key(1024, 256, 9)).map(|t| t.config), Some(64));
    // Another cache key (a spec of another element type): tuned again.
    let other = make_fake_tensor::<F16>((sym_int32(1, None),), (1,));
    let k = TuneKey::new(&other.cache_key()).part("M", 1024).part("N", 1024);
    assert_eq!(tune(&mut tuner, &k, 1024), 128);
    assert_eq!(tuner.tunings(), 3);
    // Without update_on_change only the cache key counts.
    let mut plain = Autotuner::new([64usize, 128]);
    tune(&mut plain, &key(1024, 1024, 0), 1024);
    tune(&mut plain, &key(8, 8, 0), 8);
    assert_eq!(plain.tunings(), 1);
    plain.clear();
    assert!(plain.is_empty());
}

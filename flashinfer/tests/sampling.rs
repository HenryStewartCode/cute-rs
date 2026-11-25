// SPDX-License-Identifier: Apache-2.0

//! The samplers on the simulator (with cute-rs's race checks): deterministic edge cases,
//! samples always inside the top-k / top-p set and equal to the f64 emulation with the same
//! Philox draws, a chi-squared frequency check, and Philox against known test vectors.

use cute::prelude::*;
use flashinfer::sampling::*;

const MODES: [Mode; 4] = [Mode::Plain, Mode::TopK, Mode::TopP, Mode::TopKTopP];

/// Random123's known-answer vectors for Philox4x32-10 (`kat_vectors`).
#[test]
fn philox_matches_random123() {
    assert_eq!(philox4x32_10([0; 4], [0; 2]), [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
    assert_eq!(philox4x32_10([u32::MAX; 4], [u32::MAX; 2]), [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]);
    assert_eq!(
        philox4x32_10([0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344], [0xa409_3822, 0x299f_31d0]),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
}

/// `curand_init(seed, subsequence, offset)` then five `curand()` and, after a second
/// `curand_init`, five `curand_uniform()` (as f32 bits), printed by curand from CUDA 12.8 on
/// an RTX 2000 Ada. The first block is Random123's all-zero vector.
#[test]
fn philox_matches_curand() {
    for (&(seed, sub, offset, words), bits) in CURAND_WORDS.iter().zip(CURAND_UNIFORM_BITS) {
        for i in 0..5 {
            assert_eq!(
                curand_word(seed, sub, offset + i as u64),
                words[i],
                "seed {seed} sub {sub} offset {offset} word {i}"
            );
            assert_eq!(curand_uniform(seed, sub, offset, i as u64).to_bits(), bits[i], "uniform {i}");
        }
    }
}

const CURAND_WORDS: &[(u64, u64, u64, [u32; 5])] = &[
    (0, 0, 0, [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8, 0xf8e4cca4]),
    (42, 0, 0, [0x9ceaf053, 0x77f5493b, 0x12bf50ad, 0x5742b3d7, 0xfcdb2127]),
    (42, 3, 6, [0xa9b1302c, 0x3faa46e6, 0x5229c193, 0xe58a6d8c, 0x821f0d02]),
    (0x1234567890abcdef, 77, (1 << 34) + 3, [0xf02fa879, 0xe622fcde, 0xd4154504, 0xdf994d88, 0x73f69e4e]),
    (12345, 1000000, 1, [0x503228c3, 0x4260f8e9, 0x2ba84a90, 0xd766e0bc, 0x71b73c3a]),
];

const CURAND_UNIFORM_BITS: [[u32; 5]; 5] = [
    [0x3ecc4fd2, 0x3f6169c6, 0x3f3c57ac, 0x3f1b00dc, 0x3f78e4cd],
    [0x3f1ceaf0, 0x3eefea92, 0x3d95fa85, 0x3eae8568, 0x3f7cdb21],
    [0x3f29b130, 0x3e7ea91c, 0x3ea45383, 0x3f658a6e, 0x3f021f0d],
    [0x3f702fa8, 0x3f6622fd, 0x3f541545, 0x3f5f994e, 0x3ee7ed3d],
    [0x3ea06452, 0x3e84c1f2, 0x3e2ea12a, 0x3f5766e1, 0x3ee36e78],
];

/// The device Philox equals the host one (several blocks, an offset that is not a multiple
/// of 4 and that crosses into the high counter word).
#[test]
fn device_philox_matches_host() {
    let k = uniform_kernel();
    for (seed, offset) in [(0u64, 0u64), (42, 7), (0x1234_5678_9abc_def0, (1u64 << 34) + 3)] {
        let (blocks, n) = (3usize, 9usize);
        let mut args = vec![
            KernelArg::buffer(&vec![0f32; blocks * n]),
            KernelArg::ScalarI32(n as i32),
            KernelArg::Scalar { ty: DType::U64, bits: seed },
            KernelArg::Scalar { ty: DType::U64, bits: offset },
        ];
        k.simulate(LaunchConfig::new([blocks as u32, 1, 1], [1, 1, 1]), &mut args).unwrap();
        let got = args[0].to_vec::<f32>();
        for b in 0..blocks {
            for i in 0..n {
                assert_eq!(got[b * n + i], curand_uniform(seed, b as u64, offset, i as u64), "block {b} draw {i}");
            }
        }
    }
}

/// top_k = 1 and a tiny top_p give the argmax; a one-hot row gives its token, whatever the
/// mode.
#[test]
fn edge_cases() {
    let (batch, d) = (6, 200);
    let base = Problem::new(batch, d, Dist::Normal(2.0), 1, 1e-7, 3);
    let argmax: Vec<i32> = (0..batch)
        .map(|b| {
            let r = base.row(b);
            (0..d).max_by(|&i, &j| r[i].partial_cmp(&r[j]).unwrap()).unwrap() as i32
        })
        .collect();
    let cfg = |m| Config::new(m).threads(32);
    for seed in [1u64, 2, 3] {
        let p = Problem { seed, ..base.clone() };
        assert_eq!(simulate_checked(&p, cfg(Mode::TopK)), argmax, "top_k = 1");
        assert_eq!(simulate_checked(&p, cfg(Mode::TopP)), argmax, "tiny top_p");
        assert_eq!(simulate_checked(&p, cfg(Mode::TopKTopP)), argmax, "both");
    }
    let mut onehot = vec![0f32; batch * d];
    let hot: Vec<i32> = (0..batch).map(|b| ((b * 37 + 5) % d) as i32).collect();
    for b in 0..batch {
        onehot[b * d + hot[b] as usize] = 1.0;
    }
    let p = Problem::from_probs(batch, d, onehot, 50, 0.9, 9);
    for m in MODES {
        assert_eq!(simulate_checked(&p, cfg(m)), hot, "one-hot, {}", m.name());
    }
}

/// A row with no positive probability is invalid (FlashInfer's `valid = false`, output 0).
#[test]
fn zero_row_is_invalid() {
    let p = Problem::from_probs(2, 8, vec![0.0; 16], 3, 0.5, 1);
    for m in MODES {
        let (out, valid) = simulate(&p, Config::new(m).threads(32));
        assert_eq!((out, valid), (vec![0, 0], vec![0, 0]), "{}", m.name());
    }
}

/// Every sample is inside the top-k / top-p set and agrees with the f64 emulation; several
/// chunks per row (d = 1000 with 64 threads), several distributions, per-row parameters.
#[test]
fn samples_stay_in_set() {
    for (dist, seed) in [(Dist::Normal(1.0), 5u64), (Dist::Normal(5.0), 6), (Dist::Gumbel(1.0), 7)] {
        let mut p = Problem::new(8, 1000, dist, 20, 0.6, seed);
        p.offset = 6;
        p.top_k = vec![1, 2, 5, 20, 50, 1000, 3, 10];
        p.top_p = vec![0.1, 0.3, 0.5, 0.6, 0.9, 1.0, 0.95, 0.2];
        for m in MODES {
            simulate_checked(&p, Config::new(m).threads(64).per_row(true));
        }
        // Scalar parameters.
        let p = Problem { top_k: vec![20; 8], top_p: vec![0.6; 8], ..p };
        simulate_checked(&p, Config::new(Mode::TopKTopP).threads(64));
    }
}

/// Frequencies over many rows (each row is its own Philox subsequence) match the renormalised
/// top-k / top-p distribution: chi-squared below its 0.1% critical value.
#[test]
fn chi_squared_frequencies() {
    let d = 16;
    let w: Vec<f32> = (0..d).map(|i| ((i * 7) % d + 1) as f32).collect();
    let s: f32 = w.iter().sum();
    let row: Vec<f32> = w.iter().map(|x| x / s).collect();
    let n = 2000;
    let probs: Vec<f32> = (0..n).flat_map(|_| row.iter().cloned()).collect();
    // 0.1% critical values of chi-squared for 1..=15 degrees of freedom.
    let crit =
        [10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12, 27.88, 29.59, 31.26, 32.91, 34.53, 36.12, 37.70];
    for (m, k, pp) in [(Mode::Plain, 0, 1.0), (Mode::TopK, 5, 1.0), (Mode::TopP, 16, 0.6), (Mode::TopKTopP, 6, 0.7)] {
        let p = Problem::from_probs(n, d, probs.clone(), k, pp, 11);
        let (out, valid) = simulate(&p, Config::new(m).threads(32));
        assert!(valid.iter().all(|&v| v == 1));
        let mut counts = vec![0usize; d];
        for &t in &out {
            counts[t as usize] += 1;
        }
        let (chi, df) = chi_squared(&counts, &p.target(m, 0));
        assert!(df >= 1 && chi < crit[df - 1], "{}: chi-squared {chi:.2} with {df} dof ({counts:?})", m.name());
    }
}

/// The 16-byte loads and the shuffles survive lowering.
#[test]
fn lowers_to_vector_loads() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for m in MODES {
        let compiled = kernel(Config::new(m)).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap();
        let ptx = compiled.ptx();
        assert!(ptx.contains("ld.global.v4.f32") || ptx.contains("ld.global.v4.b32"), "{ptx}");
        assert!(ptx.contains("shfl.sync.up"), "{ptx}");
    }
}

// ---------------------------------------------------------------------------------------
// min-p sampling, the renorm / mask kernels and softmax.

/// Min-p samples are in the set `p >= min_p * max(p)` and agree with the f64 emulation; per-row
/// and scalar parameters, several chunks per row; `min_p = 1` gives the argmax.
#[test]
fn min_p_samples_stay_in_set() {
    for (dist, seed) in [(Dist::Normal(1.0), 5u64), (Dist::Normal(5.0), 6), (Dist::Gumbel(1.0), 7)] {
        let mut p = Problem::new(8, 1000, dist, 0, 0.0, seed);
        p.offset = 6;
        p.top_p = vec![0.0, 0.01, 0.1, 0.3, 0.5, 0.9, 1.0, 0.05];
        simulate_checked(&p, Config::new(Mode::MinP).threads(64).per_row(true));
        let p = Problem { top_p: vec![0.2; 8], ..p };
        simulate_checked(&p, Config::new(Mode::MinP).threads(64));
    }
    let base = Problem::new(6, 200, Dist::Normal(2.0), 0, 1.0, 3);
    for b in 0..6 {
        let r = base.row(b);
        let argmax = (0..200).max_by(|&i, &j| r[i].partial_cmp(&r[j]).unwrap()).unwrap() as i32;
        assert_eq!(simulate_checked(&base, Config::new(Mode::MinP).threads(32))[b], argmax);
    }
}

/// Frequencies of min-p samples match the renormalised kept set (a few cells drop out).
#[test]
fn min_p_frequencies() {
    let d = 16;
    let w: Vec<f32> = (0..d).map(|i| ((i * 7) % d + 1) as f32).collect();
    let s: f32 = w.iter().sum();
    let row: Vec<f32> = w.iter().map(|x| x / s).collect();
    let n = 2000;
    let probs: Vec<f32> = (0..n).flat_map(|_| row.iter().cloned()).collect();
    let p = Problem::from_probs(n, d, probs, 0, 0.4, 11);
    let (out, valid) = simulate(&p, Config::new(Mode::MinP).threads(32));
    assert!(valid.iter().all(|&v| v == 1));
    let mut counts = vec![0usize; d];
    for &t in &out {
        counts[t as usize] += 1;
    }
    let (chi, df) = chi_squared(&counts, &p.target(Mode::MinP, 0));
    // 0.1% critical values of chi-squared for 1..=10 degrees of freedom.
    let crit = [10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12, 27.88, 29.59];
    assert!((1..=10).contains(&df) && chi < crit[df - 1], "chi-squared {chi:.2} with {df} dof ({counts:?})");
    assert!(counts.iter().zip(&row).all(|(&c, &r)| (r >= 0.4 * 0.1) || c == 0), "{counts:?}");
}

const OPS: [Op; 4] = [Op::TopPRenorm, Op::TopKRenorm, Op::TopKMask, Op::Softmax];

/// Every operation against its f64 reference: several chunks per row (d = 1000 with 64
/// threads: 4 buckets per thread in the scan), a row shorter than the block (d = 40), a row
/// longer than the block's first chunk, scalar and per-row parameters.
#[test]
fn ops_match_reference() {
    for op in OPS {
        for (batch, d, threads, dist) in [
            (3usize, 1000usize, 64usize, Dist::Normal(1.0)),
            (2, 40, 64, Dist::Normal(3.0)),
            (3, 2052, 256, Dist::Gumbel(0.5)),
        ] {
            let (k, param) = match op {
                Op::Softmax => (0, 0.7),
                _ => (30, 0.8),
            };
            let p = OpProblem::new(op, batch, d, dist, k, param);
            simulate_op_checked(&p, OpConfig::new(op).threads(threads));
            let mut p = p;
            p.top_k = (0..batch as i32).map(|b| 1 + b * 13).collect();
            p.param =
                (0..batch).map(|b| if op == Op::Softmax { 0.5 + b as f32 } else { 0.3 + 0.2 * b as f32 }).collect();
            simulate_op_checked(&p, OpConfig::new(op).threads(threads).per_row(true));
        }
    }
}

/// Edges: `k = 1`, `k >= d` (keep everything), `top_p` around 1, a one-hot row, ties at the
/// threshold (kept together), softmax with `-inf` entries and temperature zero.
#[test]
fn op_edge_cases() {
    let cfg = |op| OpConfig::new(op).threads(64);
    // k = 1 and k >= d.
    for op in [Op::TopKRenorm, Op::TopKMask] {
        for k in [1, 7, 199, 200, 5000] {
            simulate_op_checked(&OpProblem::new(op, 2, 200, Dist::Normal(2.0), k, 0.5), cfg(op));
        }
    }
    for pp in [1e-6f32, 0.999, 1.0, 1.5] {
        simulate_op_checked(&OpProblem::new(Op::TopPRenorm, 2, 200, Dist::Normal(2.0), 0, pp), cfg(Op::TopPRenorm));
    }
    // One-hot row: top-p and top-k renorm give back the one-hot.
    let mut onehot = OpProblem::new(Op::TopPRenorm, 1, 64, Dist::Normal(1.0), 3, 0.5);
    onehot.input = (0..64).map(|i| if i == 17 { 1.0 } else { 0.0 }).collect();
    for op in [Op::TopPRenorm, Op::TopKRenorm] {
        let out = simulate_op(&onehot, cfg(op));
        assert_eq!(out[17], 1.0);
        assert_eq!(out.iter().filter(|&&x| x != 0.0).count(), 1);
    }
    // Ties at the threshold: 5 equal values, k = 3 keeps all five.
    let mut tie = OpProblem::new(Op::TopKMask, 1, 64, Dist::Normal(1.0), 3, 0.0);
    tie.input = (0..64).map(|i| if i % 13 == 0 { 2.5 } else { -(i as f32) * 0.01 }).collect();
    let out = simulate_op(&tie, cfg(Op::TopKMask));
    assert_eq!(out.iter().filter(|&&x| x.is_finite()).count(), 5);
    simulate_op_checked(&tie, cfg(Op::TopKMask));
    // Softmax of a row with masked entries, and with temperature zero (uniform).
    let mut m = OpProblem::new(Op::Softmax, 2, 100, Dist::Normal(1.0), 0, 0.9);
    for (i, x) in m.input.iter_mut().enumerate() {
        if i % 3 != 0 {
            *x = f32::NEG_INFINITY;
        }
    }
    simulate_op_checked(&m, cfg(Op::Softmax));
    m.param = vec![0.0; 2];
    let out = simulate_op(&OpProblem { input: (0..200).map(|i| i as f32).collect(), ..m }, cfg(Op::Softmax));
    assert!(out.iter().all(|&x| (x - 0.01).abs() < 1e-7), "{out:?}");
}

/// The masked logits feed softmax to the same result as `top_k_renorm_probs` of the softmax
/// (FlashInfer's documented identity), up to rounding.
#[test]
fn mask_then_softmax_equals_renorm() {
    let (batch, d, k) = (2, 1000, 25);
    let logits = OpProblem::new(Op::TopKMask, batch, d, Dist::Normal(2.0), k, 1.0);
    let masked = simulate_op(&logits, OpConfig::new(Op::TopKMask).threads(64));
    let sm = simulate_op(&OpProblem { input: masked, ..logits.clone() }, OpConfig::new(Op::Softmax).threads(64));
    let probs = simulate_op(
        &OpProblem { input: logits.input.clone(), ..logits.clone() },
        OpConfig::new(Op::Softmax).threads(64),
    );
    let renorm = simulate_op(&OpProblem { input: probs, ..logits }, OpConfig::new(Op::TopKRenorm).threads(64));
    for (a, b) in sm.iter().zip(&renorm) {
        assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
    }
}

/// The new kernels keep their 16-byte accesses and (softmax) fast exponentials.
#[test]
fn ops_lower_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for op in OPS {
        let compiled =
            op_kernel(OpConfig::new(op)).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap();
        let ptx = compiled.ptx();
        assert!(ptx.contains("ld.global.v4.f32") || ptx.contains("ld.global.v4.b32"), "{} {ptx}", op.name());
        assert!(ptx.contains("st.global.v4"), "{} {ptx}", op.name());
        if op == Op::Softmax {
            assert!(ptx.contains("ex2.approx.ftz"), "{ptx}");
        } else {
            assert!(ptx.contains("atom.relaxed.gpu.shared.add"), "{} {ptx}", op.name());
        }
    }
}

/// Rows split over several blocks agree with the reference (including ties, `k >= d`,
/// per-row parameters, a slice shorter than a chunk and a split that does not divide the row).
#[test]
fn split_rows_match_reference() {
    for op in [Op::Softmax, Op::TopKMask, Op::TopKRenorm, Op::TopPRenorm] {
        for (batch, d, threads, split, dist) in [
            (2usize, 1000usize, 64usize, 4usize, Dist::Normal(1.0)),
            (1, 2052, 64, 3, Dist::Gumbel(0.5)),
            (2, 40, 64, 5, Dist::Normal(3.0)),
        ] {
            let k = if op == Op::Softmax { 0 } else { 30 };
            let mut p = OpProblem::new(op, batch, d, dist, k, 0.7);
            let cfg = OpConfig::new(op).threads(threads).split(split);
            simulate_op_checked(&p, cfg);
            p.top_k = (0..batch as i32).map(|b| 1 + b * 13).collect();
            p.param = (0..batch).map(|b| 0.5 + b as f32).collect();
            simulate_op_checked(&p, cfg.per_row(true));
        }
        let mut p = OpProblem::new(op, 2, 256, Dist::Normal(1.0), 1, 1.0);
        p.top_k = vec![1000, 1];
        simulate_op_checked(&p, OpConfig::new(op).threads(64).split(4).per_row(true));
    }
}

/// `top_p_renorm_probs` over several blocks per row, with `top_p` per row inside (0, 1).
#[test]
fn top_p_split_rows() {
    for (batch, d, split) in [(3usize, 1000usize, 4usize), (2, 4100, 7)] {
        let mut p = OpProblem::new(Op::TopPRenorm, batch, d, Dist::Normal(2.0), 0, 0.9);
        p.param = (0..batch).map(|b| 0.2 + 0.3 * b as f32).collect();
        simulate_op_checked(&p, OpConfig::new(Op::TopPRenorm).threads(64).split(split).per_row(true));
    }
}

/// Chain speculative sampling: first rejected position, counts, accepted prefix and padding
/// match the Philox-exact emulation, and the resampled token is the one its draw selects. Covers
/// accept-all (bonus token), reject-at-0, more than 32 draft tokens and a non-zero offset.
#[test]
fn speculative_matches_reference() {
    use flashinfer::sampling::spec;
    // Draft and target close (mostly accepted), and far apart (mostly rejected).
    for (batch, n, d, dd, td, seed, off) in [
        (6usize, 3usize, 300usize, Dist::Normal(1.0), Dist::Normal(1.0), 3u64, 0u64),
        (8, 4, 1000, Dist::Normal(3.0), Dist::Gumbel(0.5), 4, 7),
        (5, 40, 64, Dist::Normal(0.1), Dist::Normal(0.1), 5, 2),
        (4, 1, 256, Dist::Normal(2.0), Dist::Normal(2.0), 6, 1),
    ] {
        let mut p = spec::Problem::new(batch, n, d, dd, td, seed);
        p.offset = off;
        let out = spec::simulate_checked(&p, 64);
        // A mix of outcomes is exercised.
        let rejects = (0..batch).filter(|&b| p.reference(b).0 < n).count();
        println!("batch {batch} n {n}: {rejects} rows rejected a draft token; first row {:?}", &out[..n + 1]);
    }
}

/// All draft tokens accepted (draft equals target, with the bonus row): the bonus token is
/// sampled from `target[n]` and nothing is rejected early.
#[test]
fn speculative_bonus_and_reject_first() {
    use flashinfer::sampling::spec;
    let mut p = spec::Problem::new(16, 2, 128, Dist::Normal(1.0), Dist::Normal(1.0), 9);
    // Draft token probabilities near zero in the target: every first token is rejected.
    for b in 0..p.batch {
        let id = p.draft_ids[b * 2] as usize;
        p.target_probs[(b * 3) * 128 + id] = 0.0;
    }
    let (out, _, em) = spec::simulate(&p, 32);
    assert!(em.iter().all(|&e| e == 5), "emitted {em:?}");
    for b in 0..p.batch {
        assert_eq!(&out[b * 3 + 1..b * 3 + 3], &[-1, -1]);
    }
    // Target = draft everywhere, and the bonus row: u * p < p fails only for u == 1.
    let mut q = spec::Problem::new(16, 2, 128, Dist::Normal(1.0), Dist::Normal(1.0), 9);
    for b in 0..q.batch {
        for i in 0..2 {
            let src = q.draft_probs[(b * 2 + i) * 128..(b * 2 + i + 1) * 128].to_vec();
            q.target_probs[(b * 3 + i) * 128..(b * 3 + i + 1) * 128].copy_from_slice(&src);
        }
    }
    spec::simulate_checked(&q, 32);
    let (_, acc, em) = spec::simulate(&q, 32);
    assert!(acc.iter().zip(&em).all(|(a, e)| a >= &3 && e >= &5));
}

/// Speculative decoding is lossless: the first output token of every row is distributed as the
/// target row (chi-squared below its 0.1% critical value), and the resampled token after a
/// rejection as `relu(target - draft)`.
#[test]
fn speculative_frequencies() {
    use flashinfer::sampling::spec;
    let d = 16;
    let (n, batch) = (1usize, 3000usize);
    let mut p = spec::Problem::new(batch, n, d, Dist::Normal(1.0), Dist::Normal(1.0), 21);
    let (tgt, drf): (Vec<f32>, Vec<f32>) = {
        let w = |salt: usize| -> Vec<f32> {
            let w: Vec<f32> = (0..d).map(|i| (((i * salt) % d) + 1) as f32).collect();
            let s: f32 = w.iter().sum();
            w.iter().map(|x| x / s).collect()
        };
        (w(7), w(3))
    };
    p.target_probs = (0..batch * (n + 1)).flat_map(|_| tgt.clone()).collect();
    p.draft_probs = (0..batch * n).flat_map(|_| drf.clone()).collect();
    // Draft tokens from the draft row (stratified over rows).
    p.draft_ids = (0..batch)
        .map(|r| {
            let x = (r as f64 + 0.5) / batch as f64;
            let mut c = 0.0;
            drf.iter()
                .position(|&q| {
                    c += q as f64;
                    c > x
                })
                .unwrap_or(d - 1) as i32
        })
        .collect();
    let (out, _, _) = spec::simulate(&p, 32);
    let crit =
        [10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12, 27.88, 29.59, 31.26, 32.91, 34.53, 36.12, 37.70];
    let mut counts = vec![0usize; d];
    for b in 0..batch {
        let o = &out[b * (n + 1)..(b + 1) * (n + 1)];
        counts[o[0] as usize] += 1;
    }
    let target: Vec<f64> = tgt.iter().map(|&x| x as f64).collect();
    let (chi, df) = chi_squared(&counts, &target);
    assert!(chi < crit[df - 1], "first token: chi-squared {chi:.2} with {df} dof ({counts:?})");
}

/// The kernel lowers to vector loads and warp reductions.
#[test]
fn speculative_lowers() {
    use flashinfer::sampling::spec;
    let Some(tc) = Toolchain::discover() else { return };
    let ptx = spec::kernel(1024)
        .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
        .unwrap_or_else(|e| panic!("{e}"))
        .ptx()
        .to_string();
    assert!(ptx.matches("ld.global.v4").count() >= 2, "no 16-byte loads");
}

/// `top_k`: the k largest values (as a multiset) with distinct, matching indices, for logits with
/// heavy ties at the threshold, negative values, `k = 1`, `k = d` and a row with every value equal.
#[test]
fn top_k_matches_reference() {
    for (batch, d, k, dist) in [
        (3usize, 1000usize, 20i32, Dist::Normal(2.0)),
        (2, 300, 1, Dist::Gumbel(1.0)),
        (2, 64, 64, Dist::Normal(1.0)),
        (2, 2048, 700, Dist::Normal(0.5)),
    ] {
        let p = OpProblem::new(Op::TopK, batch, d, dist, k, 0.0);
        simulate_top_k(&p, OpConfig::new(Op::TopK).threads(64));
    }
    // Quantised values: many equal to the k-th.
    let mut p = OpProblem::new(Op::TopK, 3, 512, Dist::Normal(1.0), 37, 0.0);
    for x in p.input.iter_mut() {
        *x = (*x * 2.0).round() / 2.0;
    }
    p.input[..512].fill(-3.0);
    simulate_top_k(&p, OpConfig::new(Op::TopK).threads(128));
}

/// `top_k` over several blocks per row, including a row of equal values (all ties) and
/// quantised data (ties at the threshold spread over blocks).
#[test]
fn top_k_split_rows() {
    for (batch, d, k, split) in [(2usize, 1000usize, 20i32, 4usize), (1, 2048, 300, 8), (3, 256, 256, 3)] {
        let p = OpProblem::new(Op::TopK, batch, d, Dist::Gumbel(0.3), k, 0.0);
        simulate_top_k(&p, OpConfig::new(Op::TopK).threads(64).split(split));
    }
    let mut p = OpProblem::new(Op::TopK, 2, 1024, Dist::Normal(1.0), 41, 0.0);
    for x in p.input.iter_mut() {
        *x = (*x * 2.0).round() / 2.0;
    }
    p.input[..1024].fill(0.5);
    simulate_top_k(&p, OpConfig::new(Op::TopK).threads(64).split(4));
}

// ---------------------------------------------------------------------------------------
// Per-row seeds, sampling from logits.

/// Per-row `seed` / `offset` arrays: samples agree with the reference under each row's own
/// stream, and with constant arrays equal the scalar launch.
#[test]
fn per_row_seeds() {
    let mut p = Problem::new(6, 200, Dist::Normal(2.0), 20, 0.7, 4);
    p.offset = 8;
    p.seeds = (0..6u64).map(|b| 100 + 7 * b).collect();
    p.offsets = (0..6u64).map(|b| b * 3).collect();
    for m in MODES {
        let cfg = Config::new(m).threads(32);
        let (out, valid) = simulate(&p, cfg.per_seed(true));
        p.check(m, &out, &valid);
        let scalar = simulate(&p, cfg);
        let constant = Problem { seeds: vec![p.seed; 6], offsets: vec![p.offset; 6], ..p.clone() };
        assert_eq!(simulate(&constant, cfg.per_seed(true)), scalar, "{}", m.name());
    }
}

/// `sampling_from_logits`: each token equals the f64-free reference (`argmax(logit + gumbel)` from
/// the same Philox words), several chunks per row, scalar and per-row seeds.
#[test]
fn logits_match_reference() {
    for (dist, d, threads) in [(Dist::Normal(2.0), 200usize, 32usize), (Dist::Gumbel(0.5), 1000, 64)] {
        let mut p = LogitsProblem::new(8, d, dist, 17);
        p.offset = 8;
        let cfg = Config::new(Mode::Plain).threads(threads);
        p.check(&simulate_logits(&p, cfg));
        p.seeds = (0..8u64).map(|b| 3 * b + 1).collect();
        p.offsets = (0..8u64).map(|b| 4 * b).collect();
        p.check(&simulate_logits(&p, cfg.per_seed(true)));
    }
}

/// Large logits dominate the noise; a row of `-inf` gives the last token (when the row fills
/// the block's chunks; padded slots past `d` give 0, as in FlashInfer).
#[test]
fn logits_edge_cases() {
    let (batch, d) = (3, 128);
    let mut p = LogitsProblem::new(batch, d, Dist::Normal(1.0), 5);
    for b in 0..batch {
        p.logits[b * d + 11 * b + 3] = 1000.0;
    }
    let cfg = Config::new(Mode::Plain).threads(32);
    assert_eq!(simulate_logits(&p, cfg), vec![3, 14, 25]);
    p.logits[d..2 * d].fill(f32::NEG_INFINITY);
    assert_eq!(simulate_logits(&p, cfg)[1], d as i32 - 1);
}

/// Frequencies over many rows match `softmax(logits)`: chi-squared below its 0.1% critical value.
#[test]
fn logits_frequencies() {
    let d = 16;
    let row: Vec<f32> = (0..d).map(|i| ((i * 7) % d) as f32 * 0.25).collect();
    let n = 2000;
    let mut p = LogitsProblem::new(n, d, Dist::Normal(1.0), 23);
    p.logits = (0..n).flat_map(|_| row.iter().cloned()).collect();
    let out = simulate_logits(&p, Config::new(Mode::Plain).threads(32));
    let mut counts = vec![0usize; d];
    for &t in &out {
        counts[t as usize] += 1;
    }
    let target: Vec<f64> =
        softmax_rows(&row.iter().map(|&x| x as f64).collect::<Vec<_>>(), d).iter().map(|&x| x as f64).collect();
    let (chi, df) = chi_squared(&counts, &target);
    assert!(chi < 37.70 && df == 15, "chi-squared {chi:.2} with {df} dof ({counts:?})");
}

/// The Gumbel kernel lowers to vector loads and the fast `lg2`.
#[test]
fn logits_lowers() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let compiled =
        logits_kernel(Config::new(Mode::Plain)).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap();
    let ptx = compiled.ptx();
    assert!(ptx.contains("ld.global.v4.f32") || ptx.contains("ld.global.v4.b32"), "{ptx}");
    assert!(ptx.contains("lg2.approx"), "{ptx}");
}

/// The compact top-k-first kernels: softmax of the `k` logits into a zero-padded row, and the
/// map from a local sample to its token.
#[test]
fn compact_kernels() {
    let (batch, k) = (3usize, 10usize);
    let kp = compact_width(k);
    assert_eq!(kp, 12);
    let vals: Vec<f32> = (0..batch * k).map(|i| ((i * 5) % 7) as f32 * 0.5 - 1.0).collect();
    let kern = compact_softmax_kernel();
    let mut args = vec![
        KernelArg::buffer(&vals),
        KernelArg::buffer(&vec![-1.0f32; batch * kp]),
        KernelArg::ScalarI32(k as i32),
        KernelArg::ScalarI32(kp as i32),
    ];
    kern.simulate(LaunchConfig::new([batch as u32, 1, 1], [COMPACT_THREADS as u32, 1, 1]), &mut args).unwrap();
    let got = args[1].to_vec::<f32>();
    for b in 0..batch {
        let want = softmax_rows(&vals[b * k..(b + 1) * k].iter().map(|&x| x as f64).collect::<Vec<_>>(), k);
        for i in 0..kp {
            let w = if i < k { want[i] } else { 0.0 };
            assert!((got[b * kp + i] - w).abs() < 1e-6, "row {b} col {i}: {} vs {w}", got[b * kp + i]);
        }
    }
    let indices: Vec<i32> = (0..batch * k).map(|i| 100 + i as i32).collect();
    let mut args = vec![
        KernelArg::buffer(&indices),
        KernelArg::buffer(&[3i32, 0, 9]),
        KernelArg::buffer(&[1i32, 1, 0]),
        KernelArg::buffer(&[-1i32; 3]),
        KernelArg::ScalarI32(k as i32),
    ];
    gather_kernel().simulate(LaunchConfig::new([3, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[3].to_vec::<i32>(), vec![103, 110, 0]);
}

// ---------------------------------------------------------------------------------------
// Top-k with an index transform.

/// Row lengths: shorter than, equal to and longer than `k`, even a window of 0, with row starts
/// that are not multiples of 4, one and several blocks per row.
#[test]
fn top_k_transform_ragged() {
    use flashinfer::sampling::transform::*;
    let lengths = vec![1000, 37, 20, 0, 21, 513, 20, 300];
    let starts = vec![3, 0, 5, 7, 1, 2, 0, 9];
    for split in [1usize, 3] {
        let p = TProblem::new(Kind::Ragged, 1030, 20, Dist::Normal(1.0), lengths.clone(), starts.clone(), 8, 0);
        simulate_checked(&p, TConfig::new(Kind::Ragged).threads(64).split(split));
    }
    // A shorter k than every row and k = 1.
    for k in [1usize, 7] {
        let p = TProblem::new(Kind::Ragged, 1030, k, Dist::Gumbel(0.5), lengths.clone(), starts.clone(), 8, 0);
        simulate_checked(&p, TConfig::new(Kind::Ragged).threads(128).split(2));
    }
}

/// Page tables: page sizes 1, 4 and 64, a row-to-batch mapping that shares sequences between
/// rows, and quantised scores (many ties at the k-th) over one and several blocks.
#[test]
fn top_k_transform_page_table() {
    use flashinfer::sampling::transform::*;
    let lengths = vec![900, 64, 17, 250, 1];
    let starts = vec![0, 128, 3, 64, 0];
    for shift in [0u32, 2, 6] {
        for split in [1usize, 4] {
            let mut p =
                TProblem::new(Kind::PageTable, 1024, 16, Dist::Normal(1.0), lengths.clone(), starts.clone(), 3, shift);
            for x in p.input.iter_mut() {
                *x = (*x * 2.0).round() / 2.0;
            }
            simulate_checked(&p, TConfig::new(Kind::PageTable).threads(64).split(split).page_shift(shift));
        }
    }
}

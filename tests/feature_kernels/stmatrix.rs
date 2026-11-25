// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The epilogue and warp-specialised kernels follow the design of NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py and
//   dense_gemm_persistent.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! `stmatrix` atoms and `setmaxnreg` (roadmap part m8-stmatrix-setmaxnreg): every form
//! held to a Rust model of the PTX ISA's fragment tables, a dense-GEMM-style epilogue, and a
//! warp-specialised kernel that moves registers between warpgroups.

use cute::prelude::*;

use super::Case;

/// The `.m8n8.b16` forms: (matrices, transposed, swizzled shared memory).
const B16_FORMS: [(usize, bool, bool); 6] =
    [(1, false, false), (2, false, true), (4, false, true), (1, true, false), (2, true, true), (4, true, false)];

/// Elements of all forms' images (8 rows of 8 per matrix).
const B16_ELEMS: usize = 64 * (1 + 2 + 4 + 1 + 2 + 4);

/// The shared-memory element (`64 m + 8 row + col`) that value `v` of lane `l` of an
/// `stmatrix.m8n8.x{num}` lands at, from the PTX ISA's fragment table: register `m = v / 2`
/// holds row `l / 4`, columns `2 (l % 4)` and `+1` of matrix `m` (the half `v % 2`); a
/// transposed store swaps row and column.
pub fn stsm_b16_offset(l: usize, v: usize, trans: bool) -> usize {
    let (m, j) = (v / 2, v % 2);
    let (r, c) = (l / 4, 2 * (l % 4) + j);
    let (mr, mc) = if trans { (c, r) } else { (r, c) };
    64 * m + 8 * mr + mc
}

/// The shared-memory byte (`128 m + 16 row + byte`) that value `v` of lane `l` of an
/// `stmatrix.m16n8.x{num}.trans.b8` lands at: byte `q = v % 4` of register `m = v / 4` is
/// row `l / 4 + 8 (q / 2)`, column `2 (l % 4) + q % 2` of the 16x8 matrix `m` (the `mma`
/// accumulator fragment), stored transposed: column `c` is the 16-byte row `c`.
pub fn stsm_b8_offset(l: usize, v: usize) -> usize {
    let (m, q) = (v / 4, v % 4);
    let (r, c) = (l / 4 + 8 * (q / 2), 2 * (l % 4) + q % 2);
    128 * m + 16 * c + r
}

/// Every `.m8n8.b16` form, into plain and 128-byte-swizzled shared memory: each lane's
/// values (from `a`) are stored with `StMatrix8x8x16bOp` through a tiled copy of the atom's
/// own layout, read back with plain loads (`plain`, in the image's element order) and with
/// the matching `ldmatrix` (`back`, which must give `a` again).
pub fn stmatrix_forms() -> Case {
    let kernel = Kernel::builder("stmatrix_forms").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::col_major(B16_ELEMS));
        let plain = param_tensor::<F16>("plain", Layout::col_major(B16_ELEMS));
        let back = param_tensor::<F16>("back", Layout::col_major(B16_ELEMS));
        let lane = thread_idx().x;
        let (mut base, mut vbase) = (0i32, 0i32);
        for (num, trans, swizzled) in B16_FORMS {
            let rows = 8 * num;
            let layout = Layout::col_major((8, rows));
            let smem = if swizzled {
                make_smem_tensor::<F16>(make_composed_layout(Swizzle::new(3, 3, 3), 0, layout))
            } else {
                make_smem_tensor::<F16>(layout)
            };
            let values = 2 * num as i32;
            let atom = make_copy_atom::<F16>(CopyOp::StMatrix { num, trans });
            let tiled = make_tiled_copy(&atom, atom.layout_ref_tv(), (8, rows));
            let frag = make_fragment::<F16>(2 * num);
            for v in 0..2 * num {
                frag.store(v, a.load(vbase + lane * values + v as i32));
            }
            copy(&tiled, &tiled.retile(&frag), &tiled.get_slice(lane).partition_d(&smem));
            sync_threads();
            for i in 0..rows * 8 / 32 {
                let k = lane + 32 * i as i32;
                plain.store(base + k, smem.load((k % 8, k / 8)));
            }
            let latom = make_copy_atom::<F16>(CopyOp::LdMatrix { num, trans });
            let ltiled = make_tiled_copy(&latom, latom.layout_ref_tv(), (8, rows));
            let got = make_fragment::<F16>(2 * num);
            copy(&ltiled, &ltiled.get_slice(lane).partition_s(&smem), &ltiled.retile(&got));
            for v in 0..2 * num {
                back.store(vbase + lane * values + v as i32, got.load(v));
            }
            sync_threads();
            base += (rows * 8) as i32;
            vbase += 32 * values;
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, plain, back) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<F16>());
        assert_eq!(back, a, "ldmatrix reads back what stmatrix stored");
        let (mut base, mut vbase) = (0, 0);
        for (num, trans, swizzled) in B16_FORMS {
            for l in 0..32 {
                for v in 0..2 * num {
                    let want = a[vbase + l * 2 * num + v];
                    let got = plain[base + stsm_b16_offset(l, v, trans)];
                    assert_eq!(got, want, "x{num} trans {trans} swizzled {swizzled}: lane {l} value {v}");
                }
            }
            base += 64 * num;
            vbase += 64 * num;
        }
    }
    Case { name: "stmatrix_forms", kernel, cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]), small_only: false, check }
}

/// The R2S copy of a dense-GEMM epilogue: a `2 x 2`-warp SM80 m16n8k16 TiledMma (each warp
/// 16 x 16 of a 32 x 32 C tile) computes `A B^T`, its accumulators are converted to f16 and
/// stored to a row-major shared C tile by `make_tiled_copy_s(StMatrix8x8x16bOp x4,
/// make_tiled_copy_c_atom(the same atom, mma))`, as CuTe DSL's Hopper `dense_gemm` stages C
/// (with an SM80 MMA in place of WGMMA), then copied out.
pub fn stmatrix_epilogue() -> Case {
    const M: usize = 32;
    const N: usize = 32;
    const K: usize = 16;
    let kernel = Kernel::builder("stmatrix_epilogue").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F16>("b", Layout::row_major((N, K)));
        let c = param_tensor::<F16>("c", Layout::row_major((M, N)));
        let t = thread_idx().x;
        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)))
            .with_permutation_mnk([None, Some(Layout::new(32, 1)), None]);
        let thr_mma = mma.get_slice(t);
        let (fa, fb) = (thr_mma.partition_fragment_a::<F16, _>(&a), thr_mma.partition_fragment_b::<F16, _>(&b));
        let (pa, pb) = (thr_mma.partition_a(&a), thr_mma.partition_b(&b));
        for i in 0..fa.size() {
            fa.store(i, pa.load(i));
        }
        for i in 0..fb.size() {
            fb.store(i, pb.load(i));
        }
        let acc = thr_mma.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0);
        gemm(&mma, &acc, &fa, &fb, &acc);
        let acc16 = make_fragment_like::<F16, f32>(&acc);
        for i in 0..acc.size() {
            acc16.store(i, acc.load(i).to::<F16>());
        }
        let atom = make_copy_atom::<F16>(CopyOp::StMatrix { num: 4, trans: false });
        let c_atom = make_tiled_copy_c_atom(&atom, &mma);
        let r2s = make_tiled_copy_s(&atom, &c_atom);
        let sc = make_smem_tensor::<F16>(Layout::row_major((M, N)));
        copy(&r2s, &r2s.retile(&acc16), &r2s.get_slice(t).partition_d(&sc));
        sync_threads();
        for i in 0..M * N / 128 {
            let k = t + 128 * i as i32;
            c.store((k / N as i32, k % N as i32), sc.load((k / N as i32, k % N as i32)));
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, b, c) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<F16>());
        for i in 0..M {
            for j in 0..N {
                let want: f32 = (0..K).map(|l| a[i * K + l].to_f32() * b[j * K + l].to_f32()).sum();
                assert_eq!(c[i * N + j], F16::from_f32(want), "C[{i},{j}]");
            }
        }
    }
    Case { name: "stmatrix_epilogue", kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: true, check }
}

/// The `.m16n8.trans.b8` forms (x1, x2, x4): each lane's bytes stored with
/// `StMatrix16x8x8bOp`, read back plainly and, for x2 and x4, with `LdMatrix16x8x8bOp`, its
/// load-side inverse.
pub fn stmatrix_b8_forms() -> Case {
    const NUMS: [usize; 3] = [1, 2, 4];
    const BYTES: usize = 128 * (1 + 2 + 4);
    let kernel = Kernel::builder("stmatrix_b8_forms").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<u8>("a", Layout::col_major(BYTES));
        let plain = param_tensor::<u8>("plain", Layout::col_major(BYTES));
        let back = param_tensor::<u8>("back", Layout::col_major(BYTES));
        let lane = thread_idx().x;
        let mut base = 0i32;
        for num in NUMS {
            let rows = 8 * num;
            let smem = make_smem_tensor::<u8>(Layout::col_major((16, rows)));
            let values = 4 * num as i32;
            let atom = make_copy_atom::<u8>(CopyOp::StMatrix16x8x8b { num, trans: true });
            let tiled = make_tiled_copy(&atom, atom.layout_ref_tv(), (16, rows));
            let frag = make_fragment::<u8>(4 * num);
            for v in 0..4 * num {
                frag.store(v, a.load(base + lane * values + v as i32));
            }
            copy(&tiled, &tiled.retile(&frag), &tiled.get_slice(lane).partition_d(&smem));
            sync_threads();
            for i in 0..rows * 16 / 32 {
                let k = lane + 32 * i as i32;
                plain.store(base + k, smem.load((k % 16, k / 16)));
            }
            if num > 1 {
                let latom = make_copy_atom::<u8>(CopyOp::LdMatrix16x8x8b { num, trans: true, unpack_bits: None });
                let ltiled = make_tiled_copy(&latom, latom.layout_ref_tv(), (16, rows));
                let got = make_fragment::<u8>(4 * num);
                copy(&ltiled, &ltiled.get_slice(lane).partition_s(&smem), &ltiled.retile(&got));
                for v in 0..4 * num {
                    back.store(base + lane * values + v as i32, got.load(v));
                }
            } else {
                for v in 0..4usize {
                    back.store(base + lane * values + v as i32, frag.load(v));
                }
            }
            sync_threads();
            base += 128 * num as i32;
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, plain, back) = (args[0].to_vec::<u8>(), args[1].to_vec::<u8>(), args[2].to_vec::<u8>());
        assert_eq!(back, a, "LdMatrix16x8x8bOp reads back what StMatrix16x8x8bOp stored");
        let mut base = 0;
        for num in NUMS {
            for l in 0..32 {
                for v in 0..4 * num {
                    assert_eq!(
                        plain[base + stsm_b8_offset(l, v)],
                        a[base + l * 4 * num + v],
                        "x{num} lane {l} value {v}"
                    );
                }
            }
            base += 128 * num;
        }
    }
    Case { name: "stmatrix_b8_forms", kernel, cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]), small_only: false, check }
}

/// Warp specialisation with register reallocation (the pattern of CuTe DSL's persistent
/// dense GEMM): of three warpgroups, the producer lowers its budget to 40 registers and
/// fills shared memory; the two consumers raise theirs to 232, wait on a named barrier for
/// the data, use it, and after a barrier of their own warpgroup lower their budget to 160.
pub fn setmaxnreg_warp_specialized() -> Case {
    const N: usize = 256;
    let kernel = Kernel::builder("setmaxnreg_warp_specialized").block_dim([384, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(N));
        let out = param_tensor::<i32>("out", Layout::col_major(N));
        let t = thread_idx().x;
        let wg = t / 128;
        let smem = make_smem_tensor::<i32>(Layout::col_major(N));
        if_else(
            wg.eq(0),
            || {
                setmaxregister_decrease(40);
                smem.store(t, xs.load(t) * 2);
                smem.store(t + 128, xs.load(t + 128) * 2);
                barrier_arrive(1, 384);
            },
            || {
                setmaxregister_increase(232);
                barrier(1, 384);
                let i = t - 128;
                out.store(i, smem.load(i) + 1);
                barrier(wg + 1, 128);
                setmaxregister_decrease(160);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for i in 0..N {
            assert_eq!(out[i], xs[i].wrapping_mul(2).wrapping_add(1), "out[{i}]");
        }
    }
    Case {
        name: "setmaxnreg_warp_specialized",
        kernel,
        cfg: LaunchConfig::new([2, 1, 1], [384, 1, 1]),
        small_only: false,
        check,
    }
}

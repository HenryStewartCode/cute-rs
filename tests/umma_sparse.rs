// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernels' structure (operands and metadata staged in shared memory, the metadata moved
// into TMEM with a tcgen05.cp copy, a k-block loop of MMAs, a commit to an mbarrier and a
// tcgen05.ld epilogue) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py and
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py (the pair),
// and the 2:4 metadata codes (0x4, 0x8, 0xC, 0x9, 0xD, 0xE) those of
//   examples/python/CuTeDSL/utils/sparse_utils.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The 2:4 structured-sparse tcgen05 MMAs (roadmap part m10-sparse-umma): GEMM tiles on the
//! simulator for F16, BF16, E4M3 and E5M2 with M = 128 on one CTA and M = 256 on a CTA pair.
//! A is random 2:4 sparse, stored compressed in shared memory (make_sparse_smem_layout_a);
//! the metadata is placed in global memory by make_sparse_gmem_layout_e, copied into shared
//! memory (make_sparse_smem_layout_e) and from there into TMEM by a 128x128b tcgen05.cp onto
//! make_sparse_tmem_layout_e; the simulator expands A by its own reading of the PTX metadata
//! layout. Exact against a dense plain-Rust GEMM with the zeros put back. Then the build-time
//! checks, the simulator's report of bad metadata, and the PTX (through ptxas when on PATH).

use cute::block::block_copy;
use cute::prelude::*;
use cute::tcgen05::{
    self, Cp128x128bOp, CtaGroup, Field, Ld32x32bOp, MmaF16BF16SparseOp, MmaFP8SparseOp, OperandSource, Repetition,
    SmemLayoutAtomKind, SparseMetadataFormat,
};
use cute::utils::sm100::{
    make_smem_layout_b, make_sparse_e_dtype, make_sparse_gmem_layout_e, make_sparse_smem_layout_a,
    make_sparse_smem_layout_e, make_sparse_tmem_layout_e,
};

const LANE: usize = 1 << 16;
/// K of every tile: one metadata atom (128 x 128 logical) per CTA.
const K: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    F16,
    BF16,
    E4M3,
    E5M2,
}

impl Fmt {
    fn dtype(self) -> DType {
        match self {
            Fmt::F16 => DType::F16,
            Fmt::BF16 => DType::BF16,
            Fmt::E4M3 => DType::F8E4M3FN,
            Fmt::E5M2 => DType::F8E5M2,
        }
    }

    fn bytes(self) -> usize {
        self.dtype().bits() / 8
    }

    /// The little-endian bytes of `x`.
    fn encode(self, x: f32) -> Vec<u8> {
        match self {
            Fmt::F16 => F16::from_f32(x).0.to_le_bytes().to_vec(),
            Fmt::BF16 => BF16::from_f32(x).0.to_le_bytes().to_vec(),
            Fmt::E4M3 => vec![F8E4M3FN::from_f32(x).0],
            Fmt::E5M2 => vec![F8E5M2::from_f32(x).0],
        }
    }
}

/// What a kernel gets wrong on purpose.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    /// gemm without the metadata.
    NoMetadata,
}

#[derive(Clone, Copy)]
struct Case {
    m: usize,
    n: usize,
    pair: bool,
    fmt: Fmt,
    /// Pass the metadata through set_tensor(METADATA) rather than with A.
    fields: bool,
    target: Arch,
}

fn atom_of(c: Case) -> MmaAtom {
    let (s, tid, k) = (OperandSource::SMEM, SparseMetadataFormat::Tid, OperandMajorMode::K);
    let group = if c.pair { CtaGroup::TWO } else { CtaGroup::ONE };
    match c.fmt {
        Fmt::F16 | Fmt::BF16 => {
            MmaF16BF16SparseOp::new(c.fmt.dtype(), DType::F32, (c.m, c.n), group, s, k, k, tid).into()
        }
        Fmt::E4M3 | Fmt::E5M2 => MmaFP8SparseOp::new(c.fmt.dtype(), DType::F32, (c.m, c.n), group, s, k, k, tid).into(),
    }
}

/// Copy `rows x kb` bytes (rows `row0..`, row-major in `g`) into the byte view `s` of an
/// MMA-shaped operand tile, whose flat index is `row + rows * byte` (one M-block, one stage).
fn fill(g: &Tensor<u8>, s: &Tensor<u8>, row0: Int32, rows: usize, kb: usize) {
    let t = thread_idx().x;
    for_range(0, (rows * kb / 128) as i32, 1, |j| {
        let idx = j * 128 + t;
        let (r, c) = (idx % rows as i32, idx / rows as i32);
        s.store(idx, g.load((row0 + r, c)));
    });
}

/// The tile `C = A B^T` for 2:4-sparse A, C `(m, n)` row-major Float32, by one CTA or a CTA
/// pair of four warps each.
fn sp_kernel<T: Numeric>(name: &str, c: Case, fault: Fault) -> Kernel {
    let atom = atom_of(c);
    let kb = atom.shape_mnk()[2];
    let Case { m, n, pair, fmt, fields, target } = c;
    let v = if pair { 2 } else { 1 };
    let (ma, nb) = (m / v, n / v);
    let bytes = fmt.bytes();
    let (kab, kbb) = (K / 2 * bytes, K * bytes);
    let group = if pair { CtaGroup::TWO } else { CtaGroup::ONE };
    let e_ty = make_sparse_e_dtype(fmt.dtype());
    Kernel::builder(name).block_dim([128, 1, 1]).target(target).build(move || {
        let ga = param_tensor::<u8>("a", Layout::new((m, kab), (kab, 1)));
        let gb = param_tensor::<u8>("b", Layout::new((n, kbb), (kbb, 1)));
        let ge = param_tensor::<u8>("e", Layout::col_major(v * 2048));
        let gc = param_tensor::<f32>("c", Layout::new((m, n), (n, 1)));
        let mut mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let sa_layout = make_sparse_smem_layout_a(&mma, (m, n, K), fmt.dtype(), 1, 2);
        let sb_layout = make_smem_layout_b(&mma, (m, n, K), fmt.dtype(), 1, None);
        let sa = make_smem_tensor_aligned::<T>(sa_layout, 1024).slice((.., .., .., 0));
        let sb = make_smem_tensor_aligned::<T>(sb_layout, 1024).slice((.., .., .., 0));
        // The metadata: logical one-bit elements, as Uint8 tensors.
        let se_logical = make_sparse_smem_layout_e(&mma, (m, n, K), fmt.dtype(), 1);
        let se_bytes = recast_layout_from_sparse(se_logical.layout_ref(), e_ty, 8);
        let se = make_smem_tensor_aligned::<u8>(se_bytes.clone(), 128);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let warp = make_warp_uniform(warp_idx());
        let rank = if pair { block_idx_in_cluster() } else { Int32::from(0) };
        let leader = rank.eq(0);
        let e_col = n;
        let cols = (n + 4).next_power_of_two().max(32) as i32;
        if_(warp.eq(0), || alloc_tmem(cols, &slot, pair.then_some(true), Arch::Sm100a));
        fill(&ga, &sa.recast::<u8>(), rank * ma as i32, ma, kab);
        fill(&gb, &sb.recast::<u8>(), rank * nb as i32, nb, kbb);
        // This CTA's metadata atom: 2048 bytes, the same in shared memory (identity swizzle).
        let raw_e = make_tensor(se.ptr(), Layout::col_major(2048));
        for_range(0, 16, 1, |j| raw_e.store(j * 128 + t, ge.load(rank * 2048 + j * 128 + t)));
        fence_view_async_shared();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        if pair {
            cluster_arrive();
            cluster_wait();
        } else {
            sync_threads();
        }
        let tmem = retrieve_tmem_ptr::<f32>(16, &slot);
        let tmem8 = retrieve_tmem_ptr::<u8>(16, &slot);
        let te_logical = make_sparse_tmem_layout_e(((128, kb), 1, K / kb), fmt.dtype());
        let te_bytes = recast_layout_from_sparse(&te_logical, e_ty, 8);
        let te = make_tensor(tmem8.clone() + (4 * e_col) as i32, te_bytes);
        let acc =
            make_tensor(tmem.clone(), mma.make_fragment_c::<f32>(&mma.partition_shape_c((m, n))).layout().clone());
        let fa = mma.make_fragment_a(&sa);
        let fb = mma.make_fragment_b(&sb);
        if_(warp.eq(0) & leader, || {
            let catom = make_copy_atom::<u8>(Cp128x128bOp::new(group));
            let src = make_tensor(se.ptr(), se_bytes.clone());
            let tiled = tcgen05::make_s2t_copy(&catom, &te);
            block_copy(&tiled, &src, &te, CopyArgs::new());
            mma.set(Field::ACCUMULATE, false);
            for j in 0..K / kb {
                let (aj, bj, ej) = (fa.slice((.., .., j)), fb.slice((.., .., j)), te.slice((.., .., j)));
                match fault {
                    Fault::NoMetadata => gemm(&mma, &acc, &aj, &bj, &acc),
                    Fault::None if fields => {
                        mma.set_tensor(Field::METADATA, &ej);
                        gemm(&mma, &acc, &aj, &bj, &acc);
                    }
                    Fault::None => gemm(&mma, &acc, (&aj, &ej), &bj, &acc),
                }
                mma.set(Field::ACCUMULATE, true);
            }
            let mask = pair.then(|| Int16::from(0b11i16));
            if_(elect_one(), || tcgen05::commit(&mbar, mask, group));
        });
        mbarrier_wait(&mbar, 0);
        // This CTA's 128 rows of D, one per TMEM lane.
        let l = lane_idx();
        let row = rank * 128 + (warp % 4) * 32 + l;
        let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
        for chunk in 0..n / 32 {
            let layout = Layout::from_modes([Layout::new((32, 32), (1, LANE))]);
            let view = make_tensor(tmem.clone() + (warp % 4) * (32 * LANE) as i32 + (32 * chunk) as i32, layout);
            let frag = make_fragment::<f32>(32);
            copy(&ld, &view, &frag);
            fence_view_async_tmem_load();
            for e in 0..32 {
                gc.store((row, (32 * chunk + e) as i32), frag.load(e));
            }
        }
        fence_view_async_tmem_load();
        if pair {
            cluster_arrive();
            cluster_wait();
        } else {
            sync_threads();
        }
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(pair.then_some(true));
            dealloc_tmem(&tmem, cols, pair.then_some(true), Arch::Sm100a);
        });
    })
}

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// The six 2:4 metadata codes: two increasing positions of four.
const CODES: [u32; 6] = [0x4, 0x8, 0xC, 0x9, 0xD, 0xE];

/// The host operands of a case: dense A (with its zeros), compressed A, the 4-bit codes
/// (`[row * K / 4 + group]`) and B.
struct Host {
    dense: Vec<f32>,
    compressed: Vec<f32>,
    codes: Vec<u32>,
    b: Vec<f32>,
}

fn host(c: Case, seed: u64) -> Host {
    let mut r = lcg(seed);
    let mut half = move || (r() % 9) as f32 * 0.5 - 2.0;
    let (m, n) = (c.m, c.n);
    let mut pick = lcg(seed ^ 0x5eed);
    let (mut dense, mut compressed, mut codes) = (vec![0f32; m * K], Vec::with_capacity(m * K / 2), Vec::new());
    for row in 0..m {
        for g in 0..K / 4 {
            let code = CODES[(pick() % 6) as usize];
            let (i0, i1) = ((code & 3) as usize, (code >> 2) as usize);
            let (x, y) = (half(), half());
            dense[row * K + 4 * g + i0] = x;
            dense[row * K + 4 * g + i1] = y;
            compressed.extend([x, y]);
            codes.push(code);
        }
    }
    let b = (0..n * K).map(|_| half()).collect();
    Host { dense, compressed, codes, b }
}

/// The metadata's global bytes: the bit of logical element (row, k) is bit k % 4 of the code
/// of group k / 4, at make_sparse_gmem_layout_e's offset.
fn metadata_image(c: Case, codes: &[u32]) -> Vec<u8> {
    let layout = make_sparse_gmem_layout_e((c.m, c.n, K, 1), c.fmt.dtype());
    let mut out = vec![0u8; c.m * K / 8];
    for row in 0..c.m {
        for k in 0..K {
            let bit = (codes[row * K / 4 + k / 4] >> (k % 4)) & 1;
            let off = layout.call(row + c.m * k);
            out[off / 8] |= (bit as u8) << (off % 8);
        }
    }
    out
}

fn launch(c: Case) -> LaunchConfig {
    if c.pair {
        LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1])
    } else {
        LaunchConfig::new([1, 1, 1], [128, 1, 1])
    }
}

fn args(c: Case, h: &Host, e: &[u8]) -> Vec<KernelArg> {
    let enc = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&x| c.fmt.encode(x)).collect() };
    vec![
        KernelArg::buffer(&enc(&h.compressed)),
        KernelArg::buffer(&enc(&h.b)),
        KernelArg::buffer(e),
        KernelArg::buffer(&vec![0f32; c.m * c.n]),
    ]
}

fn run<T: Numeric>(name: &str, c: Case) {
    let h = host(c, 3 + c.m as u64 + c.n as u64 + c.fmt.bytes() as u64);
    let kernel = sp_kernel::<T>(name, c, Fault::None);
    let mut a = args(c, &h, &metadata_image(c, &h.codes));
    let report = kernel.simulate(launch(c), &mut a).unwrap_or_else(|e| panic!("{name}: {e}"));
    let kb = atom_of(c).shape_mnk()[2];
    assert_eq!(report.tcgen05_mma_instructions, (K / kb) as u64, "{name}");
    let got = a[3].to_vec::<f32>();
    for r in 0..c.m {
        for col in 0..c.n {
            let want: f32 = (0..K).map(|k| h.dense[r * K + k] * h.b[col * K + k]).sum();
            assert_eq!(got[r * c.n + col], want, "{name}: C({r}, {col})");
        }
    }
}

fn case(m: usize, n: usize, fmt: Fmt) -> Case {
    Case { m, n, pair: false, fmt, fields: false, target: Arch::Sm100a }
}

#[test]
fn f16_and_bf16_tiles_match_the_dense_reference() {
    run::<F16>("sp_f16", case(128, 64, Fmt::F16));
    run::<BF16>("sp_bf16_fields", Case { fields: true, ..case(128, 128, Fmt::BF16) });
}

#[test]
fn fp8_tiles_match_the_dense_reference() {
    run::<F8E4M3FN>("sp_e4m3", case(128, 64, Fmt::E4M3));
    run::<F8E5M2>("sp_e5m2", case(128, 32, Fmt::E5M2));
}

#[test]
fn pair_tiles_match_the_dense_reference() {
    run::<F16>("sp_pair_f16", Case { pair: true, ..case(256, 64, Fmt::F16) });
    run::<F8E4M3FN>("sp_pair_e4m3", Case { pair: true, ..case(256, 64, Fmt::E4M3) });
}

#[test]
fn metadata_naming_no_two_positions_is_reported() {
    let c = case(128, 64, Fmt::F16);
    let mut h = host(c, 9);
    // Row 5, group 3: positions 3 and 0 (not increasing).
    h.codes[5 * K / 4 + 3] = 0b0011;
    let kernel = sp_kernel::<F16>("sp_bad_meta", c, Fault::None);
    let mut a = args(c, &h, &metadata_image(c, &h.codes));
    let e = kernel.simulate(launch(c), &mut a).unwrap_err().to_string();
    assert!(e.contains("row 5") && e.contains("0b0011") && e.contains("increasing"), "{e}");
}

fn panic_text(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn illegal_sparse_operations_panic_when_built() {
    use OperandMajorMode::K as KM;
    let (one, s, tid) = (CtaGroup::ONE, OperandSource::SMEM, SparseMetadataFormat::Tid);
    // A missing METADATA.
    let e = panic_text(|| {
        sp_kernel::<F16>("sp_no_meta", case(128, 64, Fmt::F16), Fault::NoMetadata);
    });
    assert!(e.contains("metadata") && e.contains("METADATA"), "{e}");
    // Loading one element of a sparse-typed tensor.
    let e = panic_text(|| {
        Kernel::builder("sp_deref").block_dim([32, 1, 1]).build(|| {
            let s = make_smem_tensor::<u8>(Layout::col_major(16));
            let sparse = recast_tensor_sparse::<u8, u8>(&s, make_sparse_e_dtype(DType::F16));
            assert_eq!(sparse.layout().size(), 128);
            sparse.load(3);
        });
    });
    assert!(e.contains("sparse_elem<8, 1, Uint8>") && e.contains("physical"), "{e}");
    // A wrong K, M = 64, A from TMEM, mixed FP8 accumulators.
    let e = panic_text(|| {
        MmaF16BF16SparseOp::new(DType::F16, DType::F32, (128, 64, 16), one, s, KM, KM, tid);
    });
    assert!(e.contains("K = 32"), "{e}");
    let e = panic_text(|| {
        MmaF16BF16SparseOp::new(DType::F16, DType::F32, (64, 64), one, s, KM, KM, tid);
    });
    assert!(e.contains("M is 64"), "{e}");
    let e = panic_text(|| {
        MmaFP8SparseOp::new(DType::F8E4M3FN, DType::F32, (128, 64), one, OperandSource::TMEM, KM, KM, tid);
    });
    assert!(e.contains("OperandSource.SMEM"), "{e}");
    let e = panic_text(|| {
        MmaFP8SparseOp::new(DType::F8E4M3FN, DType::F32, (128, 24), one, s, KM, KM, tid);
    });
    assert!(e.contains("N is 24"), "{e}");
    let e = panic_text(|| {
        MmaF16BF16SparseOp::new(DType::BF16, DType::F16, (128, 64), one, s, KM, KM, tid);
    });
    assert!(e.contains("acc_dtype"), "{e}");
}

#[test]
fn sparse_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let cases: [(Case, &str); 3] = [
        (case(128, 64, Fmt::F16), "tcgen05.mma.sp.cta_group::1.kind::f16 ["),
        (case(128, 64, Fmt::E4M3), "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 ["),
        (Case { pair: true, ..case(256, 64, Fmt::BF16) }, "tcgen05.mma.sp.cta_group::2.kind::f16 ["),
    ];
    for (i, (c, text)) in cases.into_iter().enumerate() {
        let name = format!("ptx_sp_{i}");
        let kernel = match c.fmt {
            Fmt::F16 => sp_kernel::<F16>(&name, c, Fault::None),
            Fmt::BF16 => sp_kernel::<BF16>(&name, c, Fault::None),
            _ => sp_kernel::<F8E4M3FN>(&name, c, Fault::None),
        };
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        assert!(ptx.contains(text), "{name}: no `{text}` in the PTX");
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-umma-sp-{}-{name}", std::process::id()));
        let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn sparse_layouts_unit_values() {
    // The smem metadata layout of a 128 x 128 tile is the atom, 2048 bytes as Uint8.
    let c = case(128, 64, Fmt::F16);
    let mma = make_tiled_mma(atom_of(c), Layout::col_major((1, 1, 1)));
    let se = make_sparse_smem_layout_e(&mma, (128, 64, K), DType::F16, 1);
    let bytes = recast_layout_from_sparse(se.layout_ref(), make_sparse_e_dtype(DType::F16), 8);
    assert_eq!((bytes.size(), bytes.cosize()), (2048, 2048));
    // Compressed A: K / 2 per tile, SW128 for 64 Float16 values per row.
    let sa = make_sparse_smem_layout_a(&mma, (128, 64, K), DType::F16, 2, 2);
    assert_eq!(sa.layout_ref().shape().to_string(), "((128,16),1,4,(1,2))");
    let _ = SmemLayoutAtomKind::K_SW128;
}

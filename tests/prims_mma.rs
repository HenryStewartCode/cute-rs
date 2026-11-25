// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The `slicek_*` tests port the six variants of NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/experimental/primitives/tcgen05/slicek_m32n8k128.py
// (its operand layouts, descriptors, MMA sequences, write-disable masks and epilogue
// reductions),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! Roadmap part m11-prims-mma-forms: the MMA forms only `cute::prims` issues, on the
//! simulator against plain Rust (exact on small integers): weight-stationary
//! `tcgen05.mma.ws` for each kind and M (Layouts G, E, D), the B collector sequence fill ->
//! use -> use -> lastuse with B overwritten after the fill, the same for the plain MMA's A
//! collector, `.ashift` row by row, scale-input-d 0, 1 and 15 on f16 and tf32, `.ws.sp` with
//! random 2:4 patterns, `tcgen05.mma.sp.block_scale` (E8M0 and UE4M3 scales), the SM120
//! warp-level `mma.sp.block_scale`, the six SliceK variants, the diagnostics, the targets,
//! and the MLIR/PTX (assembled with `ptxas` when one is on PATH; none runs on hardware).

use cute::prelude::*;
use cute::prims::*;
use cute::tcgen05::{self, CtaGroup, Ld32x32bOp, Repetition, St32x32bOp};

const LANE: i32 = 1 << 16;
/// TMEM columns allocated.
const COLS: usize = 512;
/// Where staged TMEM words go (A, metadata, scale factors).
const TIN: usize = 256;

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// `n` small integers from -3 to 3.
fn ints(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| (r() % 7) as f32 - 3.0).collect()
}

/// Values from -2 to 2 in steps of 0.5 (exact in E2M1).
fn halves(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| (r() % 9) as f32 * 0.5 - 2.0).collect()
}

/// The byte of element (`row`, `kbyte`) of a K-major operand in the 128-byte swizzle
/// (8-row atoms of 1024 bytes, rows of 128 bytes).
fn sw128(row: usize, kbyte: usize) -> usize {
    (row / 8) * 1024 + (row % 8) * 128 + (((kbyte / 16) ^ (row % 8)) * 16) + kbyte % 16
}

/// The 32-byte swizzle (8-row atoms of 256 bytes, rows of 32 bytes).
fn sw32(row: usize, kbyte: usize) -> usize {
    let off = (row / 8) * 256 + (row % 8) * 32 + kbyte;
    off ^ (((off >> 7) & 1) << 4)
}

fn sw128_desc(p: Ptr<u8>) -> Tcgen05SmemDesc {
    let f = Tcgen05SmemDescFields {
        leading_byte_offset: 16,
        stride_byte_offset: 1024,
        swizzle: Tcgen05SmemSwizzle::Swizzle128B,
        ..Default::default()
    };
    Tcgen05SmemDesc::build(&p, &f).unwrap()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ty {
    F16,
    Tf32,
    I8,
    E4M3,
    E2M1,
}

impl Ty {
    fn bytes(self) -> usize {
        match self {
            Ty::F16 => 2,
            Ty::Tf32 => 4,
            _ => 1,
        }
    }
    fn code(self, x: f32) -> u32 {
        match self {
            Ty::F16 => F16::from_f32(x).0 as u32,
            Ty::Tf32 => x.to_bits(),
            Ty::I8 => x as i8 as u8 as u32,
            Ty::E4M3 => F8E4M3FN::from_f32(x).0 as u32,
            Ty::E2M1 => F4E2M1FN::from_f32(x).0 as u32,
        }
    }
    fn dtype(self) -> DType {
        match self {
            Ty::F16 => DType::F16,
            Ty::Tf32 => DType::TF32,
            Ty::I8 => DType::I8,
            Ty::E4M3 => DType::F8E4M3FN,
            Ty::E2M1 => DType::F4E2M1FN,
        }
    }
    fn kind(self) -> Tcgen05MmaKind {
        match self {
            Ty::F16 => Tcgen05MmaKind::F16,
            Ty::Tf32 => Tcgen05MmaKind::TF32,
            Ty::I8 => Tcgen05MmaKind::I8,
            _ => Tcgen05MmaKind::F8F6F4,
        }
    }
}

/// Write the row-major `rows x k` values `v` of type `ty` into `img` at `base` in the
/// 128-byte swizzle (FP4 packed two per byte).
fn put_sw128(img: &mut [u8], base: usize, ty: Ty, v: &[f32], rows: usize, k: usize) {
    for r in 0..rows {
        for c in 0..k {
            let code = ty.code(v[r * k + c]);
            if ty == Ty::E2M1 {
                img[base + sw128(r, c / 2)] |= (code as u8) << (4 * (c % 2));
            } else {
                for b in 0..ty.bytes() {
                    img[base + sw128(r, c * ty.bytes() + b)] = (code >> (8 * b)) as u8;
                }
            }
        }
    }
}

/// The dense instruction descriptor of `ty` into `acc`, M x N.
fn idesc(ty: Ty, acc: DType, m: u32, n: u32, sparse: bool) -> Tcgen05InstrDesc {
    let kind = ty.kind().data_kind();
    Tcgen05InstrDesc::build(&Tcgen05InstrDescFields {
        sparse,
        c_format: Tcgen05OutputFormat::for_dtype(acc).unwrap(),
        a_format: Tcgen05InputFormat::for_dtype(kind, ty.dtype()).unwrap(),
        b_format: Tcgen05InputFormat::for_dtype(kind, ty.dtype()).unwrap(),
        n,
        m,
        ..Default::default()
    })
    .unwrap()
}

/// What a test kernel's issuing thread sees.
struct Ctx {
    /// Shared memory (1024-aligned), as bytes and as words.
    smem: Ptr<u8>,
    words: Tensor<u32>,
    /// The patch image (words) for overwriting shared memory.
    patch: Tensor<u32>,
    /// The TMEM base address.
    tmem: Int32,
    mbar: Ptr<u64>,
}

impl Ctx {
    fn s(&self, off: usize) -> Ptr<u8> {
        self.smem.clone() + off as i32
    }
    fn t(&self, col: usize) -> Int32 {
        self.tmem + col as i32
    }
    /// Commit, and wait (this thread) on mbarrier phase `phase`.
    fn sync(&self, phase: i32) {
        tcgen05::commit(&self.mbar, None, CtaGroup::ONE);
        mbarrier_wait(&self.mbar, phase);
    }
    /// Overwrite shared bytes `from..from + n` with the patch image's (at the same offsets).
    fn overwrite(&self, from: usize, n: usize) {
        let w0 = (from / 4) as i32;
        for_range(w0, w0 + (n / 4) as i32, 1, |i| self.words.store(i, self.patch.load(i)));
        fence_view_async_shared();
    }
}

/// A one-CTA kernel of four warps: the shared image is copied into shared memory, the TMEM
/// image (`128 x tin` words) stored at column [`TIN`], then one elected thread runs `issue`
/// (which waited on `phases` mbarrier phases), commits, and every thread dumps TMEM columns
/// `0..out` (lane-major) to `out`.
fn harness(
    name: &str,
    arch: Arch,
    smem_len: usize,
    tin: usize,
    out: usize,
    phases: i32,
    issue: impl Fn(&Ctx) + 'static,
) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).target(arch).build(move || {
        let img = param_tensor::<u32>("smem", Layout::col_major(smem_len / 4));
        let patch = param_tensor::<u32>("patch", Layout::col_major(smem_len / 4));
        let tin_g = param_tensor::<u32>("tin", Layout::col_major((128 * tin).max(1)));
        let gout = param_tensor::<u32>("out", Layout::col_major(128 * out));
        let sm = make_smem_tensor_aligned::<u32>(Layout::col_major(smem_len / 4), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        for_range(0, (smem_len / 4 / 128) as i32, 1, |j| sm.store(j * 128 + t, img.load(j * 128 + t)));
        fence_view_async_shared();
        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(0), || alloc_tmem(COLS as i32, &slot, None, Arch::Sm100a));
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let w = warp % 4;
        let lane = w * 32 + lane_idx();
        let tm = retrieve_tmem_ptr::<u32>(16, &slot);
        let st = make_copy_atom::<u32>(St32x32bOp::new(Repetition::X1));
        for c in 0..tin {
            let frag = make_fragment::<u32>(1);
            frag.store(0, tin_g.load(lane * tin as i32 + c as i32));
            let view = make_tensor(
                tm.clone() + w * (32 * LANE) + (TIN + c) as i32,
                Layout::from_modes([Layout::new((1, 32), (1, LANE as usize))]),
            );
            copy(&st, &frag, &view);
        }
        if tin > 0 {
            fence_view_async_tmem_store();
        }
        sync_threads();
        let ctx = Ctx {
            smem: sm.recast::<u8>().ptr(),
            words: sm.clone(),
            patch: patch.clone(),
            tmem: tm.toint_i32(),
            mbar: mbar.clone(),
        };
        if_(warp.eq(0), || {
            if_(elect_one(), || {
                issue(&ctx);
                tcgen05::commit(&mbar, None, CtaGroup::ONE);
            })
        });
        sync_threads();
        mbarrier_wait(&mbar, phases % 2);
        let ld = make_copy_atom::<u32>(Ld32x32bOp::new(Repetition::X1));
        for c in 0..out {
            let frag = make_fragment::<u32>(1);
            let view = make_tensor(
                tm.clone() + w * (32 * LANE) + c as i32,
                Layout::from_modes([Layout::new((1, 32), (1, LANE as usize))]),
            );
            copy(&ld, &view, &frag);
            fence_view_async_tmem_load();
            gout.store(lane * out as i32 + c as i32, frag.load(0));
        }
        sync_threads();
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tm, COLS as i32, None, Arch::Sm100a)
        });
    })
}

fn words(img: &[u8]) -> Vec<u32> {
    img.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Simulate `k` with the shared image, patch and TMEM image; the dumped TMEM words.
fn run(k: &Kernel, smem: &[u8], patch: &[u8], tin: &[u32], out: usize, opts: &SimOptions) -> Result<Vec<u32>, String> {
    let mut args = vec![
        KernelArg::buffer(&words(smem)),
        KernelArg::buffer(&words(patch)),
        KernelArg::buffer(&if tin.is_empty() { vec![0u32] } else { tin.to_vec() }),
        KernelArg::buffer(&vec![0u32; 128 * out]),
    ];
    k.simulate_with(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args, opts).map_err(|e| e.to_string())?;
    Ok(args[3].to_vec::<u32>())
}

/// `C[r][c] = sum_k A[r][k] B[c][k]`, in f64 (exact for these tests).
fn gemm_ref(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for r in 0..m {
        for j in 0..n {
            c[r * n + j] = (0..k).map(|x| a[r * k + x] as f64 * b[j * k + x] as f64).sum();
        }
    }
    c
}

/// D element (`row`, `col`) of an accumulator of `acc` type at TMEM column `col0`, placed by
/// `place` (lane, column).
fn d_at(out: &[u32], ncols: usize, place: (usize, usize), col0: usize, acc: DType) -> f64 {
    let bits = out[place.0 * ncols + col0 + place.1];
    match acc {
        DType::I32 => bits as i32 as f64,
        DType::F16 => F16(bits as u16).to_f32() as f64,
        _ => f32::from_bits(bits) as f64,
    }
}

/// The `.ws` data-path layouts (G, E, D): (lane, column) of D element (`row`, `col`).
fn ws_place(m: usize, n: usize, row: usize, col: usize) -> (usize, usize) {
    let w = n * m / 128;
    (row + m * (col / w), col % w)
}

fn check(name: &str, got: impl Fn(usize, usize) -> f64, want: &[f64], m: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            assert_eq!(got(r, c), want[r * n + c], "{name}: D({r}, {c})");
        }
    }
}

// ---------------------------------------------------------------------------------------
// tcgen05.mma.ws
// ---------------------------------------------------------------------------------------

const B_OFF: usize = 16384;

#[test]
fn ws_tiles_for_each_kind_and_m_match_plain_rust() {
    let cases = [
        (Ty::F16, DType::F32, 32, 64, 2),
        (Ty::F16, DType::F16, 64, 128, 1),
        (Ty::Tf32, DType::F32, 64, 64, 2),
        (Ty::I8, DType::I32, 128, 64, 1),
        (Ty::E4M3, DType::F32, 32, 128, 2),
        (Ty::E4M3, DType::F32, 128, 256, 1),
    ];
    for (ty, acc, m, n, kblocks) in cases {
        let name = format!("ws_{ty:?}_{m}x{n}");
        let kb = 32 / ty.bytes();
        let k = kb * kblocks;
        let (a, b) = (ints(m * k, m as u64), ints(n * k, 7 + n as u64));
        let mut img = vec![0u8; B_OFF + n * 128];
        put_sw128(&mut img, 0, ty, &a, m, k);
        put_sw128(&mut img, B_OFF, ty, &b, n, k);
        let id = idesc(ty, acc, m as u32, n as u32, false);
        let out = n * m / 128;
        let kernel = harness(&name, Arch::Sm100a, img.len(), 0, out, 0, move |c| {
            for j in 0..kblocks {
                tcgen05_mma_ws(
                    ty.kind(),
                    c.t(0),
                    sw128_desc(c.s(32 * j)),
                    sw128_desc(c.s(B_OFF + 32 * j)),
                    id,
                    j > 0,
                    Tcgen05MmaWsOptions::default(),
                );
            }
        });
        let got = run(&kernel, &img, &img, &[], out, &SimOptions::default()).unwrap();
        let want = gemm_ref(&a, &b, m, n, k);
        check(&name, |r, j| d_at(&got, out, ws_place(m, n, r, j), 0, acc), &want, m, n);
    }
}

/// The B collector sequence: four A tiles (M = 32) times one B, FILL (b2) -> USE -> USE ->
/// LASTUSE, B overwritten in shared memory after the FILL completed.
fn ws_collector_kernel(name: &str, first: Tcgen05MmaCollectorOp) -> Kernel {
    let (m, n) = (32, 64);
    let id = idesc(Ty::F16, DType::F32, m, n, false);
    harness(name, Arch::Sm100a, B_OFF + 8192, 0, 64, 1, move |c| {
        let ops = [first, Tcgen05MmaCollectorOp::USE, Tcgen05MmaCollectorOp::USE, Tcgen05MmaCollectorOp::LASTUSE];
        for (i, op) in ops.into_iter().enumerate() {
            let opts = Tcgen05MmaWsOptions {
                collector_b_buffer: Some(Tcgen05MmaCollectorBBuffer::B2),
                collector_op: Some(op),
                col_b_zero_mask: Some(Value::constant(0u64)),
            };
            tcgen05_mma_ws(
                Tcgen05MmaKind::F16,
                c.t(16 * i),
                sw128_desc(c.s(4096 * i)),
                sw128_desc(c.s(B_OFF)),
                id,
                false,
                opts,
            );
            if i == 0 {
                c.sync(0);
                c.overwrite(B_OFF, 8192);
            }
        }
    })
}

#[test]
fn ws_b_collector_reuse_reads_the_filled_tile() {
    let (m, n, k) = (32, 64, 16);
    let a: Vec<Vec<f32>> = (0..4).map(|i| ints(m * k, 40 + i)).collect();
    let b = ints(n * k, 50);
    let mut img = vec![0u8; B_OFF + 8192];
    for (i, ai) in a.iter().enumerate() {
        put_sw128(&mut img, 4096 * i, Ty::F16, ai, m, k);
    }
    put_sw128(&mut img, B_OFF, Ty::F16, &b, n, k);
    let mut patch = vec![0u8; B_OFF + 8192];
    put_sw128(&mut patch, B_OFF, Ty::F16, &ints(n * k, 99), n, k);
    let kernel = ws_collector_kernel("ws_collector", Tcgen05MmaCollectorOp::FILL);
    // The collector holds the tile filled from B, whatever shared memory holds since.
    let opts = SimOptions { collector_source_check: false, ..SimOptions::default() };
    let got = run(&kernel, &img, &patch, &[], 64, &opts).unwrap();
    for (i, ai) in a.iter().enumerate() {
        let want = gemm_ref(ai, &b, m, n, k);
        check(
            &format!("ws collector MMA {i}"),
            |r, j| d_at(&got, 64, ws_place(m, n, r, j), 16 * i, DType::F32),
            &want,
            m,
            n,
        );
    }
    // By default changing a collected operand is reported (the hardware may reload it).
    let e = run(&kernel, &img, &patch, &[], 64, &SimOptions::default()).unwrap_err();
    assert!(e.contains("collector::b2::use") && e.contains("changed since"), "{e}");
    // A USE with nothing filled.
    let kernel = ws_collector_kernel("ws_collector_unfilled", Tcgen05MmaCollectorOp::USE);
    let e = run(&kernel, &img, &patch, &[], 64, &opts).unwrap_err();
    assert!(e.contains("collector::b2::use") && e.contains("no earlier MMA"), "{e}");
}

#[test]
fn a_collector_reuse_reads_the_filled_tile() {
    // A (M = 128) times four B tiles (N = 32): FILL -> USE -> USE -> LASTUSE, A overwritten.
    let (m, n, k) = (128, 32, 16);
    let a = ints(m * k, 60);
    let b: Vec<Vec<f32>> = (0..4).map(|i| ints(n * k, 61 + i)).collect();
    let mut img = vec![0u8; B_OFF + 4 * 4096];
    put_sw128(&mut img, 0, Ty::F16, &a, m, k);
    for (i, bi) in b.iter().enumerate() {
        put_sw128(&mut img, B_OFF + 4096 * i, Ty::F16, bi, n, k);
    }
    let mut patch = vec![0u8; img.len()];
    put_sw128(&mut patch, 0, Ty::F16, &ints(m * k, 98), m, k);
    let id = idesc(Ty::F16, DType::F32, m as u32, n as u32, false);
    let kernel = harness("a_collector", Arch::Sm100a, img.len(), 0, 128, 1, move |c| {
        use Tcgen05MmaCollectorOp::*;
        for (i, op) in [FILL, USE, USE, LASTUSE].into_iter().enumerate() {
            let opts = Tcgen05MmaOptions { collector_op: Some(op), ..Default::default() };
            tcgen05_mma(
                Tcgen05MmaKind::F16,
                CtaGroup::ONE,
                c.t(32 * i),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(B_OFF + 4096 * i)),
                id,
                false,
                opts,
            );
            if i == 0 {
                c.sync(0);
                c.overwrite(0, 16384);
            }
        }
    });
    let opts = SimOptions { collector_source_check: false, ..SimOptions::default() };
    let got = run(&kernel, &img, &patch, &[], 128, &opts).unwrap();
    for (i, bi) in b.iter().enumerate() {
        let want = gemm_ref(&a, bi, m, n, k);
        check(&format!("A collector MMA {i}"), |r, j| d_at(&got, 128, (r, j), 32 * i, DType::F32), &want, m, n);
    }
    let e = run(&kernel, &img, &patch, &[], 128, &SimOptions::default()).unwrap_err();
    assert!(e.contains("collector::a::use") && e.contains("changed since"), "{e}");
}

/// Pack row-major codes of `bits` bits into TMEM words, `per_row` words a row (lane).
fn pack_rows(codes: &[u32], rows: usize, bits: usize, per_row: usize) -> Vec<Vec<u32>> {
    let k = codes.len() / rows;
    (0..rows)
        .map(|r| {
            let mut w = vec![0u32; per_row];
            for c in 0..k {
                w[c * bits / 32] |= (codes[r * k + c] & ((1u64 << bits) - 1) as u32) << ((c * bits) % 32);
            }
            w
        })
        .collect()
}

/// A TMEM image of `tin` columns with `rows[lane]` at column offset `col`.
fn tmem_image(tin: usize, parts: &[(usize, &Vec<Vec<u32>>)]) -> Vec<u32> {
    let mut img = vec![0u32; 128 * tin];
    for (col, rows) in parts {
        for (lane, w) in rows.iter().enumerate() {
            for (i, &x) in w.iter().enumerate() {
                img[lane * tin + col + i] |= x;
            }
        }
    }
    img
}

#[test]
fn ashift_shifts_a_up_one_row_per_sub_partition() {
    let (m, n, k) = (128, 32, 16);
    let a = ints(m * k, 70);
    let b = ints(n * k, 71);
    let codes: Vec<u32> = a.iter().map(|&x| Ty::F16.code(x)).collect();
    let rows = pack_rows(&codes, m, 16, 8);
    let tin = tmem_image(8, &[(0, &rows)]);
    let mut img = vec![0u8; B_OFF + 4096];
    put_sw128(&mut img, B_OFF, Ty::F16, &b, n, k);
    let id = idesc(Ty::F16, DType::F32, m as u32, n as u32, false);
    let kernel = harness("ashift", Arch::Sm100a, img.len(), 8, 64, 0, move |c| {
        let a_tmem = c.t(TIN);
        let shift = Tcgen05MmaOptions { a_shift: true, ..Default::default() };
        tcgen05_mma(Tcgen05MmaKind::F16, CtaGroup::ONE, c.t(0), a_tmem, sw128_desc(c.s(B_OFF)), id, false, shift);
        tcgen05_mma(
            Tcgen05MmaKind::F16,
            CtaGroup::ONE,
            c.t(32),
            a_tmem,
            sw128_desc(c.s(B_OFF)),
            id,
            false,
            Default::default(),
        );
    });
    let got = run(&kernel, &img, &img, &tin, 64, &SimOptions::default()).unwrap();
    let want = gemm_ref(&a, &b, m, n, k);
    // The ashift MMA reads A unshifted; the next reads row r + 1 in row r, except the last
    // row of each 32-lane sub-partition.
    check("ashift MMA", |r, j| d_at(&got, 64, (r, j), 0, DType::F32), &want, m, n);
    for r in 0..m {
        let src = if r % 32 == 31 { r } else { r + 1 };
        for j in 0..n {
            assert_eq!(d_at(&got, 64, (r, j), 32, DType::F32), want[src * n + j], "after ashift: D({r}, {j})");
        }
    }
}

#[test]
fn scale_input_d_scales_the_accumulator() {
    for ty in [Ty::F16, Ty::Tf32] {
        for s in [0u8, 1, 15] {
            let name = format!("scale_d_{ty:?}_{s}");
            let (m, n) = (128, 32);
            let kb = 32 / ty.bytes();
            let k = 2 * kb;
            let (a, b) = (ints(m * k, 80 + s as u64), ints(n * k, 81));
            let mut img = vec![0u8; B_OFF + 4096];
            put_sw128(&mut img, 0, ty, &a, m, k);
            put_sw128(&mut img, B_OFF, ty, &b, n, k);
            let id = idesc(ty, DType::F32, m as u32, n as u32, false);
            let kernel = harness(&name, Arch::Sm100a, img.len(), 0, 32, 0, move |c| {
                for j in 0..2 {
                    let opts = Tcgen05MmaOptions { scale_input_d: (j == 1).then_some(s), ..Default::default() };
                    tcgen05_mma(
                        ty.kind(),
                        CtaGroup::ONE,
                        c.t(0),
                        sw128_desc(c.s(32 * j)),
                        sw128_desc(c.s(B_OFF + 32 * j)),
                        id,
                        j == 1,
                        opts,
                    );
                }
            });
            let got = run(&kernel, &img, &img, &[], 32, &SimOptions::default()).unwrap();
            for r in 0..m {
                for j in 0..n {
                    let blk = |q: usize| -> f64 {
                        (q * kb..(q + 1) * kb).map(|x| (a[r * k + x] * b[j * k + x]) as f64).sum()
                    };
                    let want = blk(1) + blk(0) * 2f64.powi(-(s as i32));
                    assert_eq!(d_at(&got, 32, (r, j), 0, DType::F32), want, "{name}: D({r}, {j})");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Sparse forms.
// ---------------------------------------------------------------------------------------

/// A random 2:4 (`span` 4) or pairwise 4:8 (`span` 8) sparse `rows x k` matrix: the dense
/// values, the compressed values and each row's 4-bit fields.
fn sparse(
    rows: usize,
    k: usize,
    span: usize,
    seed: u64,
    vals: fn(usize, u64) -> Vec<f32>,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<u32>>) {
    let mut r = lcg(seed);
    let kept = vals(rows * k / 2, seed + 1);
    let mut dense = vec![0f32; rows * k];
    let mut fields = vec![Vec::new(); rows];
    let unit = span / 4;
    for row in 0..rows {
        for g in 0..k / span {
            let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
            let (i0, i1) = pairs[(r() % 6) as usize];
            fields[row].push(i0 as u32 | (i1 as u32) << 2);
            for (q, i) in [i0, i1].into_iter().enumerate() {
                for e in 0..unit {
                    dense[row * k + g * span + i * unit + e] = kept[row * k / 2 + g * span / 2 + q * unit + e];
                }
            }
        }
    }
    (dense, kept, fields)
}

/// The metadata words (`cols` columns) of rows `0..rows` placed as the simulator models the
/// PTX ISA's layouts: 16-bit kinds (`eight` false) rows `16 q + 8 h + r` in lane
/// `16 q + 8 (g / 4) + r`, bit `4 (g % 4) + 16 h`; 8-bit and FP4 kinds row `i` in lane `i`,
/// field `g` at bit `4 (g % 8)` of column `g / 8`; each `128 / rows`-lane replica the same.
fn meta_words(fields: &[Vec<u32>], eight: bool) -> Vec<Vec<u32>> {
    let rows = fields.len();
    let mut lanes = vec![vec![0u32; 2]; 128];
    for (row, f) in fields.iter().enumerate() {
        for (g, &x) in f.iter().enumerate() {
            let (lane, col, bit) = if eight {
                (row, g / 8, 4 * (g % 8))
            } else {
                (16 * (row / 16) + 8 * (g / 4) + row % 8, 0, 4 * (g % 4) + 16 * (row / 8 % 2))
            };
            for rep in 0..128 / rows {
                lanes[lane + rep * rows][col] |= x << bit;
            }
        }
    }
    lanes
}

#[test]
fn ws_sp_tiles_with_random_2_4_patterns_match_plain_rust() {
    for (ty, acc, m, n) in
        [(Ty::F16, DType::F32, 128, 64), (Ty::I8, DType::I32, 32, 64), (Ty::E4M3, DType::F32, 64, 128)]
    {
        let name = format!("ws_sp_{ty:?}_{m}x{n}");
        let k = 2 * 32 / ty.bytes();
        let (dense, kept, fields) = sparse(m, k, 4, 90 + m as u64, ints);
        let b = ints(n * k, 91);
        let mut img = vec![0u8; B_OFF + n * 128];
        put_sw128(&mut img, 0, ty, &kept, m, k / 2);
        put_sw128(&mut img, B_OFF, ty, &b, n, k);
        let tin = tmem_image(2, &[(0, &meta_words(&fields, ty != Ty::F16))]);
        let id = idesc(ty, acc, m as u32, n as u32, true);
        let out = n * m / 128;
        let kernel = harness(&name, Arch::Sm100a, img.len(), 2, out, 0, move |c| {
            tcgen05_mma_ws_sp(
                ty.kind(),
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(B_OFF)),
                id,
                false,
                c.t(TIN),
                Default::default(),
            );
        });
        let got = run(&kernel, &img, &img, &tin, out, &SimOptions::default()).unwrap();
        let want = gemm_ref(&dense, &b, m, n, k);
        check(&name, |r, j| d_at(&got, out, ws_place(m, n, r, j), 0, acc), &want, m, n);
    }
}

/// Scale factors of 0.5, 1, 2 or 4 and their codes (UE8M0 or UE4M3).
fn scales(n: usize, seed: u64, ue4m3: bool) -> Vec<(f64, u32)> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            let e = (r() % 4) as i32 - 1;
            (2f64.powi(e), if ue4m3 { (0x38 + 8 * e) as u32 } else { (127 + e) as u32 })
        })
        .collect()
}

/// TMEM words of `rows x nsf` scale factors (scale-factor ID 0): row `32 c + l` in lane `l` of
/// each sub-partition, column `c` (of four), factor `j` in byte `j`.
fn sf_words(codes: &[(f64, u32)], rows: usize, nsf: usize) -> Vec<Vec<u32>> {
    let mut lanes = vec![vec![0u32; 4]; 128];
    for row in 0..rows {
        for j in 0..nsf {
            for q in 0..4 {
                lanes[row % 32 + 32 * q][row / 32] |= codes[row * nsf + j].1 << (8 * j);
            }
        }
    }
    lanes
}

#[test]
fn sparse_block_scaled_tiles_match_plain_rust() {
    // (kind, A/B type, UE4M3 scales, K, scale vector count)
    let cases =
        [(Tcgen05MmaKind::MXF8F6F4, Ty::E4M3, false, 64, 1), (Tcgen05MmaKind::MXF4NVF4, Ty::E2M1, true, 128, 4)];
    for (kind, ty, ue4m3, k, nsf) in cases {
        let name = format!("sp_block_scale_{kind:?}");
        let (m, n) = (128, 64);
        let fp4 = ty == Ty::E2M1;
        let (dense, kept, fields) = sparse(m, k, if fp4 { 8 } else { 4 }, 100, halves);
        let b = halves(n * k, 101);
        let (sa, sb) = (scales(m * nsf, 102, ue4m3), scales(n * nsf, 103, ue4m3));
        let mut img = vec![0u8; B_OFF + n * 128];
        put_sw128(&mut img, 0, ty, &kept, m, k / 2);
        put_sw128(&mut img, B_OFF, ty, &b, n, k);
        let tin = tmem_image(
            10,
            &[(0, &meta_words(&fields, true)), (2, &sf_words(&sa, m, nsf)), (6, &sf_words(&sb, n, nsf))],
        );
        let sf = if ue4m3 { MxScaleFormat::UE4M3 } else { MxScaleFormat::E8M0 };
        let bits = if fp4 {
            Tcgen05MxOmmaInstrDesc::build(&Tcgen05MxOmmaInstrDescFields {
                sparse: true,
                a_format: Tcgen05InputFormat::MX_E2M1,
                b_format: Tcgen05InputFormat::MX_E2M1,
                scale_format: sf,
                n: n as u32,
                m: m as u32,
                ..Default::default()
            })
            .unwrap()
            .bits()
        } else {
            Tcgen05MxInstrDesc::build(&Tcgen05MxInstrDescFields {
                sparse: true,
                scale_format: sf,
                n: n as u32,
                m: m as u32,
                ..Default::default()
            })
            .unwrap()
            .bits()
        };
        let id = bits.as_static().unwrap();
        let kernel = harness(&name, Arch::Sm100a, img.len(), 10, 64, 0, move |c| {
            tcgen05_mma_sp_block_scale(
                kind,
                CtaGroup::ONE,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(B_OFF)),
                id,
                false,
                c.t(TIN),
                c.t(TIN + 2),
                c.t(TIN + 6),
                Default::default(),
            );
        });
        let got = run(&kernel, &img, &img, &tin, 64, &SimOptions::default()).unwrap();
        let vec = k / nsf;
        for r in 0..m {
            for j in 0..n {
                let want = (0..k).fold(0f32, |acc, x| {
                    acc + (dense[r * k + x] as f64
                        * sa[r * nsf + x / vec].0
                        * b[j * k + x] as f64
                        * sb[j * nsf + x / vec].0) as f32
                });
                assert_eq!(d_at(&got, 64, (r, j), 0, DType::F32), want as f64, "{name}: D({r}, {j})");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// SM120 warp-level mma.sp.block_scale.
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct WarpCase {
    kind: MmaBlockScaleKind,
    fp4: bool,
    sv: Tcgen05MmaScaleVec,
    ue4m3: bool,
    ids: (u8, u8, u8, u8),
}

fn warp_kernel(name: &str, c: WarpCase, arch: Arch) -> Kernel {
    Kernel::builder(name).block_dim([32, 1, 1]).target(arch).build(move || {
        let regs = param_tensor::<u32>("regs", Layout::col_major(32 * 11));
        let out = param_tensor::<f32>("out", Layout::col_major(32 * 4));
        let l = lane_idx();
        let r = |i: i32| regs.load(l * 11 + i);
        let (k, dt) = if c.fp4 { (128, DType::F4E2M1FN) } else { (64, DType::F8E4M3FN) };
        let sf = if c.ue4m3 { DType::F8E4M3FN } else { DType::F8E8M0FNU };
        let zero = Value::constant(0f32);
        let d = mma_sp_block_scale(
            (16, 8, k),
            c.sv,
            sf,
            c.kind,
            (dt, dt),
            [r(0), r(1), r(2), r(3)],
            [r(4), r(5), r(6), r(7)],
            [zero, zero, zero, zero],
            r(8),
            0,
            ScaleOperand { reg: r(9), byte_id: c.ids.0, thread_id: c.ids.1 },
            ScaleOperand { reg: r(10), byte_id: c.ids.2, thread_id: c.ids.3 },
        );
        for (i, v) in d.into_iter().enumerate() {
            out.store(l * 4 + i as i32, v);
        }
    })
}

#[test]
fn warp_sparse_block_scaled_mmas_match_plain_rust() {
    use MmaBlockScaleKind::*;
    let cases = [
        WarpCase { kind: MXF8F6F4, fp4: false, sv: Tcgen05MmaScaleVec::X1, ue4m3: false, ids: (3, 1, 1, 2) },
        WarpCase { kind: MXF4, fp4: true, sv: Tcgen05MmaScaleVec::X2, ue4m3: false, ids: (2, 1, 0, 3) },
        WarpCase { kind: MXF4NVF4, fp4: true, sv: Tcgen05MmaScaleVec::X4, ue4m3: true, ids: (0, 0, 0, 0) },
    ];
    for c in cases {
        let name = format!("warp_sp_block_scale_{:?}", c.kind);
        let (m, n, k) = (16, 8, if c.fp4 { 128 } else { 64 });
        let ty = if c.fp4 { Ty::E2M1 } else { Ty::E4M3 };
        let nsf = match c.sv {
            Tcgen05MmaScaleVec::X1 => 1,
            Tcgen05MmaScaleVec::X2 => 2,
            Tcgen05MmaScaleVec::X4 => 4,
        };
        let span = if c.fp4 { 8 } else { 4 };
        let (dense, kept, fields) = sparse(m, k, span, 110, halves);
        let b = halves(n * k, 111);
        let (sa, sb) = (scales(m * nsf, 112, c.ue4m3), scales(n * nsf, 113, c.ue4m3));
        let mut regs = vec![0u32; 32 * 11];
        let (per, bits) = if c.fp4 { (8, 4) } else { (4, 8) };
        let kh = k / 2;
        for lane in 0..32 {
            let (g, t) = (lane / 4, lane % 4);
            for rr in 0..4 {
                for j in 0..per {
                    let (row, col) = if c.fp4 {
                        (g + 8 * (rr & 1), 8 * t + j + 32 * (rr >> 1))
                    } else {
                        (g + 8 * (rr & 1), 4 * t + j + 16 * (rr >> 1))
                    };
                    regs[lane * 11 + rr] |= ty.code(kept[row * kh + col]) << (bits * j);
                    let kk = if c.fp4 { 8 * t + j + 32 * rr } else { 4 * t + j + 16 * rr };
                    regs[lane * 11 + 4 + rr] |= ty.code(b[g * k + kk]) << (bits * j);
                }
            }
        }
        for row in 0..m {
            for (ch, &f) in fields[row].iter().enumerate() {
                let (lane, nib) = (4 * (row % 8) + ch / 4, 4 * (row / 8) + ch % 4);
                regs[lane * 11 + 8] |= f << (4 * nib);
            }
            let lane = 4 * (row % 8) + 2 * c.ids.1 as usize + row / 8;
            for j in 0..nsf {
                regs[lane * 11 + 9] |= sa[row * nsf + j].1 << (8 * (c.ids.0 as usize + j));
            }
        }
        for col in 0..n {
            for j in 0..nsf {
                regs[(4 * col + c.ids.3 as usize) * 11 + 10] |= sb[col * nsf + j].1 << (8 * (c.ids.2 as usize + j));
            }
        }
        let kernel = warp_kernel(&name, c, Arch::Sm120a);
        let mut args = vec![KernelArg::buffer(&regs), KernelArg::buffer(&vec![0f32; 128])];
        kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
        let got = args[1].to_vec::<f32>();
        let vec = k / nsf;
        for lane in 0..32 {
            for i in 0..4 {
                let (r, col) = (lane / 4 + 8 * (i >> 1), 2 * (lane % 4) + (i & 1));
                let want = (0..k).fold(0f32, |acc, x| {
                    acc + (dense[r * k + x] as f64
                        * sa[r * nsf + x / vec].0
                        * b[col * k + x] as f64
                        * sb[col * nsf + x / vec].0) as f32
                });
                assert_eq!(got[lane * 4 + i], want, "{name}: D({r}, {col})");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// SliceK M32/N8/K128 (the six variants of slicek_m32n8k128.py).
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Variant {
    AsyncS32Smem,
    ATmemCoreMatrix,
    TmaS128MmaS32,
    TmaS32MmaS32,
    NGroupATmemShiftWdm,
    NGroupSmemAReuseWdm,
}

const LM: usize = 32;
const LN: usize = 8;
const LK: usize = 128;

fn slicek_kernel(v: Variant) -> Kernel {
    use Variant::*;
    let n_group = matches!(v, NGroupATmemShiftWdm | NGroupSmemAReuseWdm);
    let a_tmem = matches!(v, ATmemCoreMatrix | NGroupATmemShiftWdm);
    let n = if n_group { LN } else { 4 * LN };
    let id = idesc(Ty::E4M3, DType::F32, 128, n as u32, false);
    let smem_len = 8192;
    // A's TMEM words are staged at column TIN (the example's A_TMEM_COL_OFF is 32).
    harness(&format!("slicek_{v:?}"), Arch::Sm100a, smem_len, if a_tmem { 8 } else { 0 }, n, 0, move |c| {
        // B (8 x 128 bytes, 128-byte swizzle) at 4096; A (128 physical rows x 32 bytes,
        // 32-byte swizzle) at 0.
        let fields = |sbo: u32, sw: Tcgen05SmemSwizzle| Tcgen05SmemDescFields {
            leading_byte_offset: 16,
            stride_byte_offset: sbo,
            swizzle: sw,
            ..Default::default()
        };
        let desc_a = Tcgen05SmemDesc::build(&c.s(0), &fields(256, Tcgen05SmemSwizzle::Swizzle32B)).unwrap();
        let desc_b_expanded = Tcgen05SmemDesc::build(&c.s(4096), &fields(32, Tcgen05SmemSwizzle::Swizzle128B)).unwrap();
        let desc_b_n_group =
            Tcgen05SmemDesc::build(&c.s(4096), &fields(1024, Tcgen05SmemSwizzle::Swizzle128B)).unwrap();
        let wdm = |slot: usize| -> Vec<Int32> {
            let mask = (0..32).filter(|l| l & 3 != slot).fold(0u32, |m, l| m | 1 << l) as i32;
            vec![Value::constant(mask); 4]
        };
        let kind = Tcgen05MmaKind::F8F6F4;
        if n_group {
            for group in 0..4 {
                let desc_b = desc_b_n_group.advance_start_address(32 * group as u64).unwrap();
                let opts = if v == NGroupATmemShiftWdm {
                    Tcgen05MmaOptions { a_shift: group != 3, write_disable_mask: Some(wdm(0)), ..Default::default() }
                } else {
                    use Tcgen05MmaCollectorOp::*;
                    let op = [FILL, USE, USE, LASTUSE][group];
                    Tcgen05MmaOptions {
                        collector_op: Some(op),
                        write_disable_mask: Some(wdm(group)),
                        ..Default::default()
                    }
                };
                if v == NGroupATmemShiftWdm {
                    tcgen05_mma(kind, CtaGroup::ONE, c.t(0), c.t(TIN), desc_b, id, group != 0, opts);
                } else {
                    tcgen05_mma(kind, CtaGroup::ONE, c.t(0), desc_a, desc_b, id, false, opts);
                }
            }
        } else if a_tmem {
            tcgen05_mma(kind, CtaGroup::ONE, c.t(0), c.t(TIN), desc_b_expanded, id, false, Default::default());
        } else {
            tcgen05_mma(kind, CtaGroup::ONE, c.t(0), desc_a, desc_b_expanded, id, false, Default::default());
        }
    })
}

#[test]
fn slicek_m32n8k128_variants_match_plain_rust() {
    use Variant::*;
    let a = ints(LM * LK, 120);
    let b = ints(LN * LK, 121);
    let code = |x: f32| F8E4M3FN::from_f32(x).0;
    let want = gemm_ref(&a, &b, LM, LN, LK);
    // Physical row pm = 4 lm + stripe holds A[lm][32 stripe .. 32 stripe + 32].
    let phys = |pm: usize, kb: usize| code(a[(pm >> 2) * LK + 32 * (pm & 3) + kb]);
    for v in [AsyncS32Smem, ATmemCoreMatrix, TmaS128MmaS32, TmaS32MmaS32, NGroupATmemShiftWdm, NGroupSmemAReuseWdm] {
        let mut img = vec![0u8; 8192];
        for r in 0..LN {
            for kb in 0..LK {
                img[4096 + sw128(r, kb)] = code(b[r * LK + kb]);
            }
        }
        // A as each variant's copy leaves it: the compact 32-byte swizzle (cp.async or a 3D TMA
        // box), or the logical rows in the 128-byte swizzle (a 2D TMA box).
        for pm in 0..128 {
            for kb in 0..32 {
                if v == TmaS128MmaS32 || v == ATmemCoreMatrix || v == NGroupATmemShiftWdm {
                    img[sw128(pm >> 2, 32 * (pm & 3) + kb)] = phys(pm, kb);
                } else {
                    img[sw32(pm, kb)] = phys(pm, kb);
                }
            }
        }
        // The TMEM variants stage physical row pm (read back from the 128-byte image) in lane pm.
        let rows: Vec<Vec<u32>> = (0..128)
            .map(|pm| (0..8).map(|w| u32::from_le_bytes([0, 1, 2, 3].map(|i| phys(pm, 4 * w + i)))).collect())
            .collect();
        let a_tmem = matches!(v, ATmemCoreMatrix | NGroupATmemShiftWdm);
        let tin = if a_tmem { tmem_image(8, &[(0, &rows)]) } else { Vec::new() };
        let n = if matches!(v, NGroupATmemShiftWdm | NGroupSmemAReuseWdm) { LN } else { 4 * LN };
        // The shifted N-group variant writes only the lanes of slot 0 (its WDM); the epilogue
        // reads all of them, as the example's does.
        let opts = SimOptions { detect_uninitialized: v != NGroupATmemShiftWdm, ..SimOptions::default() };
        let got = run(&slicek_kernel(v), &img, &img, &tin, n, &opts).unwrap_or_else(|e| panic!("slicek {v:?}: {e}"));
        let d = |pm: usize, col: usize| f32::from_bits(got[pm * n + col]) as f64;
        for lm in 0..LM {
            for nn in 0..LN {
                let c = match v {
                    NGroupATmemShiftWdm => d(4 * lm, nn),
                    NGroupSmemAReuseWdm => (0..4).map(|s| d(4 * lm + s, nn)).sum(),
                    _ => (0..4)
                        .map(|s| {
                            let pm = 4 * lm + s;
                            let visible = if v == TmaS128MmaS32 { s ^ ((pm >> 3) & 3) } else { s };
                            d(pm, visible * LN + nn)
                        })
                        .sum(),
                };
                assert_eq!(c, want[lm * LN + nn], "slicek {v:?}: C({lm}, {nn})");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Build-time checks, targets, MLIR and PTX.
// ---------------------------------------------------------------------------------------

fn panic_text(f: impl FnOnce()) -> String {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

fn build_one(issue: impl Fn(&Ctx) + 'static) -> Kernel {
    harness("diag", Arch::Sm100a, 4096, 0, 1, 0, issue)
}

#[test]
fn build_time_diagnostics() {
    let f16 = idesc(Ty::F16, DType::F32, 128, 64, false);
    // M = 256 with .ws (a pair's M in the descriptor).
    let m256 = Tcgen05InstrDesc::build(&Tcgen05InstrDescFields { n: 64, m: 256, ..Default::default() }).unwrap();
    let e = panic_text(|| {
        build_one(move |c| {
            tcgen05_mma_ws(
                Tcgen05MmaKind::F16,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(0)),
                m256,
                false,
                Default::default(),
            )
        });
    });
    assert!(e.contains("M of 32, 64 or 128") && e.contains("M = 256"), "{e}");
    // .ashift with collector::a::use.
    let e = panic_text(|| {
        build_one(move |c| {
            let o = Tcgen05MmaOptions {
                a_shift: true,
                collector_op: Some(Tcgen05MmaCollectorOp::USE),
                ..Default::default()
            };
            tcgen05_mma(Tcgen05MmaKind::F16, CtaGroup::ONE, c.t(0), c.t(64), sw128_desc(c.s(0)), f16, false, o)
        });
    });
    assert!(e.contains(".ashift with .collector::a::use"), "{e}");
    // .ashift with A in shared memory.
    let e = panic_text(|| {
        build_one(move |c| {
            let o = Tcgen05MmaOptions { a_shift: true, ..Default::default() };
            tcgen05_mma(
                Tcgen05MmaKind::F16,
                CtaGroup::ONE,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(0)),
                f16,
                false,
                o,
            )
        });
    });
    assert!(e.contains("from shared memory"), "{e}");
    // scale-input-d on kind::i8, and out of range.
    let i8d = idesc(Ty::I8, DType::I32, 128, 64, false);
    let e = panic_text(|| {
        build_one(move |c| {
            let o = Tcgen05MmaOptions { scale_input_d: Some(1), ..Default::default() };
            tcgen05_mma(
                Tcgen05MmaKind::I8,
                CtaGroup::ONE,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(0)),
                i8d,
                false,
                o,
            )
        });
    });
    assert!(e.contains("scale-input-d") && e.contains("kind::i8"), "{e}");
    let e = panic_text(|| {
        build_one(move |c| {
            let o = Tcgen05MmaOptions { scale_input_d: Some(16), ..Default::default() };
            tcgen05_mma(
                Tcgen05MmaKind::F16,
                CtaGroup::ONE,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(0)),
                f16,
                false,
                o,
            )
        });
    });
    assert!(e.contains("0 to 15"), "{e}");
    // An instruction descriptor whose formats are no kind::f16 operands (an F8F6F4 FP6 code).
    let bad = Tcgen05InstrDesc::from_bits(f16.bits().as_static().unwrap() | 3 << 7);
    let e = panic_text(|| {
        build_one(move |c| {
            tcgen05_mma(
                Tcgen05MmaKind::F16,
                CtaGroup::ONE,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(0)),
                bad,
                false,
                Default::default(),
            )
        });
    });
    assert!(e.contains("no .kind::f16 instruction"), "{e}");
    // A runtime descriptor.
    let e = panic_text(|| {
        build_one(move |c| {
            let rt = Tcgen05InstrDesc::from_bits(c.tmem.to::<u32>());
            tcgen05_mma(
                Tcgen05MmaKind::F16,
                CtaGroup::ONE,
                c.t(0),
                sw128_desc(c.s(0)),
                sw128_desc(c.s(0)),
                rt,
                false,
                Default::default(),
            )
        });
    });
    assert!(e.contains("runtime value"), "{e}");
    // The warp form's byte-id for .scale_vec::2X.
    let e = panic_text(|| {
        let c = WarpCase {
            kind: MmaBlockScaleKind::MXF4,
            fp4: true,
            sv: Tcgen05MmaScaleVec::X2,
            ue4m3: false,
            ids: (1, 0, 0, 0),
        };
        warp_kernel("diag_warp", c, Arch::Sm120a);
    });
    assert!(e.contains("byte-ids"), "{e}");
}

fn ws_kernel(arch: Arch, kind: Ty) -> Kernel {
    let id = idesc(kind, if kind == Ty::I8 { DType::I32 } else { DType::F32 }, 64, 128, false);
    harness(&format!("ws_t_{kind:?}"), arch, 32768, 0, 64, 0, move |c| {
        let o = Tcgen05MmaWsOptions {
            collector_b_buffer: Some(Tcgen05MmaCollectorBBuffer::B1),
            collector_op: Some(Tcgen05MmaCollectorOp::DISCARD),
            col_b_zero_mask: Some(Value::constant(0u64)),
        };
        tcgen05_mma_ws(kind.kind(), c.t(0), sw128_desc(c.s(0)), sw128_desc(c.s(B_OFF)), id, false, o);
    })
}

#[test]
fn targets() {
    let check = |k: &Kernel, a: Arch| cute::validate::check_target(k.ir(), a);
    let ws = ws_kernel(Arch::Sm100a, Ty::F16);
    for a in [Arch::Sm100a, Arch::Sm103a, Arch::Sm110a] {
        check(&ws, a).unwrap_or_else(|e| panic!("ws {a:?}: {}", e.message));
    }
    for a in [Arch::Sm90a, Arch::Sm120a, Arch::Sm100] {
        assert!(check(&ws, a).is_err(), "ws on {a:?}");
    }
    // kind::i8 .ws: the architecture-specific targets only (not sm_103a through sm_100f).
    let i8 = ws_kernel(Arch::Sm100a, Ty::I8);
    check(&i8, Arch::Sm100a).unwrap();
    let e = check(&i8, Arch::Sm103a).unwrap_err().message;
    assert!(e.contains("tcgen05.mma.ws.kind::i8"), "{e}");
    let c = WarpCase {
        kind: MmaBlockScaleKind::MXF4NVF4,
        fp4: true,
        sv: Tcgen05MmaScaleVec::X4,
        ue4m3: true,
        ids: (0, 0, 0, 0),
    };
    let warp = warp_kernel("t_warp", c, Arch::Sm120a);
    check(&warp, Arch::Sm120a).unwrap();
    assert!(check(&warp, Arch::Sm100a).is_err());
}

/// Compile `kernel` for `arch` and assemble it with `ptxas` when one is on PATH; the PTX, or
/// `None` without an `mlir-opt`.
fn ptx_for(kernel: &Kernel, arch: Arch) -> Option<String> {
    let tc = Toolchain::discover()?;
    let name = kernel.ir().name.clone();
    let ptx = match kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)) {
        Ok(c) => c.ptx().to_string(),
        // mlir-opt 21 knows no tcgen05 on sm_103a (LLVM 22 and 23 do).
        Err(e) if e.to_string().contains(&format!("is not supported on {}", arch.chip())) => {
            eprintln!("{name}: this mlir-opt has no tcgen05 for {arch:?}; skipped");
            return None;
        }
        Err(e) => panic!("{name} for {arch:?}: {e}"),
    };
    assert!(!ptx.contains(".extern .func"), "{name}: an unknown intrinsic became an external function:\n{ptx}");
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    if let Some(ptxas) = ptxas {
        let base = std::env::temp_dir().join(format!("cute-prims-mma-{}-{name}-{arch:?}", std::process::id()));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, &ptx).unwrap();
        let sm = format!("-arch={}", arch.chip());
        let out = std::process::Command::new(ptxas).arg(&sm).arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        let err = String::from_utf8_lossy(&out.stderr);
        if !out.status.success() && err.contains("not defined for option 'gpu-name'") {
            eprintln!("{name}: this ptxas has no {sm}; PTX checked as text only");
        } else {
            assert!(out.status.success(), "{name}: ptxas {sm}: {}{err}", String::from_utf8_lossy(&out.stdout));
            eprintln!("{name}: ptxas {sm} ok");
        }
    }
    Some(ptx)
}

/// The kernels of the MLIR/PTX check for `arch`, with text their MLIR must hold.
fn tc_kernels(arch: Arch) -> Vec<(Kernel, Vec<&'static str>)> {
    let f16 = idesc(Ty::F16, DType::F32, 128, 64, false);
    let tf32 = idesc(Ty::Tf32, DType::F32, 128, 64, false);
    let mods = harness("mods", arch, 32768, 0, 64, 0, move |c| {
        use Tcgen05MmaCollectorOp::*;
        let o = |op| Tcgen05MmaOptions { collector_op: Some(op), ..Default::default() };
        tcgen05_mma(
            Tcgen05MmaKind::F16,
            CtaGroup::ONE,
            c.t(0),
            sw128_desc(c.s(0)),
            sw128_desc(c.s(B_OFF)),
            f16,
            false,
            o(FILL),
        );
        let shift = Tcgen05MmaOptions {
            a_shift: true,
            collector_op: Some(LASTUSE),
            scale_input_d: Some(5),
            ..Default::default()
        };
        tcgen05_mma(Tcgen05MmaKind::TF32, CtaGroup::ONE, c.t(0), c.t(TIN), sw128_desc(c.s(B_OFF)), tf32, true, shift);
        let wdm = Tcgen05MmaOptions {
            a_shift: true,
            write_disable_mask: Some(vec![Value::constant(0); 4]),
            scale_input_d: Some(2),
            ..Default::default()
        };
        tcgen05_mma(Tcgen05MmaKind::F16, CtaGroup::ONE, c.t(0), c.t(TIN), sw128_desc(c.s(B_OFF)), f16, true, wdm);
    });
    let sp_id = idesc(Ty::E4M3, DType::F32, 32, 64, true);
    let ws_sp = harness("ws_sp_t", arch, 32768, 0, 64, 0, move |c| {
        let o = Tcgen05MmaWsOptions {
            collector_b_buffer: Some(Tcgen05MmaCollectorBBuffer::B3),
            collector_op: Some(Tcgen05MmaCollectorOp::FILL),
            col_b_zero_mask: None,
        };
        tcgen05_mma_ws_sp(
            Tcgen05MmaKind::F8F6F4,
            c.t(0),
            sw128_desc(c.s(0)),
            sw128_desc(c.s(B_OFF)),
            sp_id,
            false,
            c.t(TIN),
            o,
        );
    });
    let mx = Tcgen05MxOmmaInstrDesc::build(&Tcgen05MxOmmaInstrDescFields {
        sparse: true,
        a_format: Tcgen05InputFormat::MX_E2M1,
        b_format: Tcgen05InputFormat::MX_E2M1,
        scale_format: MxScaleFormat::UE4M3,
        n: 64,
        m: 128,
        ..Default::default()
    })
    .unwrap();
    let sp_bs = harness("sp_bs_t", arch, 32768, 0, 64, 0, move |c| {
        let (e, sa, sb) = (c.t(TIN), c.t(TIN + 2), c.t(TIN + 6));
        let (a, b) = (sw128_desc(c.s(0)), sw128_desc(c.s(B_OFF)));
        tcgen05_mma_sp_block_scale(
            Tcgen05MmaKind::MXF4NVF4,
            CtaGroup::ONE,
            c.t(0),
            a,
            b,
            mx,
            false,
            e,
            sa,
            sb,
            Default::default(),
        );
    });
    let ws = harness("ws_t", arch, 32768, 0, 64, 0, move |c| {
        let o = Tcgen05MmaWsOptions { col_b_zero_mask: Some(Value::constant(0u64)), ..Default::default() };
        tcgen05_mma_ws(Tcgen05MmaKind::F16, c.t(0), sw128_desc(c.s(0)), sw128_desc(c.s(B_OFF)), f16, false, o);
    });
    vec![
        (
            mods,
            vec![
                "tcgen05.mma.cta_group::1.kind::f16.collector::a::fill [",
                "tcgen05.mma.cta_group::1.kind::tf32.ashift.collector::a::lastuse [$0], [$1], $2, $3, q, 5;",
                "kind::f16.ashift [$0], [$1], $2, $3, {$6, $7, $8, $9}, q, 2;",
            ],
        ),
        (ws, vec!["tcgen05.mma.ws.cta_group::1.kind::f16 [$0], $1, $2, $3, q, $6;"]),
        (ws_sp, vec!["tcgen05.mma.ws.sp.cta_group::1.kind::f8f6f4.collector::b3::fill [$0], $1, $2, [$6], $3, q;"]),
        (
            sp_bs,
            vec![
                "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [$0], $1, $2, [$6], $3, [$7], [$8], q;",
            ],
        ),
    ]
}

#[test]
fn mlir_and_ptx() {
    if Toolchain::discover().is_none() {
        eprintln!("skipping the PTX: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    for arch in [Arch::Sm100a, Arch::Sm103a] {
        for (k, texts) in tc_kernels(arch) {
            let mlir = k.to_mlir();
            for t in texts {
                assert!(mlir.contains(t), "{}: no `{t}` in\n{mlir}", k.ir().name);
            }
            ptx_for(&k, arch);
        }
    }
    let c = WarpCase {
        kind: MmaBlockScaleKind::MXF4,
        fp4: true,
        sv: Tcgen05MmaScaleVec::X2,
        ue4m3: false,
        ids: (2, 1, 0, 3),
    };
    let warp = warp_kernel("warp_t", c, Arch::Sm120a);
    let c8 = WarpCase {
        kind: MmaBlockScaleKind::MXF8F6F4,
        fp4: false,
        sv: Tcgen05MmaScaleVec::X1,
        ue4m3: false,
        ids: (0, 0, 0, 0),
    };
    let warp8 = warp_kernel("warp8_t", c8, Arch::Sm120a);
    assert!(warp.to_mlir().contains("kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"));
    for k in [&warp, &warp8] {
        let mlir = k.to_mlir();
        assert!(mlir.contains("mma.sp::ordered_metadata.sync.aligned.m16n8k"), "{mlir}");
        ptx_for(k, Arch::Sm120a);
    }
}

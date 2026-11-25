//! TMEM loads and stores: `tcgen05.ld`/`tcgen05.st` copy atoms
//! of every shape, with several repetitions, packing and the `.16x32bx2` half split, their
//! waits, and `make_tmem_copy` round trips (one in the style of the epilogue of CUTLASS's
//! `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py`, BSD-3-Clause).
//! sm_100a: simulated, compiled for sm_100a and assembled by `tests/ptxas.rs`; no GPU here runs
//! them.

use cute::ir::TmemShape;
use cute::prelude::*;
use cute::tcgen05::{
    Ld16x32bx2Op, Ld16x64bOp, Ld16x128bOp, Ld16x256bOp, Ld32x32bOp, Pack, Repetition, St16x32bx2Op, St16x64bOp,
    St16x128bOp, St16x256bOp, St32x32bOp, Unpack, make_tmem_copy,
};

use super::Case;

const THREADS: usize = 128;
const COLUMNS: i32 = 512;

/// One store-then-load pair of the shapes kernel: the store's shape and repetition, its
/// `.unpack::16b`, the load's, its `.pack::16b`, whether the elements are 16-bit, and the
/// first column of the pair's region.
#[derive(Clone, Copy)]
struct Pair {
    st: (TmemShape, u32),
    unpack: bool,
    ld: (TmemShape, u32),
    pack: bool,
    half: bool,
    col: u32,
}

const fn p(st: (TmemShape, u32), ld: (TmemShape, u32), col: u32) -> Pair {
    Pair { st, unpack: false, ld, pack: false, half: false, col }
}

use TmemShape::{S16x32bx2, S16x64b, S16x128b, S16x256b, S32x32b};

/// Each warp stores to 32 lanes of its quadrant (a 16-lane shape with two instructions, 16
/// lanes apart) and loads them back with another shape.
const PAIRS: [Pair; 13] = [
    p((S32x32b, 32), (S16x64b, 16), 0),
    p((S32x32b, 32), (S16x128b, 8), 32),
    p((S32x32b, 32), (S16x256b, 4), 64),
    p((S32x32b, 32), (S16x32bx2, 16), 96),
    p((S16x64b, 8), (S32x32b, 16), 128),
    p((S16x128b, 1), (S32x32b, 4), 144),
    p((S16x256b, 2), (S16x32bx2, 8), 160),
    p((S16x32bx2, 2), (S16x64b, 2), 176),
    p((S32x32b, 128), (S32x32b, 128), 256),
    p((S32x32b, 1), (S32x32b, 1), 180),
    // 16-bit elements: a plain store read back packed, and unpacked stores read back packed.
    Pair { st: (S32x32b, 16), unpack: false, ld: (S32x32b, 8), pack: true, half: true, col: 192 },
    Pair { st: (S32x32b, 8), unpack: true, ld: (S16x64b, 4), pack: true, half: true, col: 208 },
    Pair { st: (S16x32bx2, 4), unpack: true, ld: (S16x128b, 2), pack: true, half: true, col: 224 },
];

fn instructions(shape: TmemShape) -> u32 {
    32 / shape.lanes()
}

/// Registers one thread loads in pair `p` (all its instructions).
fn loaded(p: &Pair) -> usize {
    (instructions(p.ld.0) * p.ld.0.registers(p.ld.1)) as usize
}

fn slots(half: bool) -> usize {
    PAIRS.iter().filter(|p| p.half == half).map(loaded).sum()
}

fn repetition(n: u32) -> Repetition {
    match n {
        1 => Repetition::X1,
        2 => Repetition::X2,
        4 => Repetition::X4,
        8 => Repetition::X8,
        16 => Repetition::X16,
        32 => Repetition::X32,
        64 => Repetition::X64,
        _ => Repetition::X128,
    }
}

fn load_op(shape: TmemShape, n: u32, pack: bool) -> CopyOp {
    let (r, pk) = (repetition(n), if pack { Pack::Pack16bIn32b } else { Pack::None });
    match shape {
        S16x64b => Ld16x64bOp::new(r).with_pack(pk).into(),
        S16x128b => Ld16x128bOp::new(r).with_pack(pk).into(),
        S16x256b => Ld16x256bOp::new(r).with_pack(pk).into(),
        S16x32bx2 => Ld16x32bx2Op::new(r).with_pack(pk).into(),
        S32x32b => Ld32x32bOp::new(r).with_pack(pk).into(),
    }
}

fn store_op(shape: TmemShape, n: u32, unpack: bool) -> CopyOp {
    let (r, u) = (repetition(n), if unpack { Unpack::Unpack32bIn16b } else { Unpack::None });
    match shape {
        S16x64b => St16x64bOp::new(r).with_unpack(u).into(),
        S16x128b => St16x128bOp::new(r).with_unpack(u).into(),
        S16x256b => St16x256bOp::new(r).with_unpack(u).into(),
        S16x32bx2 => St16x32bx2Op::new(r).with_unpack(u).into(),
        S32x32b => St32x32bOp::new(r).with_unpack(u).into(),
    }
}

/// The TMEM view one instruction of `shape` (`.x<n>`, packed) addresses: TMEM lane `lane`,
/// column `col`, laid out as the atom's TMEM side (each lane's elements, then the lanes).
fn view<T: Numeric>(base: &Ptr<T>, lane: Int32, col: u32, shape: TmemShape, n: u32, packed: bool) -> Tensor<T> {
    let per_word = 32 / T::DTYPE.bits();
    let lane_stride = (1usize << 16) * per_word;
    let bits_per_lane = 1024 * shape.registers(n) as usize / shape.lanes() as usize;
    let layout = if packed {
        // The low half of every column: every other 16-bit element.
        Layout::new((bits_per_lane / 16, shape.lanes() as usize), (2, lane_stride))
    } else {
        Layout::new((bits_per_lane / T::DTYPE.bits(), shape.lanes() as usize), (1, lane_stride))
    };
    let p = base.clone() + (lane * lane_stride as i32 + (col as usize * per_word) as i32);
    make_tensor(p, Layout::from_modes([layout]))
}

/// The value thread `t` stores as element `e` of pair `i`.
fn stored(i: usize, t: usize, e: usize, half: bool) -> u32 {
    if half { ((t << 8) | e) as u32 & 0xffff } else { ((i << 24) | (t << 12) | e) as u32 }
}

/// Store then load pair `i` with elements of `T` (u32, or u16 for the 16-bit pairs).
fn run_pair<T: Numeric>(i: usize, base: &Ptr<T>, out: &Tensor<T>, slot: usize, per_thread: usize) {
    let pr = PAIRS[i];
    let per_word = 32 / T::DTYPE.bits();
    let t = thread_idx().x;
    let quadrant = warp_idx() % 4 * 32;
    let st = make_copy_atom::<T>(store_op(pr.st.0, pr.st.1, pr.unpack));
    let ld = make_copy_atom::<T>(load_op(pr.ld.0, pr.ld.1, pr.pack));
    let st_elems = pr.st.0.registers(pr.st.1) as usize * per_word;
    for k in 0..instructions(pr.st.0) {
        let frag = make_fragment::<T>(st_elems);
        for e in 0..st_elems {
            // stored(i, t, e, half), with t at run time.
            let all = k as usize * st_elems + e;
            let tt = t.to::<u32>();
            let v = if pr.half { (tt << 8u32) + all as u32 } else { (tt << 12u32) + ((i << 24) | all) as u32 };
            frag.store(e, v.to::<T>());
        }
        let lane = quadrant + (16 * k) as i32;
        copy(&st, &frag, &view(base, lane, pr.col, pr.st.0, pr.st.1, pr.unpack));
    }
    fence_view_async_tmem_store();
    let ld_elems = pr.ld.0.registers(pr.ld.1) as usize * per_word;
    let mut frags = Vec::new();
    for k in 0..instructions(pr.ld.0) {
        let frag = make_fragment::<T>(ld_elems);
        let lane = quadrant + (16 * k) as i32;
        copy(&ld, &view(base, lane, pr.col, pr.ld.0, pr.ld.1, pr.pack), &frag);
        frags.push(frag);
    }
    fence_view_async_tmem_load();
    for (k, frag) in frags.iter().enumerate() {
        for e in 0..ld_elems {
            out.store(t * per_thread as i32 + (slot + k * ld_elems + e) as i32, frag.load(e));
        }
    }
}

fn shapes_kernel() -> Kernel {
    Kernel::builder("tmem_ldst_shapes").block_dim([THREADS as u32, 1, 1]).target(Arch::Sm100a).build(|| {
        let (n32, n16) = (slots(false), 2 * slots(true));
        let out32 = param_tensor::<u32>("out32", Layout::col_major(THREADS * n32));
        let out16 = param_tensor::<u16>("out16", Layout::col_major(THREADS * n16));
        let slot = alloc_smem::<u32>(1, 4);
        let barrier = pipeline::NamedBarrier::new(1, THREADS as u32);
        if_(warp_idx().eq(0), || alloc_tmem(COLUMNS, &slot, None, Arch::Sm100a));
        barrier.arrive_and_wait();
        let p32 = retrieve_tmem_ptr::<u32>(16, &slot);
        let p16 = retrieve_tmem_ptr::<u16>(16, &slot);
        let (mut s32, mut s16) = (0, 0);
        for (i, pr) in PAIRS.iter().enumerate() {
            if pr.half {
                run_pair::<u16>(i, &p16, &out16, s16, n16);
                s16 += 2 * loaded(pr);
            } else {
                run_pair::<u32>(i, &p32, &out32, s32, n32);
                s32 += loaded(pr);
            }
        }
        barrier.arrive_and_wait();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&p32, COLUMNS, None, Arch::Sm100a)
        });
    })
}

/// The registers (32-bit) one warp's quadrant (32 lanes, from column `col`) holds after
/// pair `i`'s stores, by (lane, column), from the fragment maps of the PTX ISA.
fn check_shapes(args: &[KernelArg]) {
    let out32 = args[0].to_vec::<u32>();
    let out16 = args[1].to_vec::<u16>();
    let (n32, n16) = (slots(false), 2 * slots(true));
    for w in 0..4usize {
        let (mut s32, mut s16) = (0, 0);
        for (i, pr) in PAIRS.iter().enumerate() {
            // TMEM cells of the quadrant: (lane 0..32, column offset) -> bits.
            let mut cells = std::collections::HashMap::<(u32, u32), u32>::new();
            let st_regs = pr.st.0.registers(pr.st.1);
            for k in 0..instructions(pr.st.0) {
                for lane in 0..32u32 {
                    let t = 32 * w + lane as usize;
                    for r in 0..st_regs {
                        let reg = if pr.half {
                            let e = (k * st_regs + r) as usize * 2;
                            stored(i, t, e, true) | (stored(i, t, e + 1, true) << 16)
                        } else {
                            stored(i, t, (k * st_regs + r) as usize, false)
                        };
                        for (h, (l, c)) in pr.st.0.cells(lane, r, pr.st.1, pr.unpack).into_iter().enumerate() {
                            let bits = if pr.unpack { (reg >> (16 * h)) & 0xffff } else { reg };
                            cells.insert((l + 16 * k, c), bits);
                        }
                    }
                }
            }
            let ld_regs = pr.ld.0.registers(pr.ld.1);
            for k in 0..instructions(pr.ld.0) {
                for lane in 0..32u32 {
                    let t = 32 * w + lane as usize;
                    for r in 0..ld_regs {
                        let mut reg = 0u32;
                        for (h, (l, c)) in pr.ld.0.cells(lane, r, pr.ld.1, pr.pack).into_iter().enumerate() {
                            let cell = *cells.get(&(l + 16 * k, c)).unwrap_or_else(|| {
                                panic!("pair {i}: the load reads lane {}, column {c}, which the store left", l + 16 * k)
                            });
                            reg |= if pr.pack { (cell & 0xffff) << (16 * h) } else { cell };
                        }
                        let j = (k * ld_regs + r) as usize;
                        if pr.half {
                            let at = t * n16 + s16 + 2 * j;
                            assert_eq!(
                                (out16[at], out16[at + 1]),
                                (reg as u16, (reg >> 16) as u16),
                                "pair {i}: thread {t}, register {j}"
                            );
                        } else {
                            assert_eq!(out32[t * n32 + s32 + j], reg, "pair {i}: thread {t}, register {j}");
                        }
                    }
                }
            }
            if pr.half {
                s16 += 2 * loaded(pr);
            } else {
                s32 += loaded(pr);
            }
        }
    }
}

pub fn tmem_ldst_shapes() -> Case {
    Case {
        name: "tmem_ldst_shapes",
        kernel: shapes_kernel(),
        cfg: LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]),
        small_only: false,
        check: check_shapes,
    }
}

/// A 128-thread `make_tmem_copy` round trip of a (128, 64) Float32 TMEM tensor: each thread
/// stores its part of `input` with a `St16x256bOp` tiled copy, and loads it back with a
/// `Ld32x32bOp(x64)` one into `output`.
fn copy_f32_kernel() -> Kernel {
    Kernel::builder("tmem_copy_f32").block_dim([THREADS as u32, 1, 1]).target(Arch::Sm100a).build(|| {
        let input = param_tensor::<f32>("input", Layout::new((128, 64), (64, 1)));
        let output = param_tensor::<f32>("output", Layout::new((128, 64), (64, 1)));
        let slot = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        let barrier = pipeline::NamedBarrier::new(1, THREADS as u32);
        if_(warp_idx().eq(0), || alloc_tmem(64, &slot, None, Arch::Sm100a));
        barrier.arrive_and_wait();
        let tmem = make_tensor(retrieve_tmem_ptr::<f32>(16, &slot), Layout::new((128, 64), (1 << 16, 1)));
        let st = make_tmem_copy(&make_copy_atom::<f32>(St16x256bOp::new(Repetition::X2)), &tmem);
        let st_thr = st.get_slice(t);
        let frag = make_fragment_like::<f32, _>(&st_thr.partition_s(&input));
        basic_copy(&st_thr.partition_s(&input), &frag);
        copy(&st, &frag, &st_thr.partition_d(&tmem));
        fence_view_async_tmem_store();
        // Other warps' quadrants are untouched, but the barrier would order them.
        barrier.arrive_and_wait();
        let ld = make_tmem_copy(&make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X64)), &tmem);
        let ld_thr = ld.get_slice(t);
        let back = make_fragment_like::<f32, _>(&ld_thr.partition_d(&output));
        copy(&ld, &ld_thr.partition_s(&tmem), &back);
        fence_view_async_tmem_load();
        basic_copy(&back, &ld_thr.partition_d(&output));
        barrier.arrive_and_wait();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&tmem.ptr(), 64, None, Arch::Sm100a)
        });
    })
}

fn check_round_trip<T: Numeric + PartialEq + std::fmt::Debug>(args: &[KernelArg]) {
    let (input, output) = (args[0].to_vec::<T>(), args[1].to_vec::<T>());
    assert_eq!(input, output, "the TMEM round trip gives the input back");
}

pub fn tmem_copy_f32() -> Case {
    Case {
        name: "tmem_copy_f32",
        kernel: copy_f32_kernel(),
        cfg: LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]),
        small_only: false,
        check: check_round_trip::<f32>,
    }
}

/// A (128, 128) Float16 accumulator in TMEM, stored whole with a `St32x32bOp(x64)` tiled copy
/// and read back in four (128, 32) epilogue subtiles (`zipped_divide` by the tile, as the
/// tutorial's epilogue splits its accumulator) with a `Ld16x64bOp(x8)` tiled copy made for
/// one subtile, then stored to global memory per subtile.
fn copy_f16_kernel() -> Kernel {
    Kernel::builder("tmem_copy_f16").block_dim([THREADS as u32, 1, 1]).target(Arch::Sm100a).build(|| {
        let input = param_tensor::<F16>("input", Layout::new((128, 128), (128, 1)));
        let output = param_tensor::<F16>("output", Layout::new((128, 128), (128, 1)));
        let slot = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        let barrier = pipeline::NamedBarrier::new(1, THREADS as u32);
        if_(warp_idx().eq(0), || alloc_tmem(64, &slot, None, Arch::Sm100a));
        barrier.arrive_and_wait();
        let acc = make_tensor(retrieve_tmem_ptr::<F16>(16, &slot), Layout::new((128, 128), (1 << 17, 1)));
        let st = make_tmem_copy(&make_copy_atom::<F16>(St32x32bOp::new(Repetition::X64)), &acc);
        let st_thr = st.get_slice(t);
        let frag = make_fragment_like::<F16, _>(&st_thr.partition_s(&input));
        basic_copy(&st_thr.partition_s(&input), &frag);
        copy(&st, &frag, &st_thr.partition_d(&acc));
        fence_view_async_tmem_store();
        barrier.arrive_and_wait();
        // (EpiTile, NumTiles)
        let epi_tiler = Tiler::from(&IntTuple::from(vec![128usize, 32]));
        let acc_epi = zipped_divide(&acc, epi_tiler.clone());
        let out_epi = zipped_divide(&output, epi_tiler);
        let ld = make_tmem_copy(&make_copy_atom::<F16>(Ld16x64bOp::new(Repetition::X8)), &acc_epi.slice((.., 0)));
        let ld_thr = ld.get_slice(t);
        // (TmemCpy, NumTmemCpy, NumTiles)
        let t_acc = ld_thr.partition_s(&acc_epi);
        let t_out = ld_thr.partition_d(&out_epi);
        for tile in 0..4i32 {
            let frag = make_fragment_like::<F16, _>(&t_out.slice((.., .., tile)));
            copy(&ld, &t_acc.slice((.., .., tile)), &frag);
            fence_view_async_tmem_load();
            basic_copy(&frag, &t_out.slice((.., .., tile)));
        }
        barrier.arrive_and_wait();
        if_(warp_idx().eq(0), || {
            relinquish_tmem_alloc_permit(None);
            dealloc_tmem(&acc.ptr(), 64, None, Arch::Sm100a)
        });
    })
}

pub fn tmem_copy_f16() -> Case {
    Case {
        name: "tmem_copy_f16",
        kernel: copy_f16_kernel(),
        cfg: LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]),
        small_only: false,
        check: check_round_trip::<F16>,
    }
}

/// The TMEM load and store kernels of the sm100 group.
pub fn tmem_ldst_cases() -> Vec<Case> {
    vec![tmem_ldst_shapes(), tmem_copy_f32(), tmem_copy_f16()]
}

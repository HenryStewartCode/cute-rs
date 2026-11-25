// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block-scaled `mma.sync` on SM120: one-warp
//! kernels for MXF4, NVFP4, MXF8 (both formats) and the mixed FP4 x FP8 forms on the
//! simulator against plain Rust (bit-exact with UE8M0 scales, within a tolerance with UE4M3
//! ones); a block-scaled GEMM tile placing the scale factors with the `utils::sm120` helpers
//! and the SM120 shared-memory layouts; the constructors' and targets' diagnostics; the
//! MLIR; and the sm_120a PTX (assembled by `ptxas` when it is on PATH).

use cute::prelude::*;

type Fp4 = cute::F4E2M1FN;
type E4 = cute::F8E4M3FN;
type E5 = cute::F8E5M2;
type Ue8 = cute::F8E8M0FNU;
type H16 = cute::F16;
use cute::utils::sm120::{
    get_layout_sfa_tv, get_layout_sfb_tv, get_permutation_mnk, partition_fragment_sfa, partition_fragment_sfb,
    partition_sfa, partition_sfb, sm120_get_smem_store_op,
};
use cute::utils::{sm120_make_smem_layout_sfa, sm120_make_smem_layout_sfb, tile_atom_to_shape_sf};
use cute::warp::{Field, MXF8F6F4_SUPPORTED_PAIRS, MmaMXF4NVF4Op, MmaMXF4Op, MmaMXF8F6F4Op, MmaMXF8Op};

/// Host values of the narrow formats. Kernels take the operands as bytes, one value per
/// byte (an FP4 value in the low nibble), since a packed FP4 tensor cannot be read element
/// by element.
trait Host: Numeric + Copy {
    fn host(x: f32) -> Self;
    fn val(self) -> f64;
    fn byte(self) -> u8;
    /// The value in the byte `v`, in a kernel.
    fn from_byte(v: Value<u8>) -> Value<Self>;
}

macro_rules! host {
    ($($t:ty),*) => {$(
        impl Host for $t {
            fn host(x: f32) -> Self {
                <$t>::from_f32(x)
            }
            fn val(self) -> f64 {
                self.to_f64()
            }
            fn byte(self) -> u8 {
                self.raw_bits() as u8
            }
            fn from_byte(v: Value<u8>) -> Value<Self> {
                if <$t>::DTYPE == DType::F4E2M1FN {
                    let [lo, _]: [Value<Self>; 2] = unpack(v);
                    lo
                } else {
                    v.bitcast()
                }
            }
        }
    )*};
}
host!(F4E2M1FN, F8E4M3FN, F8E5M2, F8E8M0FNU);

/// The operand bytes of `values`.
fn bytes<T: Host>(values: &[T]) -> KernelArg {
    KernelArg::buffer(&values.iter().map(|v| v.byte()).collect::<Vec<u8>>())
}

/// Fill the fragment `frag` from `part`, this thread's partition of an operand's bytes.
fn fill<T: Host>(frag: &Tensor<T>, part: &Tensor<u8>) {
    for i in 0..frag.size() {
        frag.store(i, T::from_byte(part.load(i)));
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn pick(&mut self, set: &[f32]) -> f32 {
        set[(self.next() % set.len() as u64) as usize]
    }
}

/// Values every operand format holds exactly (E2M1's).
const VALUES: [f32; 15] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
/// Power-of-two scales: the sums stay exact in Float32.
const UE8M0_SCALES: [f32; 3] = [0.5, 1.0, 2.0];
/// UE4M3 scales with mantissa bits.
const UE4M3_SCALES: [f32; 5] = [0.75, 1.0, 1.125, 1.5, 1.75];

/// The scale-factor tile of an `(mn, k)` operand in global memory: one scale per `vec`
/// elements of K, the `k / vec` of a row adjacent.
fn sf_layout(mn: usize, k: usize, vec: usize) -> Layout {
    let nsf = k / vec;
    Layout::new((mn, (vec, nsf)), (nsf, (0, 1)))
}

/// One warp, `C = sum over K of (A SFA)(B SFB)` with K two instructions deep. With
/// `fields`, the scale factors are given as the atom's SFA and SFB fields instead of with
/// the operands.
fn one_warp_kernel<TA: Host, TB: Host, S: Host>(name: &'static str, atom: MmaAtom, vec: usize, fields: bool) -> Kernel {
    let [m, n, ka] = atom.shape_mnk();
    let k = 2 * ka;
    Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<u8>("a", Layout::new((m, k), (k, 1)));
        let b = param_tensor::<u8>("b", Layout::new((n, k), (k, 1)));
        let sfa = param_tensor::<S>("sfa", sf_layout(m, k, vec));
        let sfb = param_tensor::<S>("sfb", sf_layout(n, k, vec));
        let c = param_tensor::<f32>("c", Layout::new((m, n), (n, 1)));
        let mut mma = make_tiled_mma(atom.clone(), Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let (ta, tb) = (thr.partition_a(&a), thr.partition_b(&b));
        let (fa, fb) = (thr.partition_fragment_a::<TA, _>(&a), thr.partition_fragment_b::<TB, _>(&b));
        fill(&fa, &ta);
        fill(&fb, &tb);
        let (fsfa, fsfb) = (partition_fragment_sfa(&sfa, &thr), partition_fragment_sfb(&sfb, &thr));
        let (psfa, psfb) = (partition_sfa(&sfa, &thr), partition_sfb(&sfb, &thr));
        assert_eq!((fsfa.size(), fsfb.size()), (ka * 2, ka * 2), "{name}: one scale per K element per MMA");
        for i in 0..psfa.size() {
            fsfa.store(i, psfa.load(i));
        }
        for i in 0..psfb.size() {
            fsfb.store(i, psfb.load(i));
        }
        let acc = thr.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0f32);
        if fields {
            mma.set_tensor(Field::SFA, &fsfa);
            mma.set_tensor(Field::SFB, &fsfb);
            gemm(&mma, &acc, &fa, &fb, &acc);
        } else {
            gemm(&mma, &acc, (&fa, &fsfa), (&fb, &fsfb), &acc);
        }
        let tc = thr.partition_c(&c);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    })
}

/// Random operands and scales for an `(m, n, k)` problem.
struct Problem<TA, TB, S> {
    a: Vec<TA>,
    b: Vec<TB>,
    sfa: Vec<S>,
    sfb: Vec<S>,
}

fn problem<TA: Host, TB: Host, S: Host>(m: usize, n: usize, k: usize, vec: usize, seed: u64) -> Problem<TA, TB, S> {
    let mut r = Rng(seed);
    let scales: &[f32] = if vec == 16 { &UE4M3_SCALES } else { &UE8M0_SCALES };
    Problem {
        a: (0..m * k).map(|_| TA::host(r.pick(&VALUES))).collect(),
        b: (0..n * k).map(|_| TB::host(r.pick(&VALUES))).collect(),
        sfa: (0..m * k / vec).map(|_| S::host(r.pick(scales))).collect(),
        sfb: (0..n * k / vec).map(|_| S::host(r.pick(scales))).collect(),
    }
}

/// `C[i][j]` (row-major) from row-major A, B and scale factors (the `k / vec` of a row
/// adjacent), each product rounded to Float32 and added in order of K.
fn reference<TA: Host, TB: Host, S: Host>(
    p: &Problem<TA, TB, S>,
    m: usize,
    n: usize,
    k: usize,
    vec: usize,
) -> Vec<f32> {
    let nsf = k / vec;
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).fold(0f32, |acc, kk| {
                let x = p.a[i * k + kk].val() * p.sfa[i * nsf + kk / vec].val();
                acc + (x * p.b[j * k + kk].val() * p.sfb[j * nsf + kk / vec].val()) as f32
            });
        }
    }
    out
}

fn run_one_warp<TA: Host, TB: Host, S: Host>(name: &'static str, atom: impl Into<MmaAtom>, vec: usize, fields: bool) {
    let atom: MmaAtom = atom.into();
    let [m, n, ka] = atom.shape_mnk();
    let k = 2 * ka;
    let kernel = one_warp_kernel::<TA, TB, S>(name, atom, vec, fields);
    let p = problem::<TA, TB, S>(m, n, k, vec, 17 + ka as u64 + vec as u64);
    let mut args = vec![
        bytes(&p.a),
        bytes(&p.b),
        KernelArg::buffer(&p.sfa),
        KernelArg::buffer(&p.sfb),
        KernelArg::buffer(&vec![0f32; m * n]),
    ];
    let report =
        kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.mma_instructions, 2, "{name}");
    let want = reference(&p, m, n, k, vec);
    let got = args[4].to_vec::<f32>();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if vec == 16 {
            assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{name}: C({}, {}) is {g}, not {w}", i / n, i % n);
        } else {
            assert_eq!(g, w, "{name}: C({}, {})", i / n, i % n);
        }
    }
    assert!(want.iter().any(|&w| w != 0.0), "{name}: a trivial reference");
}

#[test]
fn one_warp_block_scaled_mmas_match_the_rust_reference() {
    use DType::*;
    run_one_warp::<Fp4, Fp4, Ue8>("mxf4", MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU), 32, false);
    run_one_warp::<Fp4, Fp4, E4>("nvf4", MmaMXF4NVF4Op::new(F4E2M1FN, F32, F8E4M3FN), 16, false);
    run_one_warp::<E4, E4, Ue8>("mxf8_e4m3", MmaMXF8Op::new(F8E4M3FN, F32, F8E8M0FNU), 32, false);
    run_one_warp::<E5, E5, Ue8>("mxf8_e5m2", MmaMXF8Op::new(F8E5M2, F32, F8E8M0FNU), 32, false);
    let mixed = |a, b| MmaMXF8F6F4Op::new(a, b, F32, F8E8M0FNU);
    run_one_warp::<Fp4, E4, Ue8>("mixed_e2m1_e4m3", mixed(F4E2M1FN, F8E4M3FN), 32, false);
    run_one_warp::<Fp4, E5, Ue8>("mixed_e2m1_e5m2", mixed(F4E2M1FN, F8E5M2), 32, false);
    run_one_warp::<E4, Fp4, Ue8>("mixed_e4m3_e2m1", mixed(F8E4M3FN, F4E2M1FN), 32, false);
    run_one_warp::<E5, Fp4, Ue8>("mixed_e5m2_e2m1", mixed(F8E5M2, F4E2M1FN), 32, true);
    // The scale factors as the atom's fields.
    run_one_warp::<Fp4, Fp4, Ue8>("mxf4_fields", MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU), 32, true);
}

/// A `(128, 128, K)` tile by eight warps (`(4, 2, 1)` atoms with CuTe DSL's SM120
/// permutation): A and B fragments loaded from global memory with plain loads, the scale
/// factors staged in shared memory in the SM120 layouts and copied into the fragments of
/// `partition_fragment_sfa`/`sfb` by tiled copies over `get_layout_sfa_tv`/`sfb_tv`.
fn tile_kernel<TA: Host, TB: Host, S: Host>(name: &'static str, atom: MmaAtom, vec: usize, k: usize) -> Kernel {
    const T: usize = 128;
    let use_mxf8f6f4 = atom.shape_mnk()[2] == 32;
    Kernel::builder(name).block_dim([256, 1, 1]).build(|| {
        let a = param_tensor::<u8>("a", Layout::new((T, k), (k, 1)));
        let b = param_tensor::<u8>("b", Layout::new((T, k), (k, 1)));
        let gsfa = param_tensor::<S>("sfa", tile_atom_to_shape_sf((T, k), vec));
        let gsfb = param_tensor::<S>("sfb", tile_atom_to_shape_sf((T, k), vec));
        let c = param_tensor::<f32>("c", Layout::new((T, T), (T, 1)));
        let perm = get_permutation_mnk((T, T, k), vec, use_mxf8f6f4);
        let mma = make_tiled_mma(atom.clone(), Layout::col_major((4, 2, 1))).with_permutation_mnk(perm);
        let two = |l: Layout| Layout::from_modes([l.mode(0), l.mode(1)]);
        let ssfa = make_smem_tensor::<S>(two(sm120_make_smem_layout_sfa(&mma, (T, T, k), vec, 1)));
        let ssfb = make_smem_tensor::<S>(two(sm120_make_smem_layout_sfb(&mma, (T, T, k), vec, 1)));
        // Stage the scale factors: one store per (row, K-block).
        let t = thread_idx().x;
        let blocks = (T * k / vec) as i32;
        for_range(0, blocks / 256, 1, |j| {
            let idx = j * 256 + t;
            let (r, kb) = (idx % T as i32, (idx / T as i32) * vec as i32);
            ssfa.store((r, kb), gsfa.load((r, kb)));
            ssfb.store((r, kb), gsfb.load((r, kb)));
        });
        sync_threads();
        let thr = mma.get_slice(t);
        let (ta, tb) = (thr.partition_a(&a), thr.partition_b(&b));
        let (fa, fb) = (thr.partition_fragment_a::<TA, _>(&a), thr.partition_fragment_b::<TB, _>(&b));
        fill(&fa, &ta);
        fill(&fb, &tb);
        let fsfa = partition_fragment_sfa(&ssfa, &thr);
        let fsfb = partition_fragment_sfb(&ssfb, &thr);
        let atom_sf = make_copy_atom::<S>(CopyOp::Universal { bits: 8 });
        let tile = |i: usize| Layout::new(mma.tile_size(i), 1);
        let copy_a = make_tiled_copy(
            &atom_sf,
            &get_layout_sfa_tv(&mma),
            Tree::Branch(vec![Tree::Leaf(Some(tile(0))), Tree::Leaf(Some(tile(2)))]),
        );
        let copy_b = make_tiled_copy(
            &atom_sf,
            &get_layout_sfb_tv(&mma),
            Tree::Branch(vec![Tree::Leaf(Some(tile(1))), Tree::Leaf(Some(tile(2)))]),
        );
        for (tc, s, f) in [(&copy_a, &ssfa, &fsfa), (&copy_b, &ssfb, &fsfb)] {
            let thr_copy = tc.get_slice(t);
            copy(tc, &thr_copy.partition_s(s), &thr_copy.retile_d(f));
        }
        let acc = thr.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0f32);
        gemm(&mma, &acc, (&fa, &fsfa), (&fb, &fsfb), &acc);
        let tcc = thr.partition_c(&c);
        for i in 0..acc.size() {
            tcc.store(i, acc.load(i));
        }
    })
}

fn run_tile<TA: Host, TB: Host, S: Host>(name: &'static str, atom: impl Into<MmaAtom>, vec: usize, k: usize) {
    const T: usize = 128;
    let atom: MmaAtom = atom.into();
    let ka = atom.shape_mnk()[2];
    let kernel = tile_kernel::<TA, TB, S>(name, atom, vec, k);
    let p = problem::<TA, TB, S>(T, T, k, vec, 5);
    // The global scale factors in tile_atom_to_shape_SF's layout.
    let place = |sf: &[S]| -> Vec<S> {
        let l = tile_atom_to_shape_sf((T, k), vec);
        let mut out = vec![S::host(1.0); l.cosize()];
        for r in 0..T {
            for kk in (0..k).step_by(vec) {
                out[l.crd2idx(&[r, kk].into())] = sf[r * (k / vec) + kk / vec];
            }
        }
        out
    };
    let mut args = vec![
        bytes(&p.a),
        bytes(&p.b),
        KernelArg::buffer(&place(&p.sfa)),
        KernelArg::buffer(&place(&p.sfb)),
        KernelArg::buffer(&vec![0f32; T * T]),
    ];
    let report =
        kernel.simulate(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.mma_instructions as usize, (T / 16) * (T / 8) * (k / ka), "{name}");
    let want = reference(&p, T, T, k, vec);
    let got = args[4].to_vec::<f32>();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "{name}: C({}, {})", i / T, i % T);
    }
}

#[test]
fn block_scaled_gemm_tiles_match_the_rust_reference() {
    use DType::*;
    run_tile::<Fp4, Fp4, Ue8>("tile_mxf4", MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU), 32, 128);
    run_tile::<E4, Fp4, Ue8>("tile_mixed_e4m3_e2m1", MmaMXF8F6F4Op::new(F8E4M3FN, F4E2M1FN, F32, F8E8M0FNU), 32, 128);
}

/// The helpers' layouts, checked against the PTX ISA's scale-factor ownership: lanes `4 g`
/// and `4 g + 1` (thread-id 0) of each warp hold rows `g` and `g + 8` of SFA, lane `4 g`
/// column `g` of SFB, each value of the first mode one K element of the instruction.
#[test]
fn scale_factor_layouts() {
    use DType::*;
    let mma = make_tiled_mma(MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU), Layout::col_major((1, 1, 1)));
    let a = get_layout_sfa_tv(&mma);
    let b = get_layout_sfb_tv(&mma);
    for lane in 0..32usize {
        for v in 0..64usize {
            let (row, kk) = (a.crd2idx(&[lane, v].into()) % 16, a.crd2idx(&[lane, v].into()) / 16);
            assert_eq!((row, kk), (lane / 4 + 8 * (lane % 2), v), "SFA lane {lane} value {v}");
            let (col, kk) = (b.crd2idx(&[lane, v].into()) % 8, b.crd2idx(&[lane, v].into()) / 8);
            assert_eq!((col, kk), (lane / 4, v), "SFB lane {lane} value {v}");
        }
    }
    let perm = get_permutation_mnk((128, 128, 128), 16, false);
    assert_eq!(perm[1].as_ref().unwrap().to_string(), "(8,2,2):(1,16,8)");
    assert_eq!(perm[2].as_ref().unwrap().size(), 64);
    assert_eq!(get_permutation_mnk((64, 128, 64), 32, true)[0].as_ref().unwrap().size(), 64);
    let tiled = make_tiled_mma(MmaMXF8Op::new(F8E4M3FN, F32, F8E8M0FNU), Layout::col_major((4, 2, 1)));
    let l = sm120_make_smem_layout_sfa(&tiled, (128, 128, 256), 32, 2);
    assert_eq!(l.to_string(), "(((32,4),1),((32,1),4,2),2):(((16,4),512),((0,1),1,512),1024)");
    let l = sm120_make_smem_layout_sfb(&mma, (128, 256, 128), 16, 1);
    assert_eq!(l.to_string(), "(((32,4),2),((16,4),1,2),1):(((16,4),512),((0,1),4,1024),2048)");
    let _ = sm120_get_smem_store_op::<H16>(LayoutEnum::ROW_MAJOR, F32);
    let _ = sm120_get_smem_store_op::<f32>(LayoutEnum::COL_MAJOR, F32);
}

fn panic_text(f: impl FnOnce()) -> String {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn constructor_diagnostics() {
    use DType::*;
    for (a, b) in MXF8F6F4_SUPPORTED_PAIRS {
        assert_eq!(MmaMXF8F6F4Op::new(a, b, F32, F8E8M0FNU).kind().ptx().matches("kind::mxf8f6f4").count(), 1);
    }
    let e = panic_text(|| {
        MmaMXF8F6F4Op::new(F8E4M3FN, F8E5M2, F32, F8E8M0FNU);
    });
    assert!(e.contains("no form for A") && e.contains("MXF8F6F4_SUPPORTED_PAIRS"), "{e}");
    let e = panic_text(|| {
        MmaMXF8F6F4Op::new(F6E2M3FN, F8E4M3FN, F32, F8E8M0FNU);
    });
    assert!(e.contains("MXF8F6F4_SUPPORTED_PAIRS"), "{e}");
    let e = panic_text(|| {
        MmaMXF8F6F4Op::new(F4E2M1FN, F4E2M1FN, F32, F8E8M0FNU);
    });
    assert!(e.contains("use MmaMXF4Op or MmaMXF4NVF4Op"), "{e}");
    // The scale-factor type of each vector size.
    let e = panic_text(|| {
        MmaMXF4Op::new(F4E2M1FN, F32, F8E4M3FN);
    });
    assert!(e.contains("vectors of 32 elements take Float8E8M0FNU"), "{e}");
    let e = panic_text(|| {
        MmaMXF4NVF4Op::new(F4E2M1FN, F32, F8E8M0FNU);
    });
    assert!(e.contains("vectors of 16 elements take Float8E4M3FN"), "{e}");
    let e = panic_text(|| {
        MmaMXF8Op::new(F4E2M1FN, F32, F8E8M0FNU);
    });
    assert!(e.contains("takes Float8E4M3FN or Float8E5M2 operands"), "{e}");
    let e = panic_text(|| {
        MmaMXF8Op::new(F8E4M3FN, F16, F8E8M0FNU);
    });
    assert!(e.contains("accumulates in Float32 only"), "{e}");
    assert_eq!(
        MmaMXF4NVF4Op::new(F4E2M1FN, F32, F8E4M3FN).kind().ptx(),
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"
    );
    assert_eq!(
        MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU).kind().ptx(),
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"
    );
    // A block-scaled MMA without scale factors, or with the wrong type.
    let e = panic_text(|| {
        one_warp_missing_scales();
    });
    assert!(e.contains("needs scale factors for A and B"), "{e}");
}

fn one_warp_missing_scales() -> Kernel {
    Kernel::builder("bs_missing").block_dim([32, 1, 1]).build(|| {
        let mma =
            make_tiled_mma(MmaMXF8Op::new(DType::F8E4M3FN, DType::F32, DType::F8E8M0FNU), Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let a = make_fragment::<E4>((16, 1, 1));
        let b = make_fragment::<E4>((8, 1, 1));
        let c = make_fragment::<f32>((4, 1, 1));
        let _ = thr;
        gemm(&mma, &c, &a, &b, &c);
    })
}

#[test]
fn block_scaled_mmas_need_sm120() {
    use DType::*;
    let ops: Vec<MmaAtom> = vec![
        MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU).into(),
        MmaMXF4NVF4Op::new(F4E2M1FN, F32, F8E4M3FN).into(),
        MmaMXF8Op::new(F8E5M2, F32, F8E8M0FNU).into(),
        MmaMXF8F6F4Op::new(F4E2M1FN, F8E4M3FN, F32, F8E8M0FNU).into(),
    ];
    for atom in ops {
        let kernel = targeted(atom);
        for arch in [Arch::Sm120a, Arch::Sm121a, Arch::Sm120f, Arch::Sm121f] {
            cute::validate::check_target(kernel.ir(), arch).unwrap_or_else(|e| panic!("{arch:?}: {}", e.message));
        }
        for arch in [Arch::Sm100a, Arch::Sm89, Arch::Sm90a, Arch::Sm120] {
            let e = cute::validate::check_target(kernel.ir(), arch).unwrap_err().message;
            assert!(e.contains("block-scaled mma.sync"), "{arch:?}: {e}");
        }
    }
    // A kernel built for sm_100a refuses the instruction while it is built.
    let e = panic_text(|| {
        Kernel::builder("bs_sm100a")
            .block_dim([32, 1, 1])
            .target(Arch::Sm100a)
            .build(|| body_of(MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU).into()));
    });
    assert!(e.contains("sm_100a") && e.contains("block-scaled mma.sync (kind::mxf4)"), "{e}");
}

/// A minimal body issuing one MMA of `atom` on fragments loaded from global memory.
fn body<TA: Host, TB: Host, S: Numeric>(atom: MmaAtom) {
    let [_, _, ka] = atom.shape_mnk();
    let (na, nb) = if ka == 64 { (32, 16) } else { (16, 8) };
    let mma = make_tiled_mma(atom, Layout::col_major((1, 1, 1)));
    let ga = param_tensor::<u8>("a", Layout::col_major(na));
    let gb = param_tensor::<u8>("b", Layout::col_major(nb));
    let gs = param_tensor::<S>("sf", Layout::col_major(ka));
    let out = param_tensor::<f32>("out", Layout::col_major(128));
    let frag = |n: usize| (make_fragment::<TA>((n, 1, 1)), make_fragment::<TB>((n, 1, 1)));
    let (a, _) = frag(na);
    let (_, b) = frag(nb);
    fill(&a, &ga);
    fill(&b, &gb);
    let sf = make_fragment::<S>((ka, 1, 1));
    for i in 0..ka {
        sf.store(i, gs.load(i as i32));
    }
    let acc = make_fragment::<f32>((4, 1, 1));
    acc.fill(0.0f32);
    gemm(&mma, &acc, (&a, &sf), (&b, &sf), &acc);
    for i in 0..4 {
        out.store(lane_idx() * 4 + i, acc.load(i));
    }
}

/// [`body`] with the element types of `atom`.
fn body_of(atom: MmaAtom) {
    use DType::*;
    let MmaOp::Sm120(k) = atom.op() else { unreachable!() };
    match (k.a, k.b, k.sf_vec_size) {
        (F4E2M1FN, F4E2M1FN, 32) => body::<Fp4, Fp4, Ue8>(atom),
        (F4E2M1FN, F4E2M1FN, _) => body::<Fp4, Fp4, E4>(atom),
        (F4E2M1FN, F8E4M3FN, _) => body::<Fp4, E4, Ue8>(atom),
        (F8E5M2, F8E5M2, _) => body::<E5, E5, Ue8>(atom),
        _ => unreachable!(),
    }
}

fn targeted(atom: MmaAtom) -> Kernel {
    Kernel::builder("bs_ptx").block_dim([32, 1, 1]).build(move || body_of(atom))
}

fn ptx_kernels() -> Vec<(&'static str, Kernel)> {
    use DType::*;
    let k = |name: &'static str, atom: MmaAtom| {
        (name, Kernel::builder(name).block_dim([32, 1, 1]).target(Arch::Sm120a).build(move || body_of(atom)))
    };
    vec![
        k("bs_mxf4", MmaMXF4Op::new(F4E2M1FN, F32, F8E8M0FNU).into()),
        k("bs_nvf4", MmaMXF4NVF4Op::new(F4E2M1FN, F32, F8E4M3FN).into()),
        k("bs_mxf8", MmaMXF8Op::new(F8E5M2, F32, F8E8M0FNU).into()),
        k("bs_mixed", MmaMXF8F6F4Op::new(F4E2M1FN, F8E4M3FN, F32, F8E8M0FNU).into()),
    ]
}

#[test]
fn block_scaled_mlir_text() {
    for (name, kernel) in ptx_kernels() {
        let text = kernel.to_mlir_with(&MlirOptions { llvm_major: Some(22), ..Default::default() });
        assert!(text.contains("nvvm.mma.block_scale A["), "{name}: {text}");
        let inline = kernel.to_mlir_with(&MlirOptions { mma_inline_ptx: true, ..Default::default() });
        assert!(
            inline.contains(".block_scale.scale_vec::") && inline.contains("{$14}, {$15, $16}"),
            "{name}: {inline}"
        );
        assert_eq!(kernel.min_llvm_version(), Some(22), "{name}");
    }
}

#[test]
fn block_scaled_ptx_for_sm120a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let inline = tc.llvm_version().is_some_and(|v| v < 22);
    for (name, kernel) in ptx_kernels() {
        let options = CompileOptions::new().gpu_arch(Arch::Sm120a).mma_inline_ptx(inline);
        let compiled = kernel.compile_with(&tc, &options).unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        assert!(
            ptx.contains(".block_scale") && ptx.contains("mma.sync.aligned"),
            "{name}: no block-scaled mma in\n{ptx}"
        );
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-bs-{}-{name}", std::process::id()));
        let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_120a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
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

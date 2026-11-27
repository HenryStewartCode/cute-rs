// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! Warp shuffles, reductions, redux, match, barriers and short-warp kernels, with their references.

use super::*;
/// The lane a `shfl.sync` executed by lane `lane` reads, from its operand
/// `b` and its packed segment mask and clamp `c`, as the PTX ISA's
/// pseudo-code computes it; `None` when the lane keeps its own value.
pub fn shfl_ref(kind: ShuffleKind, lane: u32, b: u32, c: u32) -> Option<u32> {
    let (bval, cval, segmask) = (b & 31, c & 31, c >> 8 & 31);
    let max_lane = (lane & segmask) | (cval & !segmask);
    let min_lane = lane & segmask;
    match kind {
        ShuffleKind::Up => (lane >= bval && lane - bval >= max_lane).then(|| lane - bval),
        ShuffleKind::Down => (lane + bval <= max_lane).then(|| lane + bval),
        ShuffleKind::Bfly => (lane ^ bval <= max_lane).then_some(lane ^ bval),
        ShuffleKind::Idx => Some(min_lane | (bval & !segmask)).filter(|&j| j <= max_lane),
    }
}

/// The shuffles every thread of the shuffle kernels does over the whole
/// warp: the kind, the offset (`None`: from the data, any 32-bit value, of
/// which bits 4:0 count) and `mask_and_clamp`. The whole warp (31, and 0
/// for up), segments of 8 and 16 lanes, a clamp below the segment's top,
/// up and down past the ends (lanes keep their own value), an offset above
/// 31, up with the clamp of the other kinds (every lane keeps its own, as
/// CuTe DSL notes), and butterflies across segments, which the PTX ISA
/// allows when the lane read is at most `maxLane`.
pub const SHUFFLES: [(ShuffleKind, Option<i32>, i32); 23] = [
    (ShuffleKind::Idx, Some(3), 31),
    (ShuffleKind::Idx, None, 31),
    (ShuffleKind::Idx, Some(5), 0x1807),
    (ShuffleKind::Idx, Some(13), 0x1807),
    (ShuffleKind::Idx, Some(13), 0x100f),
    (ShuffleKind::Idx, Some(6), 0x1803),
    (ShuffleKind::Idx, None, 0x1807),
    (ShuffleKind::Up, Some(2), 0),
    (ShuffleKind::Up, Some(5), 0),
    (ShuffleKind::Up, None, 0),
    (ShuffleKind::Up, Some(3), 0x1800),
    (ShuffleKind::Up, Some(6), 0x1000),
    (ShuffleKind::Up, Some(1), 31),
    (ShuffleKind::Down, Some(5), 31),
    (ShuffleKind::Down, None, 31),
    (ShuffleKind::Down, Some(3), 0x1807),
    (ShuffleKind::Down, Some(6), 0x100f),
    (ShuffleKind::Down, Some(33), 31),
    (ShuffleKind::Bfly, Some(1), 31),
    (ShuffleKind::Bfly, Some(16), 31),
    (ShuffleKind::Bfly, None, 31),
    (ShuffleKind::Bfly, Some(4), 0x1807),
    (ShuffleKind::Bfly, Some(8), 0x1807),
];

/// The output columns of the shuffle kernels: [`SHUFFLES`], then three
/// shuffles with member masks.
const SHUFFLE_COLUMNS: usize = SHUFFLES.len() + 3;

/// Shuffles of one type: every entry of [`SHUFFLES`], then shuffles with
/// member masks (the two half-warps in divergent branches, each with its
/// own mask; per-lane half-warp masks in converged code; per-lane masks of
/// 8-lane segments with a segmented up), on the loaded values with lane 7
/// of each warp holding `special` (for the float types a NaN with a
/// payload, which must arrive unchanged; for the integers a value with the
/// sign bit set), and a Boolean butterfly.
macro_rules! shuffle_case {
    ($name:ident, $t:ty, $special:expr) => {
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                let bs = param_tensor::<i32>("bs", Layout::col_major(T));
                let out = param_tensor::<$t>("out", Layout::row_major((T, SHUFFLE_COLUMNS)));
                let bools = param_tensor::<u32>("bools", Layout::col_major(T));
                let t = thread_idx().x;
                let lane = lane_idx();
                let special: $t = $special;
                let v = select(lane.eq(7), special, xs.load(t));
                let b = bs.load(t);
                let mut results = Vec::new();
                for (kind, offset, clamp) in SHUFFLES {
                    let offset = offset.map_or(b, Int32::from);
                    results.push(shuffle_sync_op(v, offset, FULL_MASK, clamp, kind));
                }
                results.push(if_else_yield(
                    lane.lt(16),
                    || shuffle_sync_bfly_with(v, 1, 0x0000_ffffu32, 31),
                    || shuffle_sync_down_with(v, 2, 0xffff_0000u32, 31),
                ));
                let halves = select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32);
                results.push(shuffle_sync_with(v, lane ^ 5, halves, 31));
                let segment = Value::constant(0xffu32) << (lane & !7).bitcast::<u32>();
                results.push(shuffle_sync_up_with(v, 1, segment, 0x1800));
                assert_eq!(results.len(), SHUFFLE_COLUMNS);
                for (j, r) in results.into_iter().enumerate() {
                    out.store((t, j), r);
                }
                let flag = shuffle_sync_bfly((t % 3).eq(0), 3);
                bools.store(t, select(flag, 1u32, 0u32));
            });
            fn check(args: &[KernelArg]) {
                let (xs, bs) = (args[0].to_vec::<$t>(), args[1].to_vec::<i32>());
                let (out, bools) = (args[2].to_vec::<$t>(), args[3].to_vec::<u32>());
                let special: $t = $special;
                let value = |t: usize| if t % 32 == 7 { special } else { xs[t] };
                for t in 0..T {
                    let (warp, lane) = (t / 32 * 32, (t % 32) as u32);
                    let read = |source: Option<u32>| value(source.map_or(t, |s| warp + s as usize)).raw_bits();
                    let mut want: Vec<u64> = SHUFFLES
                        .iter()
                        .map(|&(kind, offset, clamp)| {
                            read(shfl_ref(kind, lane, offset.unwrap_or(bs[t]) as u32, clamp as u32))
                        })
                        .collect();
                    want.push(read(if lane < 16 {
                        shfl_ref(ShuffleKind::Bfly, lane, 1, 31)
                    } else {
                        shfl_ref(ShuffleKind::Down, lane, 2, 31)
                    }));
                    want.push(read(shfl_ref(ShuffleKind::Idx, lane, lane ^ 5, 31)));
                    want.push(read(shfl_ref(ShuffleKind::Up, lane, 1, 0x1800)));
                    let got: Vec<u64> =
                        out[t * SHUFFLE_COLUMNS..(t + 1) * SHUFFLE_COLUMNS].iter().map(|v| v.raw_bits()).collect();
                    assert_eq!(got, want, "{} thread {t}, offset {:#x}", stringify!($t), bs[t]);
                    assert_eq!(bools[t], ((warp + (lane ^ 3) as usize) % 3 == 0) as u32, "Boolean, thread {t}");
                }
            }
            Case { name: stringify!($name), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

shuffle_case!(shuffles_i8, i8, i8::MIN);
shuffle_case!(shuffles_u16, u16, u16::MAX - 1);
shuffle_case!(shuffles_f16, F16, F16(0x7e05));
shuffle_case!(shuffles_bf16, BF16, BF16(0xffc3));
shuffle_case!(shuffles_i32, i32, i32::MIN + 1);
shuffle_case!(shuffles_f32, f32, f32::from_bits(0x7fc0_0123));
shuffle_case!(shuffles_i64, i64, i64::MIN + 5);
shuffle_case!(shuffles_f64, f64, f64::from_bits(0xfff8_0000_0000_4567));

/// The group sizes of [`warp_reductions`]: every power of two up to the warp.
pub(super) const GROUPS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// `op` applied as CuTe DSL's butterfly applies it: each lane combines its
/// value with that of the lane `offset` away, for offsets from half the
/// group down to 1.
pub fn butterfly_ref<V: Copy>(values: &[V], group: usize, op: impl Fn(V, V) -> V) -> Vec<V> {
    let mut v = values.to_vec();
    let mut offset = group / 2;
    while offset > 0 {
        v = (0..v.len()).map(|l| op(v[l], v[l ^ offset])).collect();
        offset /= 2;
    }
    v
}

/// PTX's `max` on float values: a NaN is ignored, and `+0 > -0`.
pub fn max_ref(a: f64, b: f64) -> f64 {
    match (a.is_nan(), b.is_nan()) {
        (true, _) => b,
        (_, true) => a,
        _ if a == b => {
            if a.is_sign_negative() {
                b
            } else {
                a
            }
        }
        _ => a.max(b),
    }
}

/// PTX's `min` on float values: a NaN is ignored, and `-0 < +0`.
pub fn min_ref(a: f64, b: f64) -> f64 {
    -max_ref(-a, -b)
}

/// IEEE 754's `maximum` (`arith.maximumf`, CuTe DSL's `cutlass.max` on
/// floats other than Float32): a NaN operand gives NaN, here `nan`, and
/// `-0 < +0`.
pub fn maximum_ref(a: f64, b: f64, nan: f64) -> f64 {
    if a.is_nan() || b.is_nan() { nan } else { max_ref(a, b) }
}

/// The lane of each warp of [`warp_reductions`] whose Float16, BFloat16 and
/// Float64 values are NaNs (negative, with payloads) for the maxima.
pub fn reduction_nan_lane(warp: usize) -> usize {
    [3, 26][warp % 2]
}

/// `warp_reduction`, `warp_reduction_sum` and `warp_reduction_max` over
/// every group size, on Int32 (spread over every bit, so the maxima see
/// negative values and the sums wrap), Float32 (whose sums round in the
/// butterfly's order), Float16, Float64, Uint8 (8-bit values shuffled as
/// 32-bit words), Int64 (two shuffles a step) and BFloat16; and with other
/// operators: xor, min, and `2a - b`, which is not commutative, so each
/// lane's result depends on the order `op(own, partner)`; and signed zeros,
/// whose maximum is `+0` on every float type and minimum `-0`. One lane of
/// each warp holds a NaN for the Float16, BFloat16 and Float64 maxima,
/// which are IEEE 754's `maximum` as in CuTe DSL (unlike Float32's `fmax`):
/// its group's result is the quiet NaN (a group of one keeps the NaN it
/// has).
pub fn warp_reductions() -> Case {
    const N: usize = GROUPS.len();
    let kernel = Kernel::builder("warp_reductions").block_dim([T as u32, 1, 1]).build(|| {
        let xi = param_tensor::<i32>("xi", Layout::col_major(T));
        let xf = param_tensor::<f32>("xf", Layout::col_major(T));
        let xh = param_tensor::<F16>("xh", Layout::col_major(T));
        let xd = param_tensor::<f64>("xd", Layout::col_major(T));
        let xb = param_tensor::<u8>("xb", Layout::col_major(T));
        let xl = param_tensor::<i64>("xl", Layout::col_major(T));
        let oi = param_tensor::<i32>("oi", Layout::row_major((T, 2 * N + 3)));
        let of = param_tensor::<f32>("of", Layout::row_major((T, 2 * N + 2)));
        let oh = param_tensor::<F16>("oh", Layout::row_major((T, 2 * N + 1)));
        let od = param_tensor::<f64>("od", Layout::row_major((T, 2 * N + 1)));
        let ob = param_tensor::<u8>("ob", Layout::row_major((T, 2 * N)));
        let ol = param_tensor::<i64>("ol", Layout::row_major((T, 2 * N)));
        let xg = param_tensor::<BF16>("xg", Layout::col_major(T));
        let og = param_tensor::<BF16>("og", Layout::row_major((T, 2 * N + 1)));
        let t = thread_idx().x;
        let i = xi.load(t) * SPREAD as i32;
        let (f, h, d, b, l) = (xf.load(t), xh.load(t), xd.load(t), xb.load(t), xl.load(t));
        let bf = xg.load(t);
        let nan_lane = lane_idx().eq(select(t.lt(32), reduction_nan_lane(0) as i32, reduction_nan_lane(1) as i32));
        let hn = select(nan_lane, F16(0xfe05), h);
        let dn = select(nan_lane, f64::from_bits(0xfff8_0000_0000_4567), d);
        let bn = select(nan_lane, BF16(0xffc3), bf);
        for (k, &g) in GROUPS.iter().enumerate() {
            oi.store((t, 2 * k), warp_reduction_sum(i, g));
            oi.store((t, 2 * k + 1), warp_reduction_max(i, g));
            of.store((t, 2 * k), warp_reduction_sum(f, g));
            of.store((t, 2 * k + 1), warp_reduction_max(f, g));
            oh.store((t, 2 * k), warp_reduction_sum(h, g));
            oh.store((t, 2 * k + 1), warp_reduction_max(hn, g));
            od.store((t, 2 * k), warp_reduction_sum(d, g));
            od.store((t, 2 * k + 1), warp_reduction_max(dn, g));
            ob.store((t, 2 * k), warp_reduction_sum(b, g));
            ob.store((t, 2 * k + 1), warp_reduction_max(b, g));
            ol.store((t, 2 * k), warp_reduction_sum(l, g));
            ol.store((t, 2 * k + 1), warp_reduction_max(l, g));
            og.store((t, 2 * k), warp_reduction_sum(bf, g));
            og.store((t, 2 * k + 1), warp_reduction_max(bn, g));
        }
        oi.store((t, 2 * N), warp_reduction(i, |a, c| a ^ c, 8));
        oi.store((t, 2 * N + 1), warp_reduction(i, |a, c| a.min(c), 16));
        oi.store((t, 2 * N + 2), warp_reduction(i, |a, c| a * 2 - c, 4));
        let zero = select((lane_idx() & 1).eq(1), -0.0f32, 0.0f32);
        of.store((t, 2 * N), warp_reduction_max(zero, 32));
        of.store((t, 2 * N + 1), warp_reduction(zero, |a, c| a.min(c), 32));
        // IEEE 754 maximum orders -0 below +0 too.
        let negative = (lane_idx() & 1).eq(1);
        oh.store((t, 2 * N), warp_reduction_max(select(negative, F16(0x8000), F16(0)), 32));
        od.store((t, 2 * N), warp_reduction_max(select(negative, -0.0f64, 0.0f64), 32));
        og.store((t, 2 * N), warp_reduction_max(select(negative, BF16(0x8000), BF16(0)), 32));
    });
    fn check(args: &[KernelArg]) {
        let xi: Vec<i32> = args[0].to_vec::<i32>().iter().map(|v| v.wrapping_mul(SPREAD as i32)).collect();
        let (xf, xh, xd) = (args[1].to_vec::<f32>(), args[2].to_vec::<F16>(), args[3].to_vec::<f64>());
        let (xb, xl) = (args[4].to_vec::<u8>(), args[5].to_vec::<i64>());
        let (oi, of, oh) = (args[6].to_vec::<i32>(), args[7].to_vec::<f32>(), args[8].to_vec::<F16>());
        let (od, ob, ol) = (args[9].to_vec::<f64>(), args[10].to_vec::<u8>(), args[11].to_vec::<i64>());
        let (xg, og) = (args[12].to_vec::<BF16>(), args[13].to_vec::<BF16>());
        const N: usize = GROUPS.len();
        let half_add = |a: F16, b: F16| F16::from_f32(a.to_f32() + b.to_f32());
        let bf_add = |a: BF16, b: BF16| BF16::from_f32(a.to_f32() + b.to_f32());
        // IEEE 754 maximum; the NaN is the quiet NaN of each type.
        let half_max = |a: F16, b: F16| {
            let (x, y) = (a.to_f32() as f64, b.to_f32() as f64);
            if x.is_nan() || y.is_nan() { F16(0x7e00) } else { F16::from_f32(max_ref(x, y) as f32) }
        };
        let bf_max = |a: BF16, b: BF16| {
            let (x, y) = (a.to_f32() as f64, b.to_f32() as f64);
            if x.is_nan() || y.is_nan() { BF16(0x7fc0) } else { BF16::from_f32(max_ref(x, y) as f32) }
        };
        let double_max = |a: f64, b: f64| maximum_ref(a, b, f64::from_bits(0x7ff8_0000_0000_0000));
        /// The warp's 32 values, with `nan` in the NaN lane.
        fn with_nan<V: Copy>(values: &[V], w: usize, nan: V) -> Vec<V> {
            (0..32).map(|l| if l == reduction_nan_lane(w) { nan } else { values[w * 32 + l] }).collect()
        }
        for w in 0..T / 32 {
            let lanes = w * 32..(w + 1) * 32;
            let nan_lane = reduction_nan_lane(w);
            let hn = with_nan(&xh, w, F16(0xfe05));
            let dn = with_nan(&xd, w, f64::from_bits(0xfff8_0000_0000_4567));
            let bn = with_nan(&xg, w, BF16(0xffc3));
            for (k, &g) in GROUPS.iter().enumerate() {
                let i_sum = butterfly_ref(&xi[lanes.clone()], g, i32::wrapping_add);
                let i_max = butterfly_ref(&xi[lanes.clone()], g, i32::max);
                let f_sum = butterfly_ref(&xf[lanes.clone()], g, |a, b| a + b);
                let f_max = butterfly_ref(&xf[lanes.clone()], g, |a, b| max_ref(a as f64, b as f64) as f32);
                let h_sum = butterfly_ref(&xh[lanes.clone()], g, half_add);
                let h_max = butterfly_ref(&hn, g, half_max);
                let d_sum = butterfly_ref(&xd[lanes.clone()], g, |a, b| a + b);
                let d_max = butterfly_ref(&dn, g, double_max);
                let b_sum = butterfly_ref(&xb[lanes.clone()], g, u8::wrapping_add);
                let b_max = butterfly_ref(&xb[lanes.clone()], g, u8::max);
                let l_sum = butterfly_ref(&xl[lanes.clone()], g, i64::wrapping_add);
                let l_max = butterfly_ref(&xl[lanes.clone()], g, i64::max);
                let g_sum = butterfly_ref(&xg[lanes.clone()], g, bf_add);
                let g_max = butterfly_ref(&bn, g, bf_max);
                // The reference's maxima are NaN across the NaN lane's group:
                // the quiet NaN, unless the group is the lane alone.
                for l in nan_lane / g * g..(nan_lane / g + 1) * g {
                    let alone = g == 1;
                    assert_eq!(h_max[l].0, if alone { 0xfe05 } else { 0x7e00 });
                    assert_eq!(d_max[l].to_bits(), if alone { 0xfff8_0000_0000_4567 } else { 0x7ff8 << 48 });
                    assert_eq!(g_max[l].0, if alone { 0xffc3 } else { 0x7fc0 });
                }
                for l in 0..32 {
                    let t = w * 32 + l;
                    let what = format!("group {g}, thread {t}");
                    assert_eq!(
                        [oi[t * (2 * N + 3) + 2 * k], oi[t * (2 * N + 3) + 2 * k + 1]],
                        [i_sum[l], i_max[l]],
                        "Int32 {what}"
                    );
                    let f = [of[t * (2 * N + 2) + 2 * k].to_bits(), of[t * (2 * N + 2) + 2 * k + 1].to_bits()];
                    assert_eq!(f, [f_sum[l].to_bits(), f_max[l].to_bits()], "Float32 {what}");
                    assert_eq!(
                        [oh[t * (2 * N + 1) + 2 * k].0, oh[t * (2 * N + 1) + 2 * k + 1].0],
                        [h_sum[l].0, h_max[l].0],
                        "Float16 {what}"
                    );
                    let d = [od[t * (2 * N + 1) + 2 * k].to_bits(), od[t * (2 * N + 1) + 2 * k + 1].to_bits()];
                    assert_eq!(d, [d_sum[l].to_bits(), d_max[l].to_bits()], "Float64 {what}");
                    assert_eq!(
                        [ob[t * 2 * N + 2 * k], ob[t * 2 * N + 2 * k + 1]],
                        [b_sum[l], b_max[l]],
                        "Uint8 {what}"
                    );
                    assert_eq!(
                        [ol[t * 2 * N + 2 * k], ol[t * 2 * N + 2 * k + 1]],
                        [l_sum[l], l_max[l]],
                        "Int64 {what}"
                    );
                    assert_eq!(
                        [og[t * (2 * N + 1) + 2 * k].0, og[t * (2 * N + 1) + 2 * k + 1].0],
                        [g_sum[l].0, g_max[l].0],
                        "BFloat16 {what}"
                    );
                }
            }
            let xor = butterfly_ref(&xi[lanes.clone()], 8, |a, b| a ^ b);
            let min = butterfly_ref(&xi[lanes.clone()], 16, i32::min);
            let skew = butterfly_ref(&xi[lanes.clone()], 4, |a: i32, b: i32| a.wrapping_mul(2).wrapping_sub(b));
            for l in 0..32 {
                let t = w * 32 + l;
                assert_eq!(oi[t * (2 * N + 3) + 2 * N..(t + 1) * (2 * N + 3)], [xor[l], min[l], skew[l]], "thread {t}");
                let zeros = [of[t * (2 * N + 2) + 2 * N].to_bits(), of[t * (2 * N + 2) + 2 * N + 1].to_bits()];
                assert_eq!(zeros, [0, 0x8000_0000], "signed zeros, thread {t}");
                let other = [
                    oh[t * (2 * N + 1) + 2 * N].0 as u64,
                    od[t * (2 * N + 1) + 2 * N].to_bits(),
                    og[t * (2 * N + 1) + 2 * N].0 as u64,
                ];
                assert_eq!(other, [0, 0, 0], "signed zeros of Float16, Float64 and BFloat16, thread {t}");
            }
        }
    }
    Case { name: "warp_reductions", kernel, cfg: launch(1), small_only: false, check }
}

/// The integer kinds of `redux.sync`.
const REDUX_INT_KINDS: [ReduxKind; 8] = [
    ReduxKind::Add,
    ReduxKind::And,
    ReduxKind::Or,
    ReduxKind::Xor,
    ReduxKind::Max,
    ReduxKind::Min,
    ReduxKind::Umax,
    ReduxKind::Umin,
];

/// `redux.sync` over 32-bit integers, from the PTX ISA's definitions: the
/// sum wraps, `max` and `min` compare signed unless the value is unsigned
/// (CuTe DSL takes them as `umax` and `umin` on a Uint32).
pub fn redux_int_ref(kind: ReduxKind, unsigned: bool, values: &[u32]) -> u32 {
    let signed = || values.iter().map(|&v| v as i32);
    match (kind, unsigned) {
        (ReduxKind::Add, _) => values.iter().fold(0u32, |a, &v| a.wrapping_add(v)),
        (ReduxKind::And, _) => values.iter().fold(u32::MAX, |a, &v| a & v),
        (ReduxKind::Or, _) => values.iter().fold(0, |a, &v| a | v),
        (ReduxKind::Xor, _) => values.iter().fold(0, |a, &v| a ^ v),
        (ReduxKind::Max, false) => signed().max().unwrap() as u32,
        (ReduxKind::Min, false) => signed().min().unwrap() as u32,
        (ReduxKind::Max | ReduxKind::Umax, _) => *values.iter().max().unwrap(),
        (ReduxKind::Min | ReduxKind::Umin, _) => *values.iter().min().unwrap(),
        (other, _) => panic!("{other:?} is not an integer kind"),
    }
}

/// Every integer `redux.sync` kind on Int32 and Uint32 data spread over
/// every bit (so signed and unsigned maxima differ and sums wrap), over the
/// whole warp; then with member masks: the two half-warps in divergent
/// branches, per-lane masks of interleaved groups of four lanes in
/// converged code, and only the lanes of a mask executing it.
pub fn redux_int() -> Case {
    const K: usize = REDUX_INT_KINDS.len();
    let thirds = (0..32).filter(|l| l % 3 != 0).fold(0u32, |m, l| m | 1 << l);
    let kernel = Kernel::builder("redux_int").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let us = param_tensor::<u32>("us", Layout::col_major(T));
        let oi = param_tensor::<i32>("oi", Layout::row_major((T, K + 2)));
        let ou = param_tensor::<u32>("ou", Layout::row_major((T, K + 2)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let (x, u) = (xs.load(t) * SPREAD as i32, us.load(t) * SPREAD as u32);
        for (k, kind) in REDUX_INT_KINDS.into_iter().enumerate() {
            oi.store((t, k), warp_redux_sync(x, kind));
            ou.store((t, k), warp_redux_sync(u, kind));
        }
        let plain = ReduxOptions::default();
        let halves = if_else_yield(
            lane.lt(16),
            || warp_redux_sync_with(x, ReduxKind::Add, 0x0000_ffffu32, plain),
            || warp_redux_sync_with(x, ReduxKind::Max, 0xffff_0000u32, plain),
        );
        oi.store((t, K), halves);
        let quads = select((lane & 4).eq(0), 0x0f0f_0f0fu32, 0xf0f0_f0f0u32);
        oi.store((t, K + 1), warp_redux_sync_with(x, ReduxKind::Min, quads, plain));
        ou.store((t, K), warp_redux_sync_with(u, ReduxKind::Max, quads, plain));
        let partial = if_else_yield((lane % 3).ne(0), || warp_redux_sync_with(u, ReduxKind::Xor, thirds, plain), || u);
        ou.store((t, K + 1), partial);
    });
    fn check(args: &[KernelArg]) {
        const K: usize = REDUX_INT_KINDS.len();
        let xs: Vec<u32> = args[0].to_vec::<i32>().iter().map(|v| v.wrapping_mul(SPREAD as i32) as u32).collect();
        let us: Vec<u32> = args[1].to_vec::<u32>().iter().map(|v| v.wrapping_mul(SPREAD as u32)).collect();
        let (oi, ou) = (args[2].to_vec::<i32>(), args[3].to_vec::<u32>());
        let thirds = (0..32).filter(|l| l % 3 != 0).fold(0u32, |m, l| m | 1 << l);
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            let over = |data: &[u32], mask: u32| -> Vec<u32> {
                (0..32).filter(|l| mask >> l & 1 == 1).map(|l| data[warp + l]).collect()
            };
            let mut want_i: Vec<u32> =
                REDUX_INT_KINDS.iter().map(|&k| redux_int_ref(k, false, &over(&xs, !0))).collect();
            let mut want_u: Vec<u32> =
                REDUX_INT_KINDS.iter().map(|&k| redux_int_ref(k, true, &over(&us, !0))).collect();
            want_i.push(if lane < 16 {
                redux_int_ref(ReduxKind::Add, false, &over(&xs, 0x0000_ffff))
            } else {
                redux_int_ref(ReduxKind::Max, false, &over(&xs, 0xffff_0000))
            });
            let quads = if lane & 4 == 0 { 0x0f0f_0f0f } else { 0xf0f0_f0f0 };
            want_i.push(redux_int_ref(ReduxKind::Min, false, &over(&xs, quads)));
            want_u.push(redux_int_ref(ReduxKind::Umax, true, &over(&us, quads)));
            want_u.push(if lane % 3 != 0 { redux_int_ref(ReduxKind::Xor, true, &over(&us, thirds)) } else { us[t] });
            let got_i: Vec<u32> = oi[t * (K + 2)..(t + 1) * (K + 2)].iter().map(|&v| v as u32).collect();
            assert_eq!(got_i, want_i, "Int32, thread {t}");
            assert_eq!(ou[t * (K + 2)..(t + 1) * (K + 2)], want_u[..], "Uint32, thread {t}");
        }
    }
    Case { name: "redux_int", kernel, cfg: launch(1), small_only: false, check }
}

/// `match_sync` on 32- and 64-bit values: `any` on small values (so lanes
/// share them), on negative Int32 values, on Uint64 values that differ in
/// the high word only, and on Int64; `all` over the whole warp (with
/// different values: 0), over each half-warp with per-lane masks (each half
/// agrees), with one lane of a half disagreeing, on a warp-uniform Uint64
/// and on one whose high word differs in one lane; and both in divergent
/// branches, each with the mask of its own lanes.
pub fn warp_match() -> Case {
    const COLS: usize = 11;
    let kernel = Kernel::builder("warp_match").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let x = xs.load(t);
        let l = ls.load(t) * SPREAD;
        let half = (lane / 16).bitcast::<u32>();
        let halves = select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32);
        let uniform = ls.load(t - lane) * SPREAD;
        let results: [Uint32; COLS] = [
            match_sync(FULL_MASK, x & 3u32, MatchKind::Any),
            match_sync(FULL_MASK, (x & 7u32).bitcast::<i32>() - 4, MatchKind::Any),
            match_sync(FULL_MASK, half, MatchKind::All),
            match_sync(halves, half, MatchKind::All),
            match_sync(halves, select(lane.eq(20), 7u32, half), MatchKind::All),
            match_sync(FULL_MASK, ((lane % 4).to::<u64>() << 40u64) | 5u64, MatchKind::Any),
            match_sync(FULL_MASK, uniform, MatchKind::All),
            match_sync(FULL_MASK, uniform ^ select(lane.eq(9), 1u64 << 50, 0u64), MatchKind::All),
            match_sync(FULL_MASK, l & 0x3_0000_0000u64, MatchKind::Any),
            match_sync(FULL_MASK, (l % 3u64).bitcast::<i64>() - 1, MatchKind::Any),
            if_else_yield(
                (lane % 2).eq(0),
                || match_sync(0x5555_5555u32, x & 3u32, MatchKind::Any),
                || match_sync(0xaaaa_aaaau32, l & 1u64, MatchKind::All),
            ),
        ];
        for (j, r) in results.into_iter().enumerate() {
            out.store((t, j), r);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[2].to_vec::<u32>());
        let ls: Vec<u64> = args[1].to_vec::<u64>().iter().map(|v| v.wrapping_mul(SPREAD)).collect();
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            // `any`: the lanes of `mask` whose value equals this lane's;
            // `all`: `mask` if they all hold one value, else 0.
            let any = |mask: u32, v: &dyn Fn(usize) -> u64| -> u32 {
                (0..32).filter(|&o| mask >> o & 1 == 1 && v(o) == v(lane)).fold(0, |m, o| m | 1 << o)
            };
            let all = |mask: u32, v: &dyn Fn(usize) -> u64| -> u32 {
                let members: Vec<usize> = (0..32).filter(|&o| mask >> o & 1 == 1).collect();
                if members.iter().all(|&o| v(o) == v(members[0])) { mask } else { 0 }
            };
            let x = |o: usize| xs[warp + o] as u64;
            let l = |o: usize| ls[warp + o];
            let half = |o: usize| (o / 16) as u64;
            let halves = if lane < 16 { 0x0000_ffff } else { 0xffff_0000 };
            let uniform = ls[warp];
            let want = [
                any(!0, &|o| x(o) & 3),
                any(!0, &|o| ((x(o) & 7) as i32 - 4) as u32 as u64),
                all(!0, &half),
                all(halves, &half),
                all(halves, &|o| if o == 20 { 7 } else { half(o) }),
                any(!0, &|o| ((o % 4) as u64) << 40 | 5),
                all(!0, &|_| uniform),
                all(!0, &|o| uniform ^ if o == 9 { 1 << 50 } else { 0 }),
                any(!0, &|o| l(o) & 0x3_0000_0000),
                any(!0, &|o| ((l(o) % 3) as i64 - 1) as u64),
                if lane % 2 == 0 { any(0x5555_5555, &|o| x(o) & 3) } else { all(0xaaaa_aaaa, &|o| l(o) & 1) },
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t}");
        }
    }
    Case { name: "warp_match", kernel, cfg: launch(1), small_only: false, check }
}

/// `sync_warp_with` and half-warp masks: each half-warp synchronises
/// separately in a divergent branch and reads what its neighbours stored
/// to shared memory; both halves synchronise in converged code with
/// per-lane masks; and the whole warp synchronises before reading the other
/// half's values.
pub fn warp_barriers() -> Case {
    let kernel = Kernel::builder("warp_barriers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 3)));
        let smem = make_smem_tensor::<u32>(Layout::col_major(T));
        let t = thread_idx().x;
        let lane = lane_idx();
        smem.store(t, xs.load(t));
        let first = if_else_yield(
            lane.lt(16),
            || {
                sync_warp_with(0x0000_ffffu32);
                smem.load(t ^ 1)
            },
            || {
                sync_warp_with(0xffff_0000u32);
                smem.load(t - lane + 16 + (lane - 13) % 16)
            },
        );
        out.store((t, 0), first);
        sync_warp_with(select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32));
        out.store((t, 1), smem.load(t ^ 7));
        sync_warp();
        out.store((t, 2), smem.load(t ^ 16));
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            let first = if lane < 16 { xs[t ^ 1] } else { xs[warp + 16 + (lane - 13) % 16] };
            assert_eq!(out[t * 3..t * 3 + 3], [first, xs[t ^ 7], xs[t ^ 16]], "thread {t}");
        }
    }
    Case { name: "warp_barriers", kernel, cfg: launch(1), small_only: false, check }
}

/// Float32 bit patterns `fmax` and `fmin` meet in every pair: signed zeros
/// and infinities; NaNs (canonical, quiet, negative with a payload,
/// signaling); subnormals (the smallest, a negative one, the largest); the
/// smallest normal numbers; and ordinary and extreme numbers.
pub const MINMAX_VALUES: [u32; 19] = [
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fff_ffff,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x0000_0001,
    0x8000_0003,
    0x007f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3fc0_0000,
    0xc020_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
];

/// Every combination of `fmax`'s and `fmin`'s `abs`, `nan` and `ftz`.
pub fn minmax_options() -> [MinMaxOptions; 8] {
    std::array::from_fn(|k| MinMaxOptions { abs: k & 1 != 0, nan: k & 2 != 0, ftz: k & 4 != 0 })
}

/// `max{.ftz}{.NaN}{.xorsign.abs}.f32` (or `min`) on raw bits, from the PTX
/// ISA's pseudo-code: `ftz` flushes subnormal operands (and results) to
/// zeros of their signs; with `abs` the sign of the result is the
/// exclusive or of the operands' signs, taken before their absolute values
/// are compared, unless the result is NaN; NaN results are the canonical
/// NaN.
pub fn minmax_ref(is_max: bool, a: u32, b: u32, opts: MinMaxOptions) -> u32 {
    let flush = |v: u32| if opts.ftz && v & 0x7f80_0000 == 0 { v & 0x8000_0000 } else { v };
    let is_nan = |v: u32| v & 0x7fff_ffff > 0x7f80_0000;
    let (a, b) = (flush(a), flush(b));
    let sign = (a ^ b) & 0x8000_0000;
    let (a, b) = if opts.abs { (a & 0x7fff_ffff, b & 0x7fff_ffff) } else { (a, b) };
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    let d = match (is_nan(a), is_nan(b)) {
        (true, true) => 0x7fff_ffff,
        (true, false) | (false, true) if opts.nan => 0x7fff_ffff,
        (true, false) => b,
        (false, true) => a,
        // Equal values differ only as signed zeros.
        _ if x == y => {
            if (a >> 31 == 0) == is_max {
                a
            } else {
                b
            }
        }
        _ => {
            if (x > y) == is_max {
                a
            } else {
                b
            }
        }
    };
    if is_nan(d) {
        0x7fff_ffff
    } else if opts.abs {
        flush(d & 0x7fff_ffff | sign)
    } else {
        flush(d)
    }
}

/// Rounds of pairs per thread in [`fmax_fmin`]: 6 * 64 threads cover the 361
/// pairs of [`MINMAX_VALUES`].
pub(super) const MINMAX_ROUNDS: usize = 6;

/// Float32 values of the first four groups of four lanes of [`redux_f32`]:
/// signed zeros; NaNs among numbers; NaNs only (with payloads, negative,
/// signaling); infinities and subnormals.
pub const REDUX_F32_SPECIALS: [u32; 16] = [
    0x8000_0000,
    0x0000_0000,
    0x8000_0000,
    0x0000_0000,
    0x7fc0_0000,
    0x3fc0_0000,
    0xc020_0000,
    0xffc0_0001,
    0x7fc0_0000,
    0xffc0_0005,
    0x7f80_0001,
    0x7fff_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0001,
    0x8000_0003,
];

/// `redux.sync.{max,min}{.abs}{.NaN}.f32` over raw bits, from the PTX
/// ISA's definition: `.abs` reduces the absolute values; `+0 > -0`; a NaN
/// input is ignored (all NaN: the canonical NaN) unless `.NaN` makes the
/// result the canonical NaN.
pub fn redux_f32_ref(is_max: bool, abs: bool, nan: bool, values: &[u32]) -> u32 {
    let is_nan = |v: u32| v & 0x7fff_ffff > 0x7f80_0000;
    let values: Vec<u32> = values.iter().map(|&v| if abs { v & 0x7fff_ffff } else { v }).collect();
    if nan && values.iter().any(|&v| is_nan(v)) {
        return 0x7fff_ffff;
    }
    let opts = MinMaxOptions::default();
    values.into_iter().filter(|&v| !is_nan(v)).reduce(|a, b| minmax_ref(is_max, a, b, opts)).unwrap_or(0x7fff_ffff)
}

/// Shuffles of every kind whose operands `b` and `c` (`mask_and_clamp`)
/// both come from the data, different in every lane: on random data, any
/// segment mask and clamp, held to the PTX ISA's pseudo-code.
pub fn shuffle_operands() -> Case {
    const KINDS: [ShuffleKind; 4] = [ShuffleKind::Idx, ShuffleKind::Up, ShuffleKind::Down, ShuffleKind::Bfly];
    let kernel = Kernel::builder("shuffle_operands").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let bs = param_tensor::<i32>("bs", Layout::col_major(T));
        let cs = param_tensor::<i32>("cs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, KINDS.len())));
        let t = thread_idx().x;
        let (x, b, c) = (xs.load(t), bs.load(t), cs.load(t));
        for (j, kind) in KINDS.into_iter().enumerate() {
            out.store((t, j), shuffle_sync_op(x, b, FULL_MASK, c, kind));
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, bs, cs) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, (t % 32) as u32);
            let want = KINDS
                .map(|kind| shfl_ref(kind, lane, bs[t] as u32, cs[t] as u32).map_or(xs[t], |s| xs[warp + s as usize]));
            assert_eq!(out[t * 4..t * 4 + 4], want, "thread {t}: b = {:#x}, c = {:#x}", bs[t], cs[t]);
        }
    }
    Case { name: "shuffle_operands", kernel, cfg: launch(1), small_only: false, check }
}

/// Warp collectives over a random partition of each warp: every lane's
/// member mask is the set of lanes with the same group (2 bits of the data),
/// found with `match_sync(.., MatchKind::Any)`. Each group then reduces with
/// every integer `redux.sync` kind, compares with `match.all`, and
/// broadcasts its lowest lane's value with a masked shuffle.
pub fn warp_partitions() -> Case {
    const COLS: usize = REDUX_INT_KINDS.len() + 3;
    let kernel = Kernel::builder("warp_partitions").block_dim([T as u32, 1, 1]).build(|| {
        let gs = param_tensor::<u32>("gs", Layout::col_major(T));
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let group = gs.load(t) >> 3u32 & 3u32;
        let x = xs.load(t) * SPREAD as u32;
        let mask = match_sync(FULL_MASK, group, MatchKind::Any);
        let plain = ReduxOptions::default();
        let mut results: Vec<Uint32> = REDUX_INT_KINDS
            .into_iter()
            .map(|kind| warp_redux_sync_with(x.bitcast::<i32>(), kind, mask, plain).bitcast())
            .collect();
        results.push(mask);
        results.push(match_sync(mask, x & 1u32, MatchKind::All));
        results.push(shuffle_sync_with(x, mask.ctz().bitcast::<i32>(), mask, 31));
        for (j, r) in results.into_iter().enumerate() {
            out.store((t, j), r);
        }
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = REDUX_INT_KINDS.len() + 3;
        let groups: Vec<u32> = args[0].to_vec::<u32>().iter().map(|g| g >> 3 & 3).collect();
        let xs: Vec<u32> = args[1].to_vec::<u32>().iter().map(|v| v.wrapping_mul(SPREAD as u32)).collect();
        let out = args[2].to_vec::<u32>();
        let sizes: std::collections::BTreeSet<u32> =
            (0..T).map(|t| (0..32).filter(|l| groups[t / 32 * 32 + l] == groups[t]).count() as u32).collect();
        assert!(sizes.len() > 1, "the partitions should have groups of several sizes: {sizes:?}");
        for t in 0..T {
            let warp = t / 32 * 32;
            let mask = (0..32).filter(|&l| groups[warp + l] == groups[t]).fold(0u32, |m, l| m | 1 << l);
            let members: Vec<u32> = (0..32).filter(|l| mask >> l & 1 == 1).map(|l| xs[warp + l]).collect();
            let mut want: Vec<u32> = REDUX_INT_KINDS.iter().map(|&k| redux_int_ref(k, false, &members)).collect();
            want.push(mask);
            want.push(if members.iter().all(|v| v & 1 == members[0] & 1) { mask } else { 0 });
            want.push(xs[warp + mask.trailing_zeros() as usize]);
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}, mask {mask:#010x}");
        }
    }
    Case { name: "warp_partitions", kernel, cfg: launch(1), small_only: false, check }
}

/// The warp-collective kernels added with full shuffles: shuffles of every
/// kind on eight types and with operands from the data, butterfly
/// reductions, integer `redux.sync`, `match_sync`, masked `sync_warp`,
/// random partitions, and the collectives in short warps. They run on every
/// target, and are in [`all`].
pub fn warp_collectives() -> Vec<Case> {
    vec![
        shuffles_i8(),
        shuffles_u16(),
        shuffles_f16(),
        shuffles_bf16(),
        shuffles_i32(),
        shuffles_f32(),
        shuffles_i64(),
        shuffles_f64(),
        shuffle_operands(),
        warp_reductions(),
        redux_int(),
        warp_match(),
        warp_barriers(),
        warp_partitions(),
        short_warps(),
    ]
}

/// Kernels that need sm_86 or newer (`max.xorsign.abs`), which the SM89 GPU
/// runs: simulated, compiled for sm_86, and run on GPUs of sm_86 or newer.
pub fn sm86() -> Vec<Case> {
    vec![fmax_fmin()]
}

/// Kernels that need sm_100a (or the sm_100f family): simulated, compiled
/// for sm_100a with `mlir-opt`, and assembled by `tests/ptxas.rs`.
pub fn sm100a() -> Vec<Case> {
    let mut cases =
        vec![redux_f32(), ue8m0_pairs(), mx_conversions(), ldmatrix_b8_forms(), ssa_fp4_fp6(), stmatrix_b8_forms()];
    // Gather4, scatter4, byte-masked bulk stores, FP4/FP6 TMA formats.
    cases.extend(tma_sm100_cases());
    cases
}

/// Blackwell kernels with tensor memory: simulated, compiled
/// for sm_100a with `mlir-opt` and assembled by `tests/ptxas.rs`; `tests/gpu.rs` skips them
/// (no GPU here runs sm_100).
pub fn sm100() -> Vec<Case> {
    let mut cases = tmem_cases();
    // TMEM loads and stores.
    cases.extend(tmem_ldst_cases());
    cases
}

/// Kernels that need sm_90a (setmaxnreg; sm_100a and sm_120a have it too): simulated,
/// compiled for sm_90a, and assembled by `tests/ptxas.rs` and `tests/gpu.rs`.
pub fn sm90a() -> Vec<Case> {
    let mut cases = vec![setmaxnreg_warp_specialized()];
    // Warpgroup MMAs: a few here (`tests/wgmma.rs` runs
    // them all).
    cases.extend(wgmma_cases_sampled());
    // The FP8, integer and register-A variants (`tests/wgmma_variants.rs`
    // runs them all).
    cases.extend(wgmma_variant_cases_sampled());
    // In-kernel tensor-map updates.
    cases.extend(tensormap_cases_sm90a());
    cases
}

/// Kernels that need sm_120a (or the sm_120f family): simulated, compiled
/// for sm_120a with `mlir-opt`, and assembled by `tests/ptxas.rs`.
pub fn sm120a() -> Vec<Case> {
    vec![mx_conversions(), ldmatrix_b8_forms(), stmatrix_b8_forms()]
}

/// Kernels that need sm_89 or newer (the FP8 conversions), which the SM89
/// GPU runs: simulated, compiled for sm_89, and run on GPUs of sm_89 or
/// newer.
pub fn sm89() -> Vec<Case> {
    let mut cases = vec![fp8_conversions(), fp8_f16_sweep(), ssa_fp8()];
    cases.extend(mma_fp8());
    cases.push(ldmatrix_8x16x8b_fp8());
    cases.extend(pipeline_cases_sm89());
    cases.extend(async_sched_cases_sm89());
    cases.extend(cpasync_order_cases_sm89());
    cases.extend(gemm_helper_cases_sm89());
    cases
}

/// Threads per block of [`short_warps`]: a whole warp, then a warp of 16
/// lanes.
pub const SHORT_BLOCK: usize = 48;

/// The columns of [`short_warps`].
const SHORT_COLUMNS: usize = 18;

/// Warp collectives with the full member mask in blocks of [`SHORT_BLOCK`]
/// threads, whose second warp has 16 lanes. The lanes such a warp lacks
/// count as exited, as the PTX ISA says of `shfl.sync`, `vote.sync`,
/// `match.sync` and `redux.sync`: `sync_warp` and shuffles that read lanes
/// the warp has work; `match.all` returns `0x0000ffff`; `redux.sync`,
/// `warp_reduction` over groups of 16 and the votes take the 16 lanes
/// (`vote_all_sync` of "lane < 16" holds); ballots, `match.any` and
/// `activemask` have 16 bits. Two blocks, so that reading past a short warp
/// would read the next block's threads.
pub fn short_warps() -> Case {
    let kernel = Kernel::builder("short_warps").block_dim([SHORT_BLOCK as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(2 * SHORT_BLOCK));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * SHORT_BLOCK, SHORT_COLUMNS)));
        let smem = make_smem_tensor::<u32>(Layout::col_major(SHORT_BLOCK));
        let t = thread_idx().x;
        let g = block_idx().x * SHORT_BLOCK as i32 + t;
        let lane = lane_idx();
        let x = xs.load(g) * SPREAD as u32;
        smem.store(t, x);
        sync_warp();
        let low = lane.lt(16);
        let results: [Uint32; SHORT_COLUMNS] = [
            smem.load(t ^ 1),
            shuffle_sync(x, 5),
            shuffle_sync_bfly(x, 4),
            shuffle_sync_up(x, 3),
            shuffle_sync_down_with(x, 3, FULL_MASK, 0x100f),
            match_sync(FULL_MASK, x & 3u32, MatchKind::Any),
            match_sync(FULL_MASK, warp_idx(), MatchKind::All),
            match_sync(FULL_MASK, x & 1u32, MatchKind::All),
            warp_redux_sync(x, ReduxKind::Add),
            warp_redux_sync(x, ReduxKind::Max),
            warp_redux_sync(x.bitcast::<i32>(), ReduxKind::Min).bitcast(),
            warp_reduction_sum(x, 16),
            vote_ballot_sync((x & 1u32).eq(1u32)),
            vote_all_sync(low).to(),
            vote_any_sync(lane.ge(16)).to(),
            vote_uni_sync(low).to(),
            vote_ballot_sync(elect_one()),
            activemask(),
        ];
        for (j, r) in results.into_iter().enumerate() {
            out.store((g, j), r);
        }
    });
    fn check(args: &[KernelArg]) {
        let xs: Vec<u32> = args[0].to_vec::<u32>().iter().map(|v| v.wrapping_mul(SPREAD as u32)).collect();
        let out = args[1].to_vec::<u32>();
        for g in 0..2 * SHORT_BLOCK {
            let (block, t) = (g / SHORT_BLOCK, g % SHORT_BLOCK);
            let (warp, lane) = (t / 32, t % 32);
            // The warp's lanes: 32, or 16 for the second warp of a block.
            let size = (SHORT_BLOCK - warp * 32).min(32);
            let x = |l: usize| {
                assert!(l < size, "the reference reads lane {l} of a warp of {size}");
                xs[block * SHORT_BLOCK + warp * 32 + l]
            };
            let mask_of = |f: &dyn Fn(usize) -> bool| (0..size).filter(|&l| f(l)).fold(0u32, |m, l| m | 1 << l);
            let members = mask_of(&|_| true);
            let half = lane / 16 * 16;
            let want: [u32; SHORT_COLUMNS] = [
                x(lane ^ 1),
                x(5),
                x(lane ^ 4),
                if lane >= 3 { x(lane - 3) } else { x(lane) },
                if lane % 16 + 3 < 16 { x(lane + 3) } else { x(lane) },
                mask_of(&|l| x(l) & 3 == x(lane) & 3),
                members,
                if (0..size).all(|l| x(l) & 1 == x(0) & 1) { members } else { 0 },
                (0..size).fold(0u32, |a, l| a.wrapping_add(x(l))),
                (0..size).map(x).max().unwrap(),
                (0..size).map(|l| x(l) as i32).min().unwrap() as u32,
                (half..half + 16).fold(0u32, |a, l| a.wrapping_add(x(l))),
                mask_of(&|l| x(l) & 1 == 1),
                (size == 16) as u32,
                (size == 32) as u32,
                (size == 16) as u32,
                1,
                members,
            ];
            assert_eq!(
                out[g * SHORT_COLUMNS..(g + 1) * SHORT_COLUMNS],
                want,
                "block {block}, warp {warp}, lane {lane}"
            );
        }
    }
    Case { name: "short_warps", kernel, cfg: launch_n(2, SHORT_BLOCK), small_only: false, check }
}

//! Math, integer, warp-collective, redux, packed and half-precision instruction options and their diagnostics.

use super::*;
/// A kernel applying `f` to a Float32 (and a second Float32) parameter.
fn math_kernel(f: impl Fn(Float32, Float32) -> Float32) -> Kernel {
    Kernel::builder("math_misuse").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, f(x.load(t), x.load(t)));
    })
}

/// CuTe DSL's rules for `cute.math`'s keyword options: fastmath excludes the
/// options that choose an instruction, ftz on a transcendental needs approx,
/// each function takes only its own options, and approx/full/rounding are
/// alternatives.
#[test]
fn math_options_follow_cute_dsl_rules() {
    let fast_approx = MathOptions { fastmath: true, approx: true, ..MathOptions::default() };
    let msg = build_panic(|| {
        math_kernel(|x, _| math::exp2_with(x, fast_approx));
    });
    assert!(msg.contains("fastmath cannot be combined with approx"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| {
            math::add_with(x, y, MathOptions { fastmath: true, ..MathOptions::rounding(RoundingMode::Zero) })
        });
    });
    assert!(msg.contains("fastmath cannot be combined with rounding"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, _| math::sin_with(x, MathOptions { ftz: true, ..MathOptions::default() }));
    });
    assert!(msg.contains("ftz needs approx"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, _| math::tanh_with(x, MathOptions::APPROX.with_ftz()));
    });
    assert!(msg.contains("cute::math::tanh takes no ftz option"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| math::add_with(x, y, MathOptions::APPROX));
    });
    assert!(msg.contains("cute::math::add takes no approx option"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| math::min_with(x, y, MathOptions::FASTMATH));
    });
    assert!(msg.contains("cute::math::min takes no fastmath option"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| {
            math::div_with(x, y, MathOptions { approx: true, ..MathOptions::rounding(RoundingMode::Up) })
        });
    });
    assert!(msg.contains("approx, full and rounding are alternatives"), "{msg}");
}

/// The options that choose one PTX instruction need Float32 or Float64, and
/// the instruction must exist: the validator names it and the user's line.
#[test]
fn math_instruction_options_must_exist_in_ptx() {
    let msg = build_panic(|| {
        Kernel::builder("half_approx").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<F16>("x", Layout::col_major(32));
            let t = thread_idx().x;
            x.store(t, math::sqrt_with(x.load(t), MathOptions::APPROX));
        });
    });
    assert!(msg.contains("needs Float32 or Float64 operands, not f16"), "{msg}");
    let double = |f: fn(Float64) -> Float64| {
        build_panic(move || {
            Kernel::builder("double_approx").block_dim([32, 1, 1]).build(|| {
                let x = param_tensor::<f64>("x", Layout::col_major(32));
                let t = thread_idx().x;
                x.store(t, f(x.load(t)));
            });
        })
    };
    let msg = double(|x| math::sin_with(x, MathOptions::APPROX));
    assert!(msg.contains("sin.approx.f64 is not a PTX instruction") && msg.contains("Float32 only"), "{msg}");
    assert!(msg.contains("tests/diagnostics/arithmetic.rs"), "the error should point at the call: {msg}");
    let msg = double(|x| math::rcp_with(x, MathOptions::APPROX));
    assert!(msg.contains("rcp.approx.f64 is not a PTX instruction") && msg.contains("rcp.approx.ftz.f64"), "{msg}");
    // ftz on Float64 exists only in the two approximations: not with a
    // rounding mode, explicit or implied.
    let msg = double(|x| math::rcp_with(x, MathOptions { ftz: true, ..MathOptions::default() }));
    assert!(msg.contains("rcp.rn.ftz.f64 is not a PTX instruction: ftz applies to Float32 only"), "{msg}");
    let msg = double(|x| math::rcp_with(x, MathOptions::rounding(RoundingMode::Zero).with_ftz()));
    assert!(msg.contains("rcp.rz.ftz.f64 is not a PTX instruction"), "{msg}");
    let msg = double(|x| math::sqrt_with(x, MathOptions::rounding(RoundingMode::Up).with_ftz()));
    assert!(msg.contains("sqrt.rp.ftz.f64 is not a PTX instruction"), "{msg}");
    let msg = double(|x| math::min_with(x, x, MathOptions::rounding(RoundingMode::NearestEven).with_ftz()));
    assert!(msg.contains("cute::math::min takes no rounding option"), "{msg}");
    let msg = double(|x| math::min_with(x, x, MathOptions { ftz: true, ..MathOptions::default() }));
    assert!(msg.contains("min.ftz.f64 is not a PTX instruction: ftz applies to Float32 only"), "{msg}");
    let msg = double(|x| math::add_with(x, x, MathOptions { ftz: true, ..MathOptions::default() }));
    assert!(msg.contains("add.rn.ftz.f64 is not a PTX instruction"), "{msg}");
    // The plain approximations of `Value` are Float32 only too.
    let msg = double(|x| x.exp2());
    assert!(msg.contains("ex2.approx") && msg.contains("cute::math::exp2 computes Float64"), "{msg}");
}

/// Errors inside a math expansion point at the user's call, not at
/// `src/math`.
#[test]
fn math_expansion_errors_point_at_the_caller() {
    let kernel = Kernel::builder("math_oob").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t + 1, math::exp(x.load(t), false));
    });
    let mut args = vec![KernelArg::f32(vec![0.5; 32])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.to_string().contains("tests/diagnostics/arithmetic.rs"), "{e}");
    let dump = kernel.dump();
    assert!(!dump.is_empty());
    // Every instruction of the expansion is attributed to this file.
    let ir = kernel.ir();
    let mut files = std::collections::BTreeSet::new();
    fn walk(r: &cute::ir::Region, files: &mut std::collections::BTreeSet<String>) {
        for inst in &r.insts {
            files.insert(inst.loc.file().to_string());
            for region in inst.op.regions() {
                walk(region, files);
            }
        }
    }
    walk(&ir.body, &mut files);
    assert!(files.iter().all(|f| !f.contains("src/math")), "{files:?}");
}

/// The message of a kernel build applying `f` to an element of a `T`
/// parameter, which must fail, and the line `f` is on.
#[track_caller]
fn int_misuse<T: Numeric, U: Numeric>(f: impl Fn(Value<T>) -> Value<U>) -> (String, u32) {
    let line = std::panic::Location::caller().line();
    let msg = build_panic(move || {
        Kernel::builder("int_misuse").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<T>("x", Layout::col_major(32));
            let y = param_tensor::<U>("y", Layout::col_major(32));
            let t = thread_idx().x;
            y.store(t, f(x.load(t)));
        });
    });
    (msg, line)
}

/// The integer and bit instructions take the types CuTe DSL takes (the
/// rest is up to Rust's types: one type for the data operands, a `u8`
/// lookup table, Int32 saturating arithmetic), and the validator names the
/// instruction, the type and the user's line.
#[test]
fn int_instructions_take_cute_dsl_types() {
    let cases: Vec<((String, u32), &str)> = vec![
        (int_misuse(|x: Int8| bfe(x, 0, 4)), "bfe takes Int32, Uint32, Int64 or Uint64 operands, not i8"),
        (int_misuse(|x: Uint16| bfe(x, 4, 8)), "bfe takes Int32, Uint32, Int64 or Uint64 operands, not u16"),
        (int_misuse(|x: Int16| bfi(x, x, 0, 4)), "bfi takes Int32, Uint32, Int64 or Uint64 operands, not i16"),
        (int_misuse(|x: Int64| lop3(x, x, x, 0x96)), "lop3 takes Int32 or Uint32 operands, not i64"),
        (int_misuse(|x: Uint16| shf(x, x, 3, ShfKind::Left)), "shf takes Int32 or Uint32 operands, not u16"),
        (int_misuse(|x: Uint64| prmt(x, x, 0x3210u64)), "prmt takes Int32 or Uint32 operands, not u64"),
        (int_misuse(|x: Int8| prmt(x, x, 0x10i8)), "prmt takes Int32 or Uint32 operands, not i8"),
        (int_misuse(|x: Int64| mul24(x, x, true)), "mul24 takes Int32 or Uint32 operands, not i64"),
        (int_misuse(|x: Uint8| mad24(x, x, x, false)), "mad24 takes Int32 or Uint32 operands, not u8"),
        (int_misuse(|x: Int16| mul_hi(x, x)), "mul_hi takes Int32, Uint32, Int64 or Uint64 operands, not i16"),
        (int_misuse(|x: Uint8| x.mul_hi(x)), "mul_hi takes Int32, Uint32, Int64 or Uint64 operands, not u8"),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(
            msg.contains(&format!("tests/diagnostics/arithmetic.rs:{line}:")),
            "the error should point at line {line}: {msg}"
        );
    }
}

/// One warp of 32 threads running `f` on its lane index and an element of a
/// Uint32 parameter; the simulator's error, which must be reported.
fn warp_misuse(name: &str, f: impl Fn(Int32, Uint32) -> Uint32) -> SimError {
    let kernel = Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, f(lane_idx(), x.load(t)));
    });
    simulate_err(&kernel, 32, &mut [KernelArg::buffer(&[7u32; 32])])
}

/// What the PTX ISA leaves undefined for the warp collectives, reported by
/// the simulator at the user's line: a lane executing one outside its member
/// mask, a lane of the mask that never executes it (lanes of another branch
/// may, with independent thread scheduling, but these take the branch that
/// does not: a deadlock naming them), lanes whose masks disagree, and a
/// shuffle reading a lane outside the mask.
#[test]
fn warp_collectives_need_their_member_masks() {
    let plain = ReduxOptions::default();
    let cases: Vec<(SimError, &str)> = vec![
        (
            warp_misuse("shuffle_outside", |_, x| shuffle_sync_with(x, 0, 0x0000_ffffu32, 31)),
            "shuffle executed by lane 16 of warp 0, which is not in its member mask 0x0000ffff",
        ),
        (
            warp_misuse("redux_outside", |_, x| warp_redux_sync_with(x, ReduxKind::Add, 0xffff_fffeu32, plain)),
            "warp_redux_sync executed by lane 0 of warp 0, which is not in its member mask 0xfffffffe",
        ),
        (
            warp_misuse("match_outside", |_, x| match_sync(0x7fff_ffffu32, x, MatchKind::All)),
            "match_sync executed by lane 31 of warp 0, which is not in its member mask 0x7fffffff",
        ),
        (
            warp_misuse("sync_warp_outside", |_, x| {
                sync_warp_with(0x0000_ffffu32);
                x
            }),
            "sync_warp executed by lane 16 of warp 0, which is not in its member mask 0x0000ffff",
        ),
        (
            warp_misuse("redux_inactive", |lane, x| {
                if_else_yield(lane.lt(16), || warp_redux_sync(x, ReduxKind::Max), || x)
            }),
            "with member mask 0xffffffff, for lanes 16..=31 of the mask to reach it",
        ),
        (
            warp_misuse("sync_warp_inactive", |lane, x| {
                if_(lane.ge(8), || sync_warp());
                x
            }),
            "for lanes 0..=7 of the mask to reach it",
        ),
        (
            warp_misuse("match_inactive", |lane, x| {
                if_else_yield(lane.ne(5), || match_sync(FULL_MASK, x, MatchKind::Any), || x)
            }),
            "lane 5 took another path and waits at the end of a branch or loop",
        ),
        (
            warp_misuse("masks_disagree", |lane, x| {
                sync_warp_with(select(lane.eq(0), 3u32, 0xffff_fffeu32));
                x
            }),
            "lane 0 of warp 0 executes sync_warp with member mask 0x00000003, which includes lane 1, but lane 1 \
             executes it with 0xfffffffe",
        ),
        (
            warp_misuse("shuffle_non_member", |lane, x| {
                if_else_yield(lane.lt(16), || shuffle_sync_with(x, 20, 0x0000_ffffu32, 31), || x)
            }),
            "shuffle reads lane 20, which is not in the member mask 0x0000ffff",
        ),
        (
            warp_misuse("vote_inactive", |lane, x| if_else_yield(lane.lt(16), || vote_ballot_sync(x.eq(7u32)), || x)),
            "warp 0 (lanes 0..=15) of block (0,0,0) waits at vote_ballot_sync",
        ),
        (
            warp_misuse("vote_all_inactive", |lane, x| {
                if_else_yield(lane.ne(9), || vote_all_sync(x.eq(7u32)).to(), || x)
            }),
            "waits at vote_all_sync",
        ),
    ];
    for (e, want) in cases {
        assert!(e.message.contains(want), "expected {want:?} in {e}");
        assert!(
            e.to_string().contains("tests/diagnostics/arithmetic.rs"),
            "the error should point at the kernel source: {e}"
        );
    }
}

/// The kinds and options of `warp_redux_sync` are checked against the value's
/// type, at the user's line: integer kinds on Int32 and Uint32, `fmax` and
/// `fmin` on Float32, `abs` and `nan` on Float32 only, nothing wider or
/// narrower than 32 bits.
#[test]
fn redux_kinds_follow_the_value_type() {
    let abs = ReduxOptions { abs: true, ..ReduxOptions::default() };
    let cases: Vec<((String, u32), &str)> = vec![
        (
            int_misuse(|x: Float32| warp_redux_sync(x, ReduxKind::Add)),
            "warp_redux_sync: add reduces Int32 or Uint32 values, not f32 (Float32 takes fmax, fmin, max and min)",
        ),
        (
            int_misuse(|x: Int32| warp_redux_sync(x, ReduxKind::Fmax)),
            "warp_redux_sync: fmax reduces Float32 values, not i32",
        ),
        (
            int_misuse(|x: Int32| warp_redux_sync_with(x, ReduxKind::Max, FULL_MASK, abs)),
            "redux.sync.max.abs.s32 is not a PTX instruction: abs and nan apply to the Float32 kinds (fmax, fmin) only",
        ),
        (
            int_misuse(|x: Uint64| warp_redux_sync(x, ReduxKind::Max)),
            "warp_redux_sync: max reduces Int32 or Uint32 values, not u64",
        ),
        (
            int_misuse(|x: Int16| warp_redux_sync(x, ReduxKind::Xor)),
            "warp_redux_sync: xor reduces Int32 or Uint32 values, not i16",
        ),
        (
            int_misuse(|x: Float64| warp_redux_sync(x, ReduxKind::Min)),
            "warp_redux_sync: min reduces Int32 or Uint32 values, not f64",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(
            msg.contains(&format!("tests/diagnostics/arithmetic.rs:{line}:")),
            "the error should point at line {line}: {msg}"
        );
    }
}

/// Float32 `redux.sync` exists on sm_100a, and from PTX ISA 8.8 on the
/// sm_100f family, only: compiling for any other target is refused with the
/// accepted targets and the user's line, naming the instruction with its
/// modifiers.
#[test]
fn float_redux_is_architecture_specific() {
    use cute::validate::{check_target, target_requirement};
    let line = line!() + 5;
    let kernel = Kernel::builder("float_redux").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let opts = ReduxOptions { abs: true, nan: true };
        x.store(t, warp_redux_sync_with(x.load(t), ReduxKind::Min, FULL_MASK, opts));
    });
    for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm120, Arch::Sm120a] {
        let e = check_target(kernel.ir(), arch).unwrap_err();
        let want = format!(
            "redux.sync.min.abs.NaN.f32 needs sm_100a (or the sm_100f family with PTX 8.8); the kernel is being \
             compiled for {}",
            arch.chip()
        );
        assert_eq!(e.message, want);
        assert!(e.to_string().contains(&format!("tests/diagnostics/arithmetic.rs:{line}:")), "{e}");
    }
    assert_eq!(check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    let redux = kernel.ir().body.insts.iter().find(|i| matches!(i.op, cute::ir::Op::Redux { .. })).unwrap();
    let need = target_requirement(&redux.op).unwrap();
    assert_eq!((need.sm, need.ptx), (100, 86));
    assert!(need.arch_specific.is_some());
    // The integer forms need sm_80 and PTX 7.0, like every supported target.
    let int = Kernel::builder("int_redux").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, warp_redux_sync(x.load(t), ReduxKind::Add));
    });
    for arch in Arch::ALL {
        assert_eq!(check_target(int.ir(), arch).unwrap(), arch.min_ptx(), "{arch:?}");
    }
}

/// `fmax` and `fmin` with `abs` are `.xorsign.abs`, which needs sm_86 (and
/// PTX 7.2): an sm_80 kernel is refused at the user's line. `nan` needs
/// sm_80, and `ftz` nothing.
#[test]
fn fmax_abs_needs_sm86() {
    use cute::validate::check_target;
    let line = line!() + 5;
    let kernel = Kernel::builder("fmax_abs").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let opts = MinMaxOptions { abs: true, ftz: true, ..MinMaxOptions::default() };
        x.store(t, fmax_with(x.load(t), x.load((t + 1) % 32), opts));
    });
    let e = check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert_eq!(
        e.message,
        "min and max with .xorsign.abs (fmax and fmin with abs) needs sm_86 or newer; the kernel is being compiled \
         for sm_80"
    );
    assert!(e.to_string().contains(&format!("tests/diagnostics/arithmetic.rs:{line}:")), "{e}");
    assert_eq!(check_target(kernel.ir(), Arch::Sm86).unwrap(), 72);
    assert_eq!(check_target(kernel.ir(), Arch::Sm89).unwrap(), 78);
    let nan = Kernel::builder("fmin_nan").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let opts = MinMaxOptions { nan: true, ftz: true, ..MinMaxOptions::default() };
        x.store(t, fmin_with(x.load(t), 0.5f32, opts));
    });
    assert_eq!(check_target(nan.ir(), Arch::Sm80).unwrap(), 70);
}

/// `warp_reduction`'s groups are a power of two up to the warp.
#[test]
fn warp_reduction_groups_are_powers_of_two() {
    for group in [12, 0, 64, 33] {
        let msg = build_panic(|| {
            Kernel::builder("bad_group").block_dim([32, 1, 1]).build(|| {
                let x = param_tensor::<f32>("x", Layout::col_major(32));
                let t = thread_idx().x;
                x.store(t, warp_reduction_sum(x.load(t), group));
            });
        });
        let want = format!("warp_reduction: threads_in_group is {group}, but it must be a power of two from 1 to 32");
        assert!(msg.contains(&want), "{msg}");
    }
}

/// The second warp of a block of 48 threads has 16 lanes: the full mask's
/// other lanes count as exited, so warp collectives with it run there, but
/// a shuffle reading one of them (undefined in the PTX ISA) is reported.
#[test]
fn short_warps_lack_the_lanes_past_the_block() {
    let kernel = Kernel::builder("short_read").block_dim([48, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(48));
        let t = thread_idx().x;
        let all = match_sync(FULL_MASK, 1u32, MatchKind::All);
        x.store(t, shuffle_sync(x.load(t) + all, 20));
    });
    let e = simulate_err(&kernel, 48, &mut [KernelArg::buffer(&[7u32; 48])]);
    let want = "shuffle reads lane 20, which warp 1 does not have: it is the last warp of a block of 48 threads, \
                with 16 lanes";
    assert!(e.message.contains(want), "{e}");
    assert!(
        e.to_string().contains("tests/diagnostics/arithmetic.rs"),
        "the error should point at the kernel source: {e}"
    );
}

/// An operation on two Int32 words, for [`packed_kernel`].
type PackedOp = Box<dyn Fn(Int32, Int32) -> Int32>;

/// A kernel applying `f` to two Int32 words.
fn packed_kernel(f: impl Fn(Int32, Int32) -> Int32) -> Kernel {
    Kernel::builder("packed").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major(64));
        let t = thread_idx().x;
        x.store(t, f(x.load(t), x.load(t + 32)));
    })
}

/// The half-precision options are checked as CuTe DSL checks them, in its
/// order, with cute-rs's own messages: `ftz` and `sat` are Float16 only,
/// `sat` and `relu` exclude each other, `oob` excludes `ftz` and `sat`, and
/// each operation takes only the keywords its CuTe DSL counterpart has.
#[test]
fn half_precision_options_follow_cute_dsl() {
    let opts = |ftz, sat, relu, oob| HalfOptions { ftz, sat, relu, oob, ..HalfOptions::default() };
    let cases: [(PackedOp, &str); 10] = [
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, a, opts(false, true, true, false))),
            "cute: fma_packed_f16x2: sat and relu exclude each other, but both are set",
        ),
        (
            Box::new(move |a, b| fma_packed_bf16x2(a, b, a, opts(true, false, false, false))),
            "cute: fma_packed_bf16x2: ftz applies to Float16 only, but it is set on a BFloat16 fma",
        ),
        (
            Box::new(move |a, b| add_packed_bf16x2(a, b, opts(false, true, false, false))),
            "cute: add_packed_bf16x2: sat applies to Float16 only, but it is set on a BFloat16 add",
        ),
        (
            Box::new(move |a, b| sub_packed_bf16x2(a, b, opts(true, false, false, false))),
            "cute: sub_packed_bf16x2: ftz applies to Float16 only, but it is set on a BFloat16 sub",
        ),
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, b, opts(true, false, false, true))),
            "cute: fma_packed_f16x2: oob excludes ftz and sat, but oob is set with ftz",
        ),
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, b, opts(false, true, false, true))),
            "cute: fma_packed_f16x2: oob excludes ftz and sat, but oob is set with sat",
        ),
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, b, opts(true, true, false, true))),
            "cute: fma_packed_f16x2: oob excludes ftz and sat, but oob is set with ftz and sat",
        ),
        (
            Box::new(move |a, b| add_packed_f16x2(a, b, opts(false, false, true, false))),
            "cute: add_packed_f16x2 takes no relu option",
        ),
        (
            Box::new(move |a, b| sub_packed_f16x2(a, b, opts(false, false, false, true))),
            "cute: sub_packed_f16x2 takes no oob option",
        ),
        (
            Box::new(|a, b| {
                let predicate = Some(a.lt(0));
                fma_packed_f16x2(a, b, b, HalfOptions { predicate, ..HalfOptions::default() })
            }),
            "cute: fma_packed_f16x2 takes no predicate",
        ),
    ];
    for (f, want) in cases {
        let msg = build_panic(|| {
            packed_kernel(f);
        });
        assert!(msg.contains(want), "expected {want:?} in {msg}");
    }
    let msg = build_panic(|| {
        Kernel::build("scalar", || {
            let x = param_tensor::<BF16>("x", Layout::col_major(32));
            let t = thread_idx().x;
            let opts = HalfOptions { sat: true, ..HalfOptions::default() };
            x.store(t, fma_bf16(x.load(t), x.load(t), x.load(t), opts));
        });
    });
    assert!(msg.contains("cute: fma_bf16: sat applies to Float16 only, but it is set on a BFloat16 fma"), "{msg}");
}

/// bf16x2 `add` and `sub`, `ex2.approx.ftz.bf16x2` and fma with `.oob` need
/// sm_90: compiling for sm_89 is refused at the user's line; `.oob` also
/// raises sm_90's PTX version to 8.1.
#[test]
fn bf16x2_add_needs_sm90() {
    use cute::validate::check_target;
    let line = line!() + 1;
    let kernel = packed_kernel(|a, b| add_packed_bf16x2(a, b, HalfOptions::default()));
    let e = check_target(kernel.ir(), Arch::Sm89).unwrap_err();
    assert_eq!(
        e.message,
        "add.bf16x2 (add_packed_bf16x2) needs sm_90 or newer; the kernel is being compiled for sm_89"
    );
    assert!(e.to_string().contains(&format!("tests/diagnostics/arithmetic.rs:{line}:")), "{e}");
    // sm_90's own PTX ISA version, 7.8.
    assert_eq!(check_target(kernel.ir(), Arch::Sm90).unwrap(), 78);
    let kernel = packed_kernel(|a, _| exp2_packed_bf16x2(a, None));
    let e = check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert!(e.message.contains("ex2.approx.ftz.bf16x2 (exp2_packed_bf16x2) needs sm_90 or newer"), "{e}");
    let oob = HalfOptions { oob: true, ..HalfOptions::default() };
    let kernel = packed_kernel(move |a, b| fma_packed_f16x2(a, b, b, oob));
    let e = check_target(kernel.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("fma with .oob (oob=True) needs sm_90 or newer"), "{e}");
    assert_eq!(check_target(kernel.ir(), Arch::Sm90).unwrap(), 81);
    // The Float16 forms and BFloat16 fma need nothing beyond sm_80 and PTX 7.0.
    let kernel = packed_kernel(|a, b| {
        let relu = HalfOptions { relu: true, ..HalfOptions::default() };
        fma_packed_bf16x2(add_packed_f16x2(a, b, HalfOptions::default()), b, a, relu)
    });
    assert_eq!(check_target(kernel.ir(), Arch::Sm80).unwrap(), 70);
}

/// The s2f6x2 path of the int8 to BFloat16 conversion is
/// architecture-specific: sm_100 (no `a`) is refused with the targets that
/// have it and the user's line.
#[test]
fn s2f6x2_path_needs_an_arch_specific_target() {
    use cute::validate::check_target;
    let line = line!() + 6;
    let kernel = Kernel::builder("s2f6x2").block_dim([32, 1, 1]).build(|| {
        let q = param_tensor::<i8>("q", Layout::col_major(128));
        let out = param_tensor::<BF16>("out", Layout::col_major(128));
        let t = thread_idx().x;
        let values: Vec<Int8> = (0..4).map(|k| q.load(t * 4 + k)).collect();
        let converted = cvt_i8_bf16_intrinsic_with(&values, 4, I8ToBf16Path::S2f6x2);
        for (k, v) in converted.into_iter().enumerate() {
            out.store(t * 4 + k as i32, v);
        }
    });
    let e = check_target(kernel.ir(), Arch::Sm100).unwrap_err();
    assert_eq!(
        e.message,
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 (the s2f6x2 int8 conversion) needs sm_100a, sm_103a, \
         sm_107a, sm_110a, sm_120a or sm_121a; the kernel is being compiled for sm_100"
    );
    assert!(e.to_string().contains(&format!("tests/diagnostics/arithmetic.rs:{line}:")), "{e}");
    for arch in [Arch::Sm80, Arch::Sm90, Arch::Sm90a, Arch::Sm120] {
        assert!(check_target(kernel.ir(), arch).is_err(), "{arch:?}");
    }
    assert_eq!(check_target(kernel.ir(), Arch::Sm100a).unwrap(), 91);
    assert_eq!(check_target(kernel.ir(), Arch::Sm120a).unwrap(), 91);
}

/// `cvt_i8_bf16_intrinsic` converts one to `vec_i8.len()` values and
/// `sext_unpacked_i4_i8_intrinsic` a positive multiple of four; other
/// lengths are refused.
#[test]
fn conversion_intrinsic_lengths_are_checked() {
    let cvt = |n: usize, length: usize| {
        build_panic(move || {
            Kernel::build("bad_length", || {
                let q = param_tensor::<i8>("q", Layout::col_major(64));
                let out = param_tensor::<BF16>("out", Layout::col_major(64));
                let t = thread_idx().x;
                let values: Vec<Int8> = (0..n as i32).map(|k| q.load(t + k)).collect();
                for v in cvt_i8_bf16_intrinsic(&values, length) {
                    out.store(t, v);
                }
            });
        })
    };
    let msg = cvt(4, 0);
    assert!(
        msg.contains("cute: cvt_i8_bf16_intrinsic: length is 0, but it must be from 1 to vec_i8.len() (4)"),
        "{msg}"
    );
    let msg = cvt(3, 4);
    assert!(
        msg.contains("cute: cvt_i8_bf16_intrinsic: length is 4, but it must be from 1 to vec_i8.len() (3)"),
        "{msg}"
    );
    for (n, length) in [(8usize, 6usize), (8, 0), (4, 8)] {
        let msg = build_panic(move || {
            Kernel::build("bad_sext", || {
                let q = param_tensor::<i8>("q", Layout::col_major(64));
                let t = thread_idx().x;
                let values: Vec<Int8> = (0..n as i32).map(|k| q.load(t + k)).collect();
                for v in sext_unpacked_i4_i8_intrinsic(&values, length) {
                    q.store(t, v);
                }
            });
        });
        let want = format!(
            "cute: sext_unpacked_i4_i8_intrinsic: length is {length}, but it must be a multiple of 4 from 4 to \
             vec_unpacked_i4.len() ({n})"
        );
        assert!(msg.contains(&want), "{msg}");
    }
}

/// `pack` and `unpack` need two or more parts whose widths add up to the
/// whole's, and no Booleans.
#[test]
fn pack_widths_must_match() {
    let msg = build_panic(|| {
        Kernel::build("bad_pack", || {
            let x = param_tensor::<i8>("x", Layout::col_major(32));
            let y = param_tensor::<i32>("y", Layout::col_major(32));
            let t = thread_idx().x;
            y.store(t, pack::<i8, i32, 3>([x.load(t), x.load(t), x.load(t)]));
        });
    });
    assert!(msg.contains("cute: pack: 3 x i8 is 24 bits, but i32 is 32"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bad_unpack", || {
            let x = param_tensor::<f64>("x", Layout::col_major(32));
            let t = thread_idx().x;
            let [a, _, _]: [Float32; 3] = unpack(x.load(t));
            x.store(t, a.to::<f64>());
        });
    });
    assert!(msg.contains("cute: unpack: 3 x f32 is 96 bits, but f64 is 64"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bool_pack", || {
            let y = param_tensor::<u8>("y", Layout::col_major(32));
            let t = thread_idx().x;
            let b = t.lt(3);
            y.store(t, pack::<bool, u8, 8>([b; 8]));
        });
    });
    assert!(msg.contains("Boolean values have no bit width"), "{msg}");
}

/// The validator checks half-precision instructions and Pack/Unpack in IR
/// built by other means: modifiers PTX lacks, an undefined predicate, a
/// predicate on fma, a predicate that is not a Boolean, operand types, and
/// widths.
#[test]
fn validator_checks_half_instructions_and_packing() {
    use cute::ir::{HalfFn, Op, Region};
    fn edit(r: &mut Region, f: &mut dyn FnMut(&mut Op)) {
        for inst in &mut r.insts {
            f(&mut inst.op);
            for region in inst.op.regions_mut() {
                edit(region, f);
            }
        }
    }
    let kernel = packed_kernel(|a, b| fma_packed_bf16x2(a, b, a, HalfOptions::default()));
    let mut ir = kernel.ir().clone();
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { mods, .. } = op {
            mods.ftz = true;
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("fma.rn.ftz.bf16x2 is not a PTX instruction: ftz applies to Float16 only"), "{e}");
    let mut ir = kernel.ir().clone();
    let p = ir.value_types.len();
    ir.value_types.push(DType::Bool);
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { predicate, .. } = op {
            *predicate = Some(cute::ir::ValueId(p as u32));
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains(&format!("%{p} is used before it is defined")), "{e}");
    // The predicate of an add (a Boolean defined before the fma) moved onto
    // the fma, and an Int32 as the add's predicate.
    let kernel = packed_kernel(|a, b| {
        let sum = add_packed_f16x2(a, b, HalfOptions { predicate: Some(a.eq(b)), ..HalfOptions::default() });
        fma_packed_f16x2(sum, b, a, HalfOptions::default())
    });
    let mut ir = kernel.ir().clone();
    let mut add_predicate = None;
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { func, predicate, .. } = op {
            match func {
                HalfFn::Add => add_predicate = *predicate,
                _ => *predicate = add_predicate,
            }
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("fma.rn.f16x2: CuTe DSL's fma helpers take no predicate"), "{e}");
    let mut ir = kernel.ir().clone();
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { func: HalfFn::Add, args, predicate, .. } = op {
            *predicate = Some(args[0]);
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("the predicate of add.f16x2 must be i1, but"), "{e}");
    let kernel = Kernel::build("pairs", || {
        let x = param_tensor::<F16>("x", Layout::col_major(64));
        let y = param_tensor::<i32>("y", Layout::col_major(32));
        let t = thread_idx().x;
        y.store(t, pack([x.load(t), x.load(t + 32)]));
    });
    let mut ir = kernel.ir().clone();
    let mut parts = None;
    edit(&mut ir.body, &mut |op| {
        if let Op::Pack { parts: p, .. } = op {
            parts = Some(p.clone());
            p.pop();
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("pack needs at least two parts, not 1"), "{e}");
    let mut ir = kernel.ir().clone();
    let first = parts.unwrap()[0];
    ir.value_types[first.0 as usize] = DType::I8;
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("pack: operand types differ"), "{e}");
}

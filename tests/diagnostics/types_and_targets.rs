//! Type and format misuse (tf32, fp8, fp4, int128, sub-byte), sparse and ldmatrix forms, bulk stores, build-time targets and structs.

use super::*;
#[test]
fn tf32_fragments_need_a_tf32_mma() {
    let msg = build_panic(|| {
        Kernel::build("tf32_on_f16_mma", || {
            let a = param_tensor::<F16>("a", Layout::col_major((16, 8)));
            let mma = make_tiled_mma(MmaKind::M16N8K8F32F16, Layout::col_major((1, 1, 1)));
            let thr = mma.get_slice(lane_idx());
            let _ = thr.partition_fragment_a::<TF32, _>(&a);
        });
    });
    assert!(msg.contains("fragments of") && msg.contains("hold f16, not tf32"), "{msg}");
}

#[test]
fn tf32_values_have_no_tf32_arithmetic_type() {
    // TFloat32 arithmetic is computed in Float32: the result is a Float32.
    let kernel = Kernel::build("tf32_sum", || {
        let out = param_tensor::<f32>("out", Layout::col_major(1));
        let x: TFloat32 = Float32::constant(1.0).to();
        let y: Float32 = x + x;
        out.store(0, y);
    });
    let mut args = vec![KernelArg::buffer(&[0f32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<f32>(), vec![2.0]);
}

#[test]
fn conversions_into_storage_only_fp8_formats_are_refused() {
    for (name, msg) in [
        ("Float8E4M3", "cannot convert Float32 to Float8E4M3: CuTe DSL refuses it as a conversion destination"),
        ("Float8E4M3B11FNUZ", "cannot convert Float32 to Float8E4M3B11FNUZ"),
        ("FloatNV8E5M3FNU", "cannot convert Float32 to FloatNV8E5M3FNU: its cvt (.ue5m3x2) needs PTX ISA 9.4"),
    ] {
        let got = build_panic(|| {
            Kernel::build("into_storage_fp8", || {
                let x = param_tensor::<f32>("x", Layout::col_major(32));
                let out = param_tensor::<u8>("out", Layout::col_major(32));
                let t = thread_idx().x;
                let v = x.load(t);
                let byte: Uint8 = match name {
                    "Float8E4M3" => v.to::<F8E4M3>().bitcast(),
                    "Float8E4M3B11FNUZ" => v.to::<F8E4M3B11FNUZ>().bitcast(),
                    _ => v.to::<F8E5M3FNU>().bitcast(),
                };
                out.store(t, byte);
            });
        });
        assert!(got.contains(msg) && got.contains("tests/diagnostics/types_and_targets.rs"), "{name}: {got}");
    }
    // Reading them is fine.
    Kernel::build("from_storage_fp8", || {
        let x = param_tensor::<F8E4M3B11FNUZ>("x", Layout::col_major(32));
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, x.load(t).to::<f32>());
    });
}

#[test]
fn ue8m0_conversions_need_sm100a() {
    let kernel = Kernel::build("ue8m0_on_sm90", || {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let out = param_tensor::<F8E8M0FNU>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let [a, b] = cvt_f32x2_to_ue8m0x2([x.load(2 * t), x.load(2 * t + 1)], true);
        out.store(2 * t, a);
        out.store(2 * t + 1, b);
    });
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90).unwrap_err();
    assert!(e.message.contains("cvt with .ue8m0x2") && e.message.contains("sm_100a"), "{e}");
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    // FP8 conversions need sm_89.
    let fp8 = Kernel::build("fp8_on_sm80", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let out = param_tensor::<F8E4M3FN>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, x.load(t).to::<F8E4M3FN>());
    });
    let e = cute::validate::check_target(fp8.ir(), Arch::Sm86).unwrap_err();
    assert!(e.message.contains("needs sm_89 or newer"), "{e}");
}

#[test]
fn int128_values_are_refused_where_there_is_no_lowering() {
    let printf_msg = build_panic(|| {
        Kernel::build("printf_i128", || {
            let x = Int128::constant(1 << 100);
            printf("%lld\n", &[x.into()]);
        });
    });
    assert!(
        printf_msg.contains("cannot take a 128-bit value")
            && printf_msg.contains("tests/diagnostics/types_and_targets.rs"),
        "{printf_msg}"
    );
    let shuffle_msg = build_panic(|| {
        Kernel::build("shuffle_i128", || {
            let out = param_tensor::<i128>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, shuffle_sync(t.to::<i128>(), 0));
        });
    });
    assert!(shuffle_msg.contains("128-bit"), "{shuffle_msg}");
    let ptx_msg = build_panic(|| {
        Kernel::build("inline_ptx_i128", || {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            let y: Int32 = InlinePtx::new("mov.b32 {$w0}, {$r0};").write::<i32>().read(t.to::<i128>()).emit();
            out.store(t, y);
        });
    });
    assert!(ptx_msg.contains("cannot take a 128-bit value"), "{ptx_msg}");
    let float_msg = build_panic(|| {
        Kernel::build("i128_to_f32", || {
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, t.to::<i128>().to::<f32>());
        });
    });
    assert!(float_msg.contains("convert through Int64"), "{float_msg}");
}

/// A scalar access to a sub-byte tensor is refused when the kernel is
/// built, as CuTe DSL's `_check_can_dereference` refuses it.
#[test]
fn int4_scalar_access_is_refused() {
    let msg = build_panic(|| {
        Kernel::build("store_i4", || {
            let out = param_tensor::<I4>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, t.to::<I4>());
        });
    });
    assert!(msg.contains("a store of one Int4 element") && msg.contains("element by element"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("load_i4_ptr", || {
            let src = param_tensor::<I4>("src", Layout::col_major(32));
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            out.store(0, src.ptr().load().to::<i32>());
        });
    });
    assert!(msg.contains("a load of one Int4 element"), "{msg}");
    let popc = build_panic(|| {
        Kernel::build("popc_i4", || {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, t.to::<I4>().popc().to::<i32>());
        });
    });
    assert!(popc.contains("cannot take an Int4 value"), "{popc}");
}

#[test]
fn int128_arithmetic_breaks_a_carry_chain() {
    let msg = build_panic(|| {
        Kernel::build("carry_around_i128", || {
            let out = param_tensor::<u32>("out", Layout::col_major(32));
            let wide = param_tensor::<i128>("wide", Layout::col_major(32));
            let t = thread_idx().x.to::<u32>();
            let lo = add_cc(t, 1u32);
            wide.store(0, wide.load(1) + wide.load(2));
            let hi = addc(t, 0u32);
            out.store(0, lo ^ hi);
        });
    });
    assert!(msg.contains("128-bit integer arithmetic"), "{msg}");
}

/// CuTe DSL's `MmaFP8Op` and `MmaF16BF16SparseOp` constructors pick the
/// instruction and reject what PTX lacks, with `MmaF16BF16Op`'s wording.
#[test]
fn fp8_and_sparse_mma_constructors() {
    let fp8 = |ab, acc, shape| MmaFP8Op::new(ab, acc, shape).kind();
    assert_eq!(fp8(DType::F8E4M3FN, DType::F32, (16, 8, 32)), MmaKind::M16N8K32F32E4M3);
    assert_eq!(fp8(DType::F8E5M2, DType::F16, (16, 8, 16)), MmaKind::M16N8K16F16E5M2);
    assert_eq!(MmaKind::M16N8K16F16E5M2.ptx(), "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16");
    let sp = |ab, acc, shape| MmaF16BF16SparseOp::new(ab, acc, shape, SparseMetadataFormat::Tid).kind();
    assert_eq!(sp(DType::BF16, DType::F32, (16, 8, 32)), MmaKind::SpM16N8K32F32BF16);
    assert_eq!(sp(DType::F16, DType::F16, (16, 8, 16)), MmaKind::SpM16N8K16F16F16);
    assert_eq!(
        MmaKind::SpM16N8K32F32BF16.ptx(),
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"
    );
    for (f, want) in [
        (
            Box::new(|| {
                let _ = MmaFP8Op::new(DType::F16, DType::F32, (16, 8, 32));
            }) as Box<dyn FnOnce()>,
            "'ab_dtype'",
        ),
        (
            Box::new(|| {
                let _ = MmaFP8Op::new(DType::F8E4M3FN, DType::BF16, (16, 8, 32));
            }),
            "'acc_dtype'",
        ),
        (
            Box::new(|| {
                let _ = MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 8));
            }),
            "(16,8,16) and (16,8,32)",
        ),
        (
            Box::new(|| {
                let _ = sp(DType::F8E4M3FN, DType::F32, (16, 8, 16));
            }),
            "'ab_dtype'",
        ),
        (
            Box::new(|| {
                let _ = sp(DType::BF16, DType::F16, (16, 8, 16));
            }),
            "Float32 only",
        ),
        (
            Box::new(|| {
                let _ = sp(DType::F16, DType::F32, (16, 8, 8));
            }),
            "'shape_mnk'",
        ),
    ] {
        let msg = build_panic(f);
        assert!(msg.contains(want), "{msg}");
    }
    // A sparse MMA goes through gemm_sparse, a dense one through gemm.
    let msg = build_panic(|| {
        Kernel::build("dense_gemm_on_sparse", || {
            let mma = make_tiled_mma(
                MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid),
                Layout::col_major((1, 1, 1)),
            );
            let (a, b, c) =
                (make_fragment::<F16>((4, 1, 1)), make_fragment::<F16>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
            gemm(&mma, &c, &a, &b, &c);
        });
    });
    assert!(msg.contains("call gemm_sparse"), "{msg}");
    // FP8 MMAs need sm_89.
    let kernel = Kernel::build("fp8_mma_on_sm80", || {
        let mma = make_tiled_mma(MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)), Layout::col_major((1, 1, 1)));
        let (a, b, c) = (
            make_fragment::<F8E4M3FN>((16, 1, 1)),
            make_fragment::<F8E4M3FN>((8, 1, 1)),
            make_fragment::<f32>((4, 1, 1)),
        );
        a.fill(F8E4M3FN::from_f32(1.0));
        b.fill(F8E4M3FN::from_f32(1.0));
        c.fill(0.0);
        gemm(&mma, &c, &a, &b, &c);
    });
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert!(e.to_string().contains("sm_89") && e.to_string().contains("FP8 mma.sync"), "{e}");
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm89).unwrap(), 84);
}

/// `LdMatrix8x16x8bOp` takes 8-bit elements, unpacks only 4 or 6 bits, and
/// needs K-major (row-contiguous) operands: an M-major FP8 A in shared
/// memory cannot feed it.
#[test]
fn ldmatrix_8x16x8b_misuse() {
    let msg = build_panic(|| drop(make_copy_atom::<F16>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None })));
    assert!(msg.contains("8-bit elements"), "{msg}");
    let msg = build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: Some(5) })));
    assert!(
        msg.contains("LdMatrix8x16x8bOp unpacks 4- or 6-bit elements") && msg.contains("not unpack_bits=5"),
        "{msg}"
    );
    let msg = build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix8x16x8b { num: 3, unpack_bits: Some(4) })));
    assert!(msg.contains("1, 2 or 4 matrices, not 3"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("m_major_fp8_a").block_dim([32, 1, 1]).build(|| {
            let sa = make_smem_tensor::<F8E4M3FN>(Layout::col_major((16, 32)));
            let mma =
                make_tiled_mma(MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)), Layout::col_major((1, 1, 1)));
            let ldsm = make_tiled_copy_a(
                &make_copy_atom::<F8E4M3FN>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None }),
                &mma,
            );
            let t = thread_idx().x;
            let fa = mma.get_slice(t).partition_fragment_a::<F8E4M3FN, _>(&sa);
            copy(&ldsm, &ldsm.get_slice(t).partition_s(&sa), &ldsm.retile(&fa));
        });
    });
    assert!(msg.contains("contiguous source elements"), "{msg}");
}

/// A sparse MMA's metadata must name two increasing positions per chunk of
/// four; anything else (here `0b0000`) is undefined on the GPU and reported
/// by the simulator.
#[test]
fn invalid_sparse_metadata() {
    let kernel = Kernel::builder("bad_sparse_metadata").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(128));
        let op = MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid);
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let (a, b, c) =
            (make_fragment::<F16>((4, 1, 1)), make_fragment::<F16>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
        let e = make_fragment::<u32>((1, 1, 1));
        a.fill(F16::ONE);
        b.fill(F16::ONE);
        c.fill(0.0);
        let lane = lane_idx();
        e.store(0, select(lane.eq(4), Uint32::from(0x4444_4440u32), Uint32::from(0x4444_4444u32)));
        gemm_sparse(&mma, &c, &a, &e, &b, &c);
        for i in 0..4usize {
            out.store(lane * 4 + i as i32, c.load(i));
        }
    });
    let mut args = vec![KernelArg::f32(vec![0.0; 128])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("nibble 0 of lane 4") && e.message.contains("0b0000"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/types_and_targets.rs"), "{e}");
}

/// Sub-byte tensors: shared memory in whole bytes, vectors of whole 32-bit
/// words, recasts whose unit-stride mode divides evenly, and two warps
/// writing different nibbles of one byte (whole words on hardware) race.
#[test]
fn sub_byte_tensor_misuse() {
    let msg = build_panic(|| {
        Kernel::build("odd_int4_smem", || {
            let _ = make_smem_tensor::<I4>(Layout::col_major(15));
        });
    });
    assert!(msg.contains("15 Int4 values is 60 bits, not a whole number of bytes"), "{msg}");
    let msg = build_panic(|| drop(make_copy_atom::<I4>(CopyOp::Universal { bits: 16 })));
    assert!(msg.contains("multiple of 32 bits"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("int4_vector_of_16_bits", || {
            let src = param_tensor::<I4>("src", Layout::col_major(64));
            let _: [Int4; 4] = load_vector(&src.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("4 Int4 values is 16 bits"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("int4_autovec_strided", || {
            let src = param_tensor::<I4>("src", Layout::new(8, 2));
            let frag = make_fragment::<I4>(8);
            autovec_copy(&src, &frag);
        });
    });
    assert!(msg.contains("whole 32-bit words"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("recast_uneven", || {
            let src = param_tensor::<I4>("src", Layout::row_major((4, 3)));
            let _ = recast_tensor::<u8, _>(&src);
        });
    });
    assert!(msg.contains("unit-stride mode") && msg.contains("extent 3"), "{msg}");

    // Warp 0 and warp 1 each store a word of eight Int4 values to the same
    // shared word, meaning to change one nibble each.
    let kernel = Kernel::builder("nibble_race").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<I4>("out", Layout::col_major(8));
        let s = make_smem_tensor::<I4>(Layout::col_major(8));
        let t = thread_idx().x;
        for (thread, nibble, value) in [(0, 0, 3), (32, 1, 5)] {
            if_(t.eq(thread), || {
                let frag = make_fragment::<I4>(8);
                frag.fill(I4::ZERO);
                frag.store(nibble, I4::new(value));
                autovec_copy(&frag, &s);
            });
        }
        sync_threads();
        if_(t.eq(0), || {
            let back = make_fragment::<I4>(8);
            autovec_copy(&s, &back);
            autovec_copy(&back, &out);
        });
    });
    let mut args = vec![KernelArg::buffer(&[I4::ZERO; 8])];
    let e = simulate_err(&kernel, 64, &mut args);
    assert!(e.message.contains("race"), "{e}");
}

/// FP4 and FP6: no scalar `to()` into them (CuTe DSL refuses them as
/// destinations), no packed view as a conversion operand, their `cvt`s need
/// sm_100a, sm_101a or sm_120a, and an FP6 vector is a whole number of
/// 16-value (96-bit) groups.
#[test]
fn fp4_fp6_misuse() {
    let msg = build_panic(|| {
        Kernel::build("to_fp4", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let out = param_tensor::<F4E2M1FN>("out", Layout::col_major(8));
            let frag = make_fragment::<F4E2M1FN>(8);
            frag.fill(x.load(0).to::<F4E2M1FN>());
            autovec_copy(&frag, &out);
        });
    });
    assert!(msg.contains("to() destination") && msg.contains("cvt_f32x2_to_f4e2m1x2"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("to_fp6", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let out = param_tensor::<F6E3M2FN>("out", Layout::col_major(16));
            let frag = make_fragment::<F6E3M2FN>(16);
            frag.fill(x.load(0).to::<F6E3M2FN>());
            autovec_copy(&frag, &out);
        });
    });
    assert!(msg.contains("Float6E3M2FN") && msg.contains("cvt_f32x2_to_f6x2"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("view_to_f32", || {
            let x = param_tensor::<F4E2M1FNx2>("x", Layout::col_major(32));
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            out.store(0, x.load(0).to::<f32>());
        });
    });
    assert!(msg.contains("packed view") && msg.contains("Float4E2M1FN"), "{msg}");
    let kernel = Kernel::build("fp4_cvt", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let out = param_tensor::<F16>("out", Layout::col_major(32));
        let [a, b] = cvt_f32x2_to_f4e2m1x2([x.load(0), x.load(1)], false);
        let (lo, _): (Float16, Float16) = cvt_f4e2m1x2_to_f16x2([a, b]);
        out.store(0, lo);
    });
    for arch in [Arch::Sm90, Arch::Sm90a, Arch::Sm100] {
        let e = cute::validate::check_target(kernel.ir(), arch).unwrap_err().to_string();
        assert!(e.contains(".e2m1x2") && e.contains("sm_100a"), "{e}");
    }
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm120a).unwrap(), 87);
    let msg = build_panic(|| {
        Kernel::build("fp6_by_eights", || {
            let src = param_tensor::<F6E2M3FN>("src", Layout::col_major(64));
            let frag = make_fragment::<F6E2M3FN>(8);
            copy(&make_copy_atom::<F6E2M3FN>(CopyOp::Universal { bits: 48 }), &local_tile(&src, 8, 0), &frag);
        });
    });
    assert!(msg.contains("multiple of 32 bits"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("fp6_misaligned", || {
            let src = param_tensor::<F6E2M3FN>("src", Layout::col_major(64));
            let _: [Float6E2M3FN; 16] = load_vector(&src.ptr().offset(8usize), LoadOptions::default());
        });
    });
    assert!(msg.contains("aligned to 16 elements") || msg.contains("whole number of 32-bit words"), "{msg}");
}

/// The tensor metadata and `autovec_copy` refuse what CuTe DSL refuses:
/// the stride of a swizzled tensor, the address space of a coordinate
/// tensor, a copy between tensors of different sizes, and an eviction
/// priority outside a copy between global memory and registers.
#[test]
fn tensor_metadata_and_autovec_misuse() {
    let msg = build_panic(|| {
        Kernel::build("swizzled_stride", || {
            let s = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(3, 2, 3), 0, Layout::row_major((8, 32))));
            let _ = s.stride();
        });
    });
    assert!(msg.contains("Tensor::stride: tensor (8,32):(32,1) is swizzled by S<3,2,3>"), "{msg}");
    assert!(msg.contains("use tensor.layout().stride()"), "{msg}");
    let msg = build_panic(|| {
        let _ = make_identity_tensor((4, 8)).memspace();
    });
    assert!(msg.contains("Tensor::memspace: tensor (4,8):(1@0,1@1) is a coordinate tensor"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("autovec_sizes", || {
            let x = param_tensor::<f32>("x", Layout::row_major((4, 8)));
            autovec_copy(&x, &make_fragment::<f32>((4, 4)));
        });
    });
    assert!(
        msg.contains(
            "autovec_copy: tensors have different sizes: the source (4,8):(8,1) has 32 elements and the \
                      destination (4,4):(1,4) has 16"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("autovec_evict_smem", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let s = make_smem_tensor::<f32>(Layout::col_major(32));
            let opts = AutovecOptions { l1c_evict_priority: CacheEvictionPriority::EvictFirst };
            autovec_copy_with(&x, &s, opts);
        });
    });
    assert!(
        msg.contains("autovec_copy_with: an L1 eviction priority (evict_first) is for a copy between global"),
        "{msg}"
    );
    assert!(msg.contains("this copy is from gmem to smem"), "{msg}");
}

/// A 32-byte vector needs a 32-byte-aligned global buffer of 32- or 64-bit
/// elements: the validator refuses one to shared memory, of 16-bit
/// elements, or from a buffer declared 16-byte aligned.
#[test]
fn wide_vector_misuse() {
    let msg = build_panic(|| {
        Kernel::build("wide_smem", || {
            let s = make_smem_tensor::<f32>(Layout::col_major(64));
            let _: [Float32; 8] = load_vector(&s.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("vector access of 8 x f32 is not a power-of-two width of at most 16 bytes (32-byte"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide_f16", || {
            let x = param_tensor_aligned::<F16>("x", Layout::col_major(64), 32);
            let _: [Float16; 16] = load_vector(&x.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("vector access of 16 x f16 is not a power-of-two width of at most 16 bytes"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide_16", || {
            let x = param_tensor::<f32>("x", Layout::col_major(64));
            let _: [Float32; 8] = load_vector(&x.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("a load of 32 bytes in `x`, which is only declared 16-byte aligned"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide_param", || {
            let _ = param_tensor_aligned::<f32>("x", Layout::col_major(64), 24);
        });
    });
    assert!(msg.contains("param_tensor_aligned: an alignment of 24 bytes is not a power of two of at least 16"));
}

/// What each bulk-store diagnostic kernel gets wrong (see [`bulk_store_bug`]).
#[derive(Clone, Copy, PartialEq)]
enum BulkBug {
    /// Generic writes, a barrier and the copy, without a proxy fence.
    NoProxyFence,
    /// Warp 1 fences after the barrier, so nothing orders its fence before
    /// warp 0's copy.
    UnorderedFence,
    /// The source is overwritten without `cp_async_bulk_wait_group(0, true)`.
    NoReadWait,
    /// The destination is read after the `.read` wait, before the plain one.
    EarlyGlobalRead,
    /// Thread 0 waits for the source to be read, but warp 1 overwrites it
    /// with no barrier ordering it after that wait.
    ReadWaitNoBarrier,
    /// A runtime size of 24 bytes.
    RuntimeSize,
    /// A runtime source offset of 4 elements: 16 bytes past an aligned start
    /// of an f32 buffer is fine, so the offset is 2 elements (8 bytes).
    RuntimeMisaligned,
}

/// 64 threads write a 64-word shared tile; thread 0 stores it to `out` with
/// a bulk copy, as [`BulkBug`] says, wrongly.
fn bulk_store_bug(bug: BulkBug) -> Kernel {
    Kernel::builder("bulk_store_bug").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let n = param_scalar::<i32>("n");
        let smem = alloc_smem::<f32>(64, 128);
        let t = thread_idx().x;
        let tile = make_tensor(&smem, Layout::col_major(64));
        if bug != BulkBug::UnorderedFence {
            tile.store(t, t.to::<f32>());
        } else {
            if_(warp_idx().eq(1), || tile.store(t, t.to::<f32>()));
        }
        if !matches!(bug, BulkBug::NoProxyFence | BulkBug::UnorderedFence) {
            fence_view_async_shared();
        }
        sync_threads();
        if bug == BulkBug::UnorderedFence {
            if_(warp_idx().eq(1), fence_view_async_shared);
        }
        if_(t.eq(0), || {
            let (src, bytes) = match bug {
                BulkBug::RuntimeSize => (smem.clone(), Int32::from(n)),
                BulkBug::RuntimeMisaligned => (smem.offset(n), Int32::from(64)),
                _ => (smem.clone(), Int32::from(256)),
            };
            cp_async_bulk_global_shared_cta(&out.ptr(), &src, bytes);
            cp_async_bulk_commit_group();
            if bug != BulkBug::NoReadWait {
                cp_async_bulk_wait_group(0, true);
            }
            if bug == BulkBug::EarlyGlobalRead {
                tile.store(0, out.load(0));
            }
        });
        if bug != BulkBug::ReadWaitNoBarrier {
            sync_threads();
        }
        tile.store(t, 0.0f32);
        if_(t.eq(0), || cp_async_bulk_wait_group(0, false));
    })
}

#[test]
fn bulk_store_misuse_in_the_simulator() {
    let run = |bug, n: i32| {
        let mut args = vec![KernelArg::f32(vec![0.0; 64]), KernelArg::ScalarI32(n)];
        bulk_store_bug(bug).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).err()
    };
    // The kernel as written for the runtime cases is correct with the right values.
    assert!(run(BulkBug::RuntimeSize, 256).is_none());
    let e = run(BulkBug::NoProxyFence, 0).expect("a missing proxy fence");
    assert!(e.message.contains("cp.async.bulk.global.shared::cta reads shared memory through the async proxy"), "{e}");
    assert!(e.message.contains("through the generic proxy, without a fence.proxy.async since"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/types_and_targets.rs"), "{e}");
    let e = run(BulkBug::UnorderedFence, 0).expect("an unordered fence");
    assert!(e.message.contains("nothing orders the fence before this operation's issue"), "{e}");
    let e = run(BulkBug::NoReadWait, 0).expect("a source overwritten too early");
    assert!(e.message.contains("data race: warp"), "{e}");
    assert!(e.message.contains("which an asynchronous operation issued by"), "{e}");
    let e = run(BulkBug::EarlyGlobalRead, 0).expect("a destination read too early");
    assert!(e.message.contains("reads element 0 of `out`, which a cp.async.bulk issued by thread 0"), "{e}");
    // Warp 1's overwrite must be ordered after thread 0's `.read` wait,
    // whichever order the warps run in and whenever the copy completes.
    for schedule in Schedule::ALL {
        for async_completion in [AsyncCompletion::Late, AsyncCompletion::Early] {
            let opts = SimOptions { schedule, async_completion, ..SimOptions::default() };
            let mut args = vec![KernelArg::f32(vec![0.0; 64]), KernelArg::ScalarI32(0)];
            let e = bulk_store_bug(BulkBug::ReadWaitNoBarrier)
                .simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts)
                .expect_err("an overwrite not ordered after the .read wait");
            assert!(e.message.contains("data race"), "{schedule:?} {async_completion:?}: {e}");
        }
    }
    let e = run(BulkBug::RuntimeSize, 24).expect("a size of 24 bytes");
    assert!(e.message.contains("cp.async.bulk of 24 bytes is not one or more whole 16-byte units"), "{e}");
    let e = run(BulkBug::RuntimeMisaligned, 2).expect("an 8-byte offset");
    assert!(
        e.message.contains("misaligned cp.async.bulk source read: element 2 of `smem0` is not 16-byte aligned"),
        "{e}"
    );
}

#[test]
fn bulk_store_misuse_when_built() {
    let msg = build_panic(|| {
        Kernel::build("bulk_size", || {
            let out = param_tensor::<f32>("out", Layout::col_major(64));
            let smem = alloc_smem::<f32>(64, 128);
            cp_async_bulk_global_shared_cta(&out.ptr(), &smem, 24);
        });
    });
    assert!(
        msg.contains("cp.async.bulk.global.shared::cta of 24 bytes: bulk copies move one or more whole 16-byte units"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("bulk_offset", || {
            let out = param_tensor::<f32>("out", Layout::col_major(64));
            let smem = alloc_smem::<f32>(64, 128);
            cp_async_bulk_global_shared_cta(&out.ptr().offset(2), &smem, 32);
        });
    });
    assert!(msg.contains("the destination is at byte 8 of its buffer, which is not 16-byte aligned"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_atom_size", || {
            let out = param_tensor::<f32>("out", Layout::col_major(6));
            let smem = make_smem_tensor::<f32>(Layout::col_major(6));
            copy(&make_copy_atom::<f32>(cpasync::CopyBulkS2GOp), &smem, &out);
        });
    });
    assert!(msg.contains("copy: a CopyBulkS2GOp copy of 6 f32 elements is 24 bytes, not a multiple of 16"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_tensormap", || fence_proxy(FenceProxyKind::Tensormap));
    });
    assert!(msg.contains("use fence_tma_desc_acquire or fence_tma_desc_release"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_space", || fence_proxy(FenceProxyKind::AsyncShared));
    });
    assert!(msg.contains("the AsyncShared kind needs a space"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_wait", || cp_async_bulk_wait_group(param_scalar::<i32>("n"), false));
    });
    assert!(msg.contains("cp_async_bulk_wait_group: the group count is a runtime value"), "{msg}");
}

/// Threads initialise a shared stage with ordinary stores that a bulk copy
/// then overwrites through the async proxy: without a proxy fence between
/// them the simulator reports it; with `fence_view_async_shared` the kernel
/// runs.
#[test]
fn bulk_load_over_generic_writes_needs_a_proxy_fence() {
    let kernel = |fenced: bool| {
        Kernel::builder("bulk_load_overwrite").block_dim([32, 1, 1]).build(move || {
            let src = param_tensor::<f32>("src", Layout::col_major(32));
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            let (stage, bar) = (alloc_smem::<f32>(32, 128), alloc_smem::<u64>(1, 8));
            let t = thread_idx().x;
            make_tensor(&stage, Layout::col_major(32)).store(t, 0.0f32);
            if fenced {
                fence_view_async_shared();
            }
            if_(t.eq(0), || mbarrier_init(&bar, 1));
            mbarrier_init_fence();
            sync_threads();
            if_(t.eq(0), || {
                mbarrier_arrive_and_expect_tx(&bar, 128);
                cp_async_bulk(&src.ptr(), &stage, 128, &bar);
            });
            mbarrier_wait(&bar, 0);
            out.store(t, make_tensor(&stage, Layout::col_major(32)).load(t));
        })
    };
    let mut args = vec![KernelArg::f32((0..32).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; 32])];
    let e = kernel(false).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("cp.async.bulk overwrites shared memory through the async proxy"), "{e}");
    assert!(e.message.contains("without a fence.proxy.async since"), "{e}");
    kernel(true).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].as_f32()[31], 31.0);
}

/// The 8-bit `ldmatrix` atoms of CuTe DSL's `LdMatrix16x16x8bOp` and
/// `LdMatrix16x8x8bOp` must be transposed and take 1 or 2 (2 or 4) matrices
/// and 4, 6 or no unpacked bits, as CuTe DSL's constructors check; the
/// `.m16n16` and `.m8n16` shapes need an sm_100a-class target (not sm_90a
/// or plain sm_100), compiled at the PTX ISA version of that target.
#[test]
fn ldmatrix_b8_misuse() {
    let msg = build_panic(|| {
        drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 1, trans: false, unpack_bits: None }))
    });
    assert!(msg.contains("LdMatrix16x16x8bOp must be transposed"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 4, trans: true, unpack_bits: None })));
    assert!(msg.contains("LdMatrix16x16x8bOp loads 1 or 2 matrices, not 4"), "{msg}");
    let msg = build_panic(|| {
        drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 2, trans: true, unpack_bits: Some(8) }))
    });
    assert!(msg.contains("LdMatrix16x16x8bOp unpacks") && msg.contains("not unpack_bits=8"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x8x8b { num: 2, trans: false, unpack_bits: None })));
    assert!(msg.contains("LdMatrix16x8x8bOp must be transposed"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x8x8b { num: 1, trans: true, unpack_bits: None })));
    assert!(msg.contains("LdMatrix16x8x8bOp loads 2 or 4 matrices, not 1"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<F16>(CopyOp::LdMatrix16x8x8b { num: 2, trans: true, unpack_bits: None })));
    assert!(msg.contains("8-bit elements"), "{msg}");
    let kernel = Kernel::builder("ldmatrix_m16n16_on_sm90").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u8>("out", Layout::row_major((32, 8)));
        let smem = make_smem_tensor::<u8>(Layout::col_major((16, 16)));
        let atom = make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 1, trans: true, unpack_bits: None });
        let tiled = make_tiled_copy(&atom, atom.layout_ref_tv(), (16, 16));
        let t = thread_idx().x;
        let frag = make_fragment::<u8>(8);
        copy(&tiled, &tiled.get_slice(t).partition_s(&smem), &tiled.retile(&frag));
        for v in 0..8 {
            out.store((t, v), frag.load(v));
        }
    });
    for arch in [Arch::Sm90a, Arch::Sm100] {
        let e = cute::validate::check_target(kernel.ir(), arch).unwrap_err().to_string();
        assert!(e.contains("ldmatrix.m16n16 with .b8") && e.contains("needs sm_100a"), "{e}");
        assert!(e.contains(&format!("compiled for {}", arch.chip())), "{e}");
    }
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm120a).unwrap(), 87);
    assert_eq!(kernel.min_llvm_version(), Some(22));
}

/// Pointer swizzles (`recast_ptr`): one with a swizzled composed layout, a
/// buffer not aligned to the swizzle's period, and a byte swizzle finer than
/// the element are refused at the user's line.
#[test]
fn pointer_swizzle_misuse() {
    let msg = build_panic(|| {
        Kernel::build("two_swizzles", || {
            let p = recast_ptr::<F16, _>(&alloc_smem::<F16>(512, 1024), Some(make_swizzle(3, 4, 3)));
            let _ = make_tensor(&p, make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64))));
        });
    });
    assert!(
        msg.contains("make_tensor: the pointer is swizzled by S<3,4,3> and the layout S<3,3,3> o 0 o (8,64):(64,1)")
            && msg.contains("get_nonswizzle_portion"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("unaligned_swizzle_base", || {
            let _ = recast_ptr::<f32, _>(&param_ptr::<f32>("x"), Some(make_swizzle(3, 4, 3)));
        });
    });
    assert!(
        msg.contains("the pointer swizzle S<3,4,3> acts on absolute byte addresses")
            && msg.contains("period of 1024 bytes")
            && msg.contains("only known to be 16-byte aligned"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("fine_swizzle", || {
            let _ = recast_ptr::<f32, _>(&alloc_smem::<f32>(256, 1024), Some(make_swizzle(3, 1, 3)));
        });
    });
    assert!(
        msg.contains("a pointer to Float32 swizzled by S<3,1,3> (on bytes)")
            && msg.contains("finer than the 4 bytes of an element"),
        "{msg}"
    );
    // A swizzled composed layout on a pointer that is not a whole number of
    // periods into its buffer.
    let msg = build_panic(|| {
        Kernel::build("swizzle_off_period", || {
            let p = alloc_smem::<f32>(1024, 128).offset(4);
            let _ = make_tensor(&p, make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32))));
        });
    });
    assert!(msg.contains("which cute-rs needs to be a multiple of 128 elements"), "{msg}");
}

/// A kernel built for a target (`KernelOptions::target`) is checked when the
/// build ends, naming the line of the instruction the target lacks; it
/// compiles for that target by default and refuses another; the body reads
/// the target with `target_arch`.
#[test]
fn build_time_targets() {
    use cute::validate::check_target;
    let fp4 = |arch: Arch| {
        Kernel::builder("fp4_pair").block_dim([32, 1, 1]).target(arch).build(|| {
            let out = param_tensor::<F4E2M1FN>("out", Layout::col_major(64));
            assert_eq!(target_arch(), Some(arch));
            let t = thread_idx().x;
            let x = t.to::<f32>();
            let line = line!() + 1;
            let [p, q] = cvt_f32x2_to_f4e2m1x2([x, -x], false);
            let pair = make_fragment::<F4E2M1FN>(8);
            for i in 0..4 {
                pair.store(2 * i, p);
                pair.store(2 * i + 1, q);
            }
            autovec_copy(&pair, &local_tile(&out, 8, t));
            let _ = line;
        })
    };
    // The FP4 conversion: sm_100a or sm_101a (PTX 8.6), sm_110a (9.0),
    // sm_120a (8.7), and the sm_100f, sm_110f and sm_120f families (8.8).
    let msg = build_panic(|| {
        fp4(Arch::Sm90a);
    });
    assert!(
        msg.contains("kernel `fp4_pair` is built for sm_90a (KernelOptions::target), which it cannot run on")
            && msg.contains("tests/diagnostics/types_and_targets.rs:")
            && msg.contains("needs sm_100a, sm_101a, sm_110a or sm_120a (or the sm_100f family with PTX 8.8"),
        "{msg}"
    );
    for (arch, ptx) in
        [(Arch::Sm100a, 86), (Arch::Sm103f, 88), (Arch::Sm110a, 90), (Arch::Sm121a, 88), (Arch::Sm120a, 87)]
    {
        let k = fp4(arch);
        assert_eq!((k.target(), check_target(k.ir(), arch).unwrap()), (Some(arch), ptx), "{arch}");
    }
    for arch in [Arch::Sm100, Arch::Sm103, Arch::Sm107, Arch::Sm120] {
        assert!(build_panic(|| drop(fp4(arch))).contains("which it cannot run on"), "{arch}");
    }
    // Compiling: the kernel's target by default, another one refused.
    if let Some(tc) = Toolchain::discover() {
        let k = fp4(Arch::Sm100a);
        let e = k.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm120a)).unwrap_err().to_string();
        assert!(
            e.contains("was built for sm_100a (KernelOptions::target), so it cannot be compiled for sm_120a"),
            "{e}"
        );
        let compiled = k.compile_with(&tc, &CompileOptions::new()).unwrap();
        assert_eq!((compiled.arch(), compiled.ptx_version()), (Arch::Sm100a, 86));
        // Without a target, nothing names one.
        let e = Kernel::build("untargeted", || {}).compile_with(&tc, &CompileOptions::new()).unwrap_err();
        assert!(e.to_string().contains("KernelOptions::target"), "{e}");
        // Rubin: no LLVM emits sm_107a yet, so compiling fails clearly, though
        // the kernel builds and checks for it.
        let k = Kernel::builder("rubin").target(Arch::Sm107a).build(|| {});
        assert_eq!(check_target(k.ir(), Arch::Sm107a).unwrap(), 94);
        if tc.targets().is_some_and(|s| !s.has("sm_107a")) {
            let e = k.compile_with(&tc, &CompileOptions::new()).unwrap_err().to_string();
            assert!(e.contains("cannot emit sm_107a: its NVPTX backend") && e.contains("no such target"), "{e}");
        }
    }
    // Outside a target, target_arch reads CUTE_RS_ARCH (unset here).
    if std::env::var_os("CUTE_RS_ARCH").is_none() {
        Kernel::build("no_target", || assert_eq!(target_arch(), None));
    }
}

/// A kernel that needs more PTX than the toolchain emits fails to compile,
/// naming the version, the toolchain's newest and the instruction that
/// needs it, at its line.
#[test]
fn ptx_ceiling_of_the_toolchain() {
    let Some(tc) = Toolchain::discover() else { return };
    let Some(max) = tc.targets().map(|s| s.max_ptx) else { return };
    let line = line!() + 4;
    let kernel = Kernel::build("s2f6", || {
        let out = param_tensor::<BF16>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let v = cvt_i8_bf16_intrinsic_with(&[t.to::<i8>(), (t + 1).to::<i8>()], 2, I8ToBf16Path::S2f6x2);
        out.store(2 * t, v[0]);
        out.store(2 * t + 1, v[1]);
    });
    let result = kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a));
    if max >= 91 {
        assert!(result.is_ok(), "{:?}", result.err());
        return;
    }
    let e = result.unwrap_err().to_string();
    assert!(
        e.contains("needs PTX ISA 9.1 on sm_100a")
            && e.contains(&format!("emits PTX ISA {}.{} at most", max / 10, max % 10))
            && e.contains(&format!("tests/diagnostics/types_and_targets.rs:{line}:"))
            && e.contains("s2f6x2"),
        "{e}"
    );
}

cute_struct! {
    struct EmptyStruct {}
}

cute_union! {
    struct EmptyUnion {}
}

cute_struct! {
    struct OddAlign { #[align(48)] x: f32 }
}

cute_struct! {
    struct HugeAlign { #[align(2048)] x: MemRange<f32>[4] }
}

cute_struct! {
    struct Ranges { head: u32, r: MemRange<f32>[4], marker: MemRange<u64>[0], #[align(16)] h: MemRange<F16>[512] }
}

/// The shared-storage struct mistakes CuTe DSL refuses, each refused when
/// the kernel is built with the member and the rule.
#[test]
fn struct_misuse() {
    let e = build_panic(|| {
        EmptyStruct::layout();
    });
    assert!(e.contains("struct EmptyStruct has no members"), "{e}");
    let e = build_panic(|| {
        EmptyUnion::layout();
    });
    assert!(e.contains("union EmptyUnion has no members"), "{e}");
    let e = build_panic(|| {
        OddAlign::layout();
    });
    assert!(e.contains("member `x` of OddAlign: an alignment of 48 bytes is not a power of two"), "{e}");
    let e = build_panic(|| {
        HugeAlign::layout();
    });
    assert!(e.contains("an alignment of 2048 bytes is more than shared memory's 1024"), "{e}");
    let e = build_panic(|| {
        StructLayout::builder("Nibbles").mem_range::<I4>("odd", 3, None);
    });
    assert!(e.contains("MemRange[Int4, 3] is 12 bits, not a whole number of bytes"), "{e}");
    let e = build_panic(|| {
        let mut b = StructLayout::builder("Twice");
        b.scalar::<f32>("x", None);
        b.scalar::<i32>("x", None);
    });
    assert!(e.contains("Twice has two members named `x`"), "{e}");

    let in_kernel = |f: fn(&Ranges)| {
        build_panic(move || {
            let _ = Kernel::builder("struct_misuse").block_dim([32, 1, 1]).build(|| f(&Ranges::allocate()));
        })
    };
    let e = in_kernel(|s| {
        s.marker.get_tensor(Layout::col_major(1));
    });
    assert!(e.contains("get_tensor on MemRange `marker` of length 0"), "{e}");
    let e = in_kernel(|s| {
        s.r.store(4, 1.0f32);
    });
    assert!(e.contains("a store of element 4 of MemRange `r`, which has 4 elements"), "{e}");
    let e = in_kernel(|s| {
        s.r.get_tensor(Layout::col_major((2, 3)));
    });
    assert!(e.contains("get_tensor on MemRange `r`: the layout needs 6 elements, but the range has 4"), "{e}");
    let e = in_kernel(|s| {
        s.h.get_tensor_composed(make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((16, 64))));
    });
    assert!(
        e.contains("get_tensor_composed on MemRange `h`: the layout needs 1024 elements, but the range has 512"),
        "{e}"
    );
    // A swizzled composed layout needs its range at a multiple of the swizzle's period.
    let e = in_kernel(|s| {
        s.h.get_tensor_composed(make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64))));
    });
    assert!(e.contains("needs to be a multiple of 512 elements (the swizzle's period) into its buffer"), "{e}");
    let e = in_kernel(|s| {
        s.h.get_tensor_swizzled(Layout::row_major((8, 64)), make_swizzle(3, 4, 3));
    });
    assert!(e.contains("the range must start at a known multiple of its period of 1024 bytes"), "{e}");
    let e = build_panic(|| {
        let _ = Kernel::builder("struct_misuse").block_dim([32, 1, 1]).build(|| {
            let _ = Ranges::allocate_aligned(4096);
        });
    });
    assert!(e.contains("allocating Ranges: an alignment of 4096 bytes is more than shared memory's 1024"), "{e}");
}

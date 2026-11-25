//! The driver's tensor-map encoders against cute-rs's validator for the im2col descriptors
//! (`cuTensorMapEncodeIm2col`) and the FP4/FP6 encodings
//! (`cuTensorMapEncodeTiled` with `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`, `16U4_ALIGN16B`,
//! `16U6_ALIGN16B`). Below compute capability 9.0 the driver encodes
//! nothing but checks some fields first: what it rejects as invalid, cute-rs must reject
//! too. Needs the `cuda` feature:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_tma_modes -- --nocapture
//! ```

use cute::cuda::{CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED, CudaContext};
use cute::tma::{Im2colParams, TensorMapDataType, TensorMapSwizzle, TmaDesc};

fn device() -> Option<CudaContext> {
    CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()
}

/// An NHWC f16 im2col descriptor over `base`, valid, and variants breaking one rule each.
fn im2col_cases(base: u64) -> Vec<(&'static str, TmaDesc)> {
    let mut ok = TmaDesc::new(TensorMapDataType::float16, 4);
    ok.global_address = base;
    ok.global_dims[..4].copy_from_slice(&[32, 6, 5, 2]);
    ok.global_strides[..3].copy_from_slice(&[64, 384, 1920]);
    ok.swizzle = TensorMapSwizzle::s64b;
    ok.im2col = Some(Im2colParams {
        lower_corner: [-1, -1, 0],
        upper_corner: [-1, -1, 0],
        channels_per_pixel: 32,
        pixels_per_column: 64,
    });
    let mut out = vec![("valid NHWC f16", ok.clone())];
    let mut push = |name, f: &dyn Fn(&mut TmaDesc)| {
        let mut d = ok.clone();
        f(&mut d);
        out.push((name, d));
    };
    let set = |d: &mut TmaDesc, f: &dyn Fn(&mut Im2colParams)| f(d.im2col.as_mut().unwrap());
    push("valid traversal stride 2", &|d| d.element_strides[1] = 2);
    push("valid corners -128 and 127", &|d| {
        set(d, &|p| {
            p.lower_corner[0] = -128;
            p.upper_corner[1] = 127;
        })
    });
    push("valid 1024 pixels", &|d| set(d, &|p| p.pixels_per_column = 1024));
    push("corner -129 at rank 4", &|d| set(d, &|p| p.lower_corner[0] = -129));
    push("corner 128 at rank 4", &|d| set(d, &|p| p.upper_corner[1] = 128));
    push("0 channels", &|d| set(d, &|p| p.channels_per_pixel = 0));
    push("257 channels", &|d| set(d, &|p| p.channels_per_pixel = 257));
    push("0 pixels", &|d| set(d, &|p| p.pixels_per_column = 0));
    push("1025 pixels", &|d| set(d, &|p| p.pixels_per_column = 1025));
    push("rank 2", &|d| d.rank = 2);
    push("rank 6", &|d| d.rank = 6);
    push("misaligned base", &|d| d.global_address += 8);
    push("stride 40 bytes", &|d| d.global_strides[0] = 40);
    push("element stride 9", &|d| d.element_strides[1] = 9);
    push("channels 128 bytes with s64b", &|d| set(d, &|p| p.channels_per_pixel = 64));
    // Rank 5: corners are 5-bit.
    let mut r5 = ok.clone();
    r5.rank = 5;
    r5.global_dims[..5].copy_from_slice(&[32, 6, 5, 4, 2]);
    r5.global_strides[..4].copy_from_slice(&[64, 384, 1920, 7680]);
    out.push(("valid NDHWC", r5.clone()));
    let mut d = r5.clone();
    d.im2col.as_mut().unwrap().lower_corner[2] = -17;
    out.push(("corner -17 at rank 5", d));
    out
}

/// The sub-byte tiled encodings: valid, and breaking the padded formats' rules.
fn subbyte_cases(base: u64) -> Vec<(&'static str, TmaDesc)> {
    let tiled = |t: TensorMapDataType, inner: u64, stride: u64, box0: u32| {
        let mut d = TmaDesc::new(t, 2);
        d.global_address = base;
        d.global_dims[..2].copy_from_slice(&[inner, 8]);
        d.global_strides[0] = stride;
        d.box_dims[..2].copy_from_slice(&[box0, 4]);
        d
    };
    use TensorMapDataType::*;
    vec![
        ("valid 16U4_ALIGN8B", tiled(f416u4_align8b, 256, 128, 64)),
        ("valid 16U4_ALIGN16B", tiled(f416u4_align16b, 256, 128, 128)),
        ("valid 16U6_ALIGN16B", tiled(f416u6_align16b, 256, 192, 128)),
        ("16U4_ALIGN16B box 64", tiled(f416u4_align16b, 256, 128, 64)),
        ("16U6_ALIGN16B box 96", tiled(f416u6_align16b, 256, 192, 96)),
        ("16U4_ALIGN16B stride 144", tiled(f416u4_align16b, 256, 144, 128)),
    ]
}

#[test]
fn driver_encoding_agrees_with_the_validator() {
    let Some(ctx) = device() else { return };
    if !ctx.has_tensor_map_encode() || !ctx.has_tensor_map_encode_im2col() {
        eprintln!("skipping: the driver lacks the tensor-map encoders");
        return;
    }
    let buf = ctx.alloc_bytes(1 << 20).unwrap();
    let base = buf.as_raw();
    let (major, minor) = ctx.compute_capability();
    let full = major >= 9;
    let mut disagree = Vec::new();
    for (name, d) in im2col_cases(base).into_iter().chain(subbyte_cases(base)) {
        let ours = d.validate();
        let driver = ctx.encode_tensor_map_unchecked(&d);
        let code = driver.as_ref().err().and_then(|e| e.code);
        println!(
            "sm_{major}{minor} {name:40} cute-rs {:8} driver {}",
            if ours.is_ok() { "accepts" } else { "rejects" },
            match &driver {
                Ok(_) => "accepts".to_string(),
                Err(e) => format!("rejects ({:?})", e.code),
            }
        );
        let bad = if full {
            ours.is_ok() != driver.is_ok()
        } else {
            driver.is_ok()
                || code == Some(CUDA_ERROR_INVALID_VALUE) && ours.is_ok()
                || !matches!(code, Some(CUDA_ERROR_INVALID_VALUE | CUDA_ERROR_NOT_SUPPORTED))
        };
        if bad {
            disagree.push(format!("{name}: cute-rs {ours:?}, driver {:?}", driver.map(|_| ()).map_err(|e| e.message)));
        }
    }
    assert!(disagree.is_empty(), "the driver and the validator disagree:\n{}", disagree.join("\n"));
}

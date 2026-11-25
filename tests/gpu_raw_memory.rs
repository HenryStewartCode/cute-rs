//! Host-side reads and writes of raw device addresses on a real GPU. Needs the `cuda`
//! feature; skipped (with a note) without an NVIDIA GPU:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_raw_memory
//! ```

use cute::cuda::{CudaContext, DevicePtr};
use cute::prelude::*;

fn device() -> Option<CudaContext> {
    CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()
}

fn raw_f32(address: u64, bytes: Option<usize>) -> DevicePtr {
    // SAFETY: the tests pass addresses of live buffers, or deliberately bad ones to `read_raw`
    // and `write_raw`, which report the driver's error.
    unsafe { DevicePtr::from_raw(DType::F32, address, None, bytes).unwrap() }
}

#[test]
fn raw_write_then_buffer_read_and_the_reverse() {
    let Some(ctx) = device() else { return };
    let buffer = ctx.alloc::<f32>(8).unwrap();
    let ptr = raw_f32(buffer.as_raw(), Some(buffer.bytes()));
    let values: Vec<f32> = (0..8).map(|i| i as f32 * 1.5).collect();
    // SAFETY: the address is the live buffer's.
    unsafe { ctx.write_raw(&ptr, 0, &values).unwrap() };
    assert_eq!(buffer.to_vec::<f32>().unwrap(), values);

    let other: Vec<f32> = (0..8).map(|i| -(i as f32)).collect();
    buffer.write(0, &other).unwrap();
    // SAFETY: as above.
    assert_eq!(unsafe { ctx.read_raw::<f32>(&ptr, 0, 8).unwrap() }, other);
    // With an offset.
    assert_eq!(unsafe { ctx.read_raw::<f32>(&ptr, 16, 2).unwrap() }, other[4..6]);
}

#[test]
fn raw_async_copies_on_a_stream() {
    let Some(ctx) = device() else { return };
    let stream = ctx.stream().unwrap();
    let buffer = ctx.alloc::<f32>(4).unwrap();
    let ptr = raw_f32(buffer.as_raw(), None);
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut dst = [0.0f32; 4];
    // SAFETY: the host arrays outlive the synchronisation below.
    unsafe {
        stream.write_raw_async(&ptr, 0, &src).unwrap();
        stream.read_raw_async(&ptr, 0, &mut dst).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(dst, src);
}

#[test]
fn raw_access_is_checked_before_the_driver() {
    let Some(ctx) = device() else { return };
    let buffer = ctx.alloc::<f32>(4).unwrap();
    let bounded = raw_f32(buffer.as_raw(), Some(16));
    // SAFETY: every call below is refused before reaching the driver.
    unsafe {
        let e = ctx.read_raw::<f32>(&bounded, 0, 5).unwrap_err();
        assert!(e.message.contains("outside the 16 bytes"), "{e}");
        assert!(e.message.contains(&format!("{:#x}", buffer.as_raw())), "{e}");
        assert!(ctx.write_raw(&bounded, 8, &[0.0f32; 3]).is_err());
        let misaligned = ctx.read_raw::<f32>(&bounded, 2, 1).unwrap_err();
        assert!(misaligned.message.contains("aligned"), "{misaligned}");
        let null = DevicePtr::null(DType::F32);
        assert!(ctx.read_raw::<f32>(&null, 0, 1).unwrap_err().message.contains("null"));
    }
}

#[test]
fn an_unmapped_address_is_an_error_not_a_crash() {
    let Some(ctx) = device() else { return };
    let ptr = raw_f32(0x10, None);
    // SAFETY: the address is unmapped on purpose; the driver rejects the copy.
    unsafe {
        let e = ctx.read_raw::<f32>(&ptr, 0, 4).unwrap_err();
        assert!(e.message.contains("0x10"), "{e}");
        assert!(e.code.is_some(), "{e}");
        assert!(ctx.write_raw(&ptr, 0, &[1.0f32]).unwrap_err().message.contains("0x10"));
    }
    // The context still works.
    let buffer = ctx.upload_slice(&[7.0f32]).unwrap();
    assert_eq!(buffer.to_vec::<f32>().unwrap(), [7.0]);
}

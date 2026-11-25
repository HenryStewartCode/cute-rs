//! Sub-byte tensors: Int4 in global, shared and register memory, moved in
//! packed 32-bit words (vectorised loads and stores, tiled copies,
//! `copy_if`), and `recast_tensor` between bytes and Int4 and between Float32
//! and Int32, in memory and in registers.

use cute::prelude::*;

use super::{Case, T, launch};

/// Int4 values per row: one 128-bit vector per thread.
const ROW: usize = 32;

/// A thread's row of `t` (a `(T, ROW)` row-major tensor).
fn row<E>(t: &Tensor<E>, thread: Int32) -> Tensor<E> {
    t.slice((thread, ..))
}

/// Int4 copies: global to shared memory by a 128-bit tiled copy, shared
/// memory to registers and back to global memory by `autovec_copy` (128-bit
/// vectors), register arithmetic in between (`x + 1`, wrapping at 4 bits),
/// a `copy_if` of 32-bit words under a predicate, and 64-bit vectors through
/// a `make_rmem_tensor`.
pub fn int4_copies() -> Case {
    let kernel = Kernel::builder("int4_copies").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<I4>("src", Layout::row_major((T, ROW)));
        let incremented = param_tensor::<I4>("incremented", Layout::row_major((T, ROW)));
        let masked = param_tensor::<I4>("masked", Layout::row_major((T, ROW)));
        let halves = param_tensor::<I4>("halves", Layout::row_major((T, ROW)));
        let t = thread_idx().x;
        let smem = make_smem_tensor::<I4>(Layout::row_major((T, ROW)));
        let g2s = make_tiled_copy_tv(
            &make_copy_atom::<I4>(CopyOp::Universal { bits: 128 }),
            &Layout::row_major((T, 1)),
            &Layout::row_major((1, ROW)),
        );
        copy(&g2s, &g2s.get_slice(t).partition_s(&src), &g2s.get_slice(t).partition_d(&smem));
        sync_threads();
        // Thread t reads row (t + 1) % T, which another thread wrote.
        let frag = make_fragment::<I4>(ROW);
        autovec_copy(&row(&smem, (t + 1) % T as i32), &frag);
        frag.map_in_place(|x| x + I4::new(1));
        autovec_copy(&frag, &row(&incremented, t));
        // Every other 8-value word, by a predicate on its first value.
        let words = make_copy_atom::<I4>(CopyOp::Universal { bits: 32 });
        let pred = make_fragment::<bool>(ROW);
        for i in 0..ROW {
            pred.store(i, ((t + (i / 8) as i32) % 2).eq(0));
        }
        let zeros = make_fragment::<I4>(ROW);
        zeros.fill(I4::ZERO);
        autovec_copy(&zeros, &row(&masked, t));
        copy_if(&words, &pred, &frag, &row(&masked, t));
        // 64-bit vectors through per-thread memory.
        let stack = make_rmem_tensor::<I4>(Layout::col_major(ROW));
        copy(&make_copy_atom::<I4>(CopyOp::Universal { bits: 64 }), &row(&src, t), &stack);
        let back = make_fragment::<I4>(ROW);
        autovec_copy(&stack, &back);
        back.map_in_place(|x| x >> I4::new(1));
        autovec_copy(&back, &row(&halves, t));
    });
    fn check(args: &[KernelArg]) {
        let src = args[0].to_vec::<I4>();
        let (inc, masked, halves) = (args[1].to_vec::<I4>(), args[2].to_vec::<I4>(), args[3].to_vec::<I4>());
        for t in 0..T {
            for i in 0..ROW {
                let from = src[((t + 1) % T) * ROW + i];
                let want = I4::wrapping(from.get() as i64 + 1);
                assert_eq!(inc[t * ROW + i], want, "incremented[{t}][{i}]");
                let copied = (t + i / 8).is_multiple_of(2);
                let m = masked[t * ROW + i];
                if copied {
                    assert_eq!(m, want, "masked[{t}][{i}]");
                } else {
                    assert_eq!(m, I4::ZERO, "masked[{t}][{i}] (predicated off)");
                }
                assert_eq!(halves[t * ROW + i], I4::new(src[t * ROW + i].get() >> 1), "halves[{t}][{i}]");
            }
        }
    }
    Case { name: "int4_copies", kernel, cfg: launch(1), small_only: false, check }
}

/// `recast_tensor`: a byte tensor in global memory viewed as Int4 (twice the
/// elements, element 0 in the low nibble), an Int4 register fragment packed
/// into bytes, a shared-memory Float32 tensor viewed as Int32 bits, a
/// Float32 fragment viewed as Int32 and back, and an Int4 fragment unpacked
/// from Int32 words.
pub fn subbyte_recasts() -> Case {
    let kernel = Kernel::builder("subbyte_recasts").block_dim([T as u32, 1, 1]).build(|| {
        let bytes = param_tensor::<u8>("bytes", Layout::row_major((T, ROW / 2)));
        let nibbles = param_tensor::<i8>("nibbles", Layout::row_major((T, ROW)));
        let repacked = param_tensor::<u8>("repacked", Layout::row_major((T, ROW / 2)));
        let floats = param_tensor::<f32>("floats", Layout::row_major((T, 4)));
        let float_bits = param_tensor::<i32>("float_bits", Layout::row_major((T, 4)));
        let round_trip = param_tensor::<f32>("round_trip", Layout::row_major((T, 4)));
        let smem_bits = param_tensor::<i32>("smem_bits", Layout::row_major((T, 4)));
        let rewords = param_tensor::<i8>("rewords", Layout::row_major((T, ROW)));
        let t = thread_idx().x;
        // Bytes as Int4, in global memory.
        let as_i4 = recast_tensor::<I4, _>(&bytes);
        assert_eq!(as_i4.size(), T * ROW);
        let frag = make_fragment::<I4>(ROW);
        autovec_copy(&row(&as_i4, t), &frag);
        for i in 0..ROW {
            nibbles.store((t, i as i32), frag.load(i).to::<i8>());
        }
        // Int4 registers packed into bytes, and Int32 words unpacked into Int4.
        let packed = frag.recast::<u8>();
        assert_eq!(packed.size(), ROW / 2);
        autovec_copy(&packed, &row(&repacked, t));
        let words = packed.recast::<i32>().recast::<I4>();
        assert_eq!(words.size(), ROW);
        for i in 0..ROW {
            rewords.store((t, i as i32), words.load(i).to::<i8>());
        }
        // Float32 bits as Int32: in registers and in shared memory.
        let f = make_fragment::<f32>(4);
        autovec_copy(&row(&floats, t), &f);
        let bits = f.recast::<i32>();
        autovec_copy(&bits, &row(&float_bits, t));
        autovec_copy(&bits.recast::<f32>(), &row(&round_trip, t));
        let s = make_smem_tensor::<f32>(Layout::row_major((T, 4)));
        autovec_copy(&f, &row(&s, t));
        sync_threads();
        let s_bits = recast_tensor::<i32, _>(&s);
        let other = make_fragment::<i32>(4);
        autovec_copy(&row(&s_bits, (t + 1) % T as i32), &other);
        autovec_copy(&other, &row(&smem_bits, t));
    });
    fn check(args: &[KernelArg]) {
        let bytes = args[0].to_vec::<u8>();
        let (nibbles, repacked) = (args[1].to_vec::<i8>(), args[2].to_vec::<u8>());
        let floats = args[3].to_vec::<f32>();
        let (float_bits, round_trip, smem_bits) =
            (args[4].to_vec::<i32>(), args[5].to_vec::<f32>(), args[6].to_vec::<i32>());
        assert_eq!(args[7].to_vec::<i8>(), nibbles);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(nibbles[2 * i], ((b << 4) as i8) >> 4, "low nibble of byte {i}");
            assert_eq!(nibbles[2 * i + 1], (b as i8) >> 4, "high nibble of byte {i}");
        }
        assert_eq!(repacked, bytes);
        for t in 0..T {
            for i in 0..4 {
                let k = t * 4 + i;
                assert_eq!(float_bits[k], floats[k].to_bits() as i32);
                assert_eq!(round_trip[k].to_bits(), floats[k].to_bits());
                assert_eq!(smem_bits[k], floats[((t + 1) % T) * 4 + i].to_bits() as i32);
            }
        }
    }
    Case { name: "subbyte_recasts", kernel, cfg: launch(1), small_only: false, check }
}

/// Every sub-byte tensor kernel.
pub fn subbyte_tensors() -> Vec<Case> {
    vec![int4_copies(), subbyte_recasts()]
}

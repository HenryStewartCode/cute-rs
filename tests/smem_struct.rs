//! Shared-storage structs (CuTe DSL's `@cute.struct`, `MemRange`, `Align`
//! and `@cute.union`): the layout rules on the shapes of CuTe DSL's
//! documented examples, worked out by hand, the builder API, and the field
//! table in `Kernel::dump`. The kernels that use them are feature kernels
//! (`tests/feature_kernels/structs.rs`).

use cute::prelude::*;

cute_struct! {
    pub struct Complex { real: f32, imag: f32 }
}

cute_struct! {
    pub struct StorageA(stage: usize) {
        mbar_a: MemRange<i64>[stage],
        comp_a: Complex(),
        int_a: i16,
    }
}

cute_struct! {
    pub struct StorageB(size_a: usize, size_b: usize) {
        #[align(1024)]
        a: MemRange<f32>[size_a],
        #[align(1024)]
        b: MemRange<f32>[size_b],
        #[align(16)]
        x: i32,
        #[align(16)]
        comp_a: Complex(),
    }
}

cute_union! {
    pub struct Aliased {
        halves: MemRange<F16>[4096],
        words: MemRange<f32>[1024],
        #[align(16)]
        tag: u64,
    }
}

cute_struct! {
    pub struct Marked {
        a: u8,
        here: MemRange<u64>[0],
        b: u32,
    }
}

cute_struct! {
    pub struct Outer(stages: usize) {
        flag: bool,
        flags: MemRange<bool>[5],
        inner: StorageA(stages),
        scratch: Aliased(),
        nibbles: MemRange<I4>[6],
    }
}

cute_struct! {
    pub struct Typed<AB>(n: usize) {
        #[align(128)]
        a: MemRange<AB>[n],
        scale: AB,
    }
}

#[test]
fn complex() {
    let l = Complex::layout();
    assert_eq!((l.size_of(), l.align_of(), l.size_in_bytes()), (8, 4, 8));
    assert_eq!(l.field_table(), ["real:4:0", "imag:4:4"]);
    assert!(!l.is_union());
}

/// `mbar_a` (stage Int64s at 0), the 4-byte-aligned complex after it, then
/// the Int16; the size rounds up to the Int64's alignment.
#[test]
fn storage_a() {
    for (stage, table, size) in
        [(1, ["mbar_a:8:0", "comp_a:8:8", "int_a:2:16"], 24), (2, ["mbar_a:16:0", "comp_a:8:16", "int_a:2:24"], 32)]
    {
        let l = StorageA::layout(stage);
        assert_eq!(l.field_table(), table);
        assert_eq!((l.size_of(), l.align_of()), (size, 8));
    }
    assert_eq!(StorageA::size_of(3), 40);
}

/// Each `Align[.., 1024]` range starts a new 1024-byte block; `x` and the
/// complex go to the next multiples of 16; the struct is 1024-aligned.
#[test]
fn storage_b() {
    let l = StorageB::layout(100, 50);
    assert_eq!(l.field_table(), ["a:400:0", "b:200:1024", "x:4:1232", "comp_a:8:1248"]);
    assert_eq!((l.size_of(), l.align_of()), (2048, 1024));
    let m = l.member("comp_a");
    assert_eq!((m.offset, m.size, m.align), (1248, 8, 16));
    assert_eq!(l.member("x").align, 16);
    // A range of exactly 1024 elements fills its block.
    assert_eq!(StorageB::layout(256, 256).field_table(), ["a:1024:0", "b:1024:1024", "x:4:2048", "comp_a:8:2064"]);
    assert_eq!(StorageB::size_of(256, 256), 3072);
}

#[test]
fn union_of_ranges() {
    let l = Aliased::layout();
    assert!(l.is_union());
    assert_eq!(l.field_table(), ["halves:8192:0", "words:4096:0", "tag:8:0"]);
    assert_eq!((l.size_of(), l.align_of()), (8192, 16));
}

/// A zero-size range marks an address: it takes no bytes, but is aligned.
#[test]
fn zero_size_marker() {
    let l = Marked::layout();
    assert_eq!(l.field_table(), ["a:1:0", "here:0:8", "b:4:8"]);
    assert_eq!((l.size_of(), l.align_of()), (16, 8));
}

/// Booleans take a byte each (a MemRange of them too), nested structs and
/// unions their own size and alignment, and Int4 ranges half a byte per
/// element.
#[test]
fn nesting_and_booleans() {
    let l = Outer::layout(2);
    assert_eq!(l.field_table(), ["flag:1:0", "flags:5:1", "inner:32:8", "scratch:8192:48", "nibbles:3:8240"]);
    assert_eq!((l.size_of(), l.align_of()), (8256, 16));
    assert_eq!(l.member("inner").nested_layout().field_table(), StorageA::layout(2).field_table());
    match &l.member("flags").kind {
        MemberKind::MemRange { elem, len } => assert_eq!((*elem, *len), (DType::Bool, 5)),
        other => panic!("{other:?}"),
    }
}

#[test]
fn type_parameters() {
    assert_eq!(Typed::<F16>::layout(100).field_table(), ["a:200:0", "scale:2:200"]);
    assert_eq!(Typed::<f64>::layout(3).field_table(), ["a:24:0", "scale:8:24"]);
    assert_eq!(Typed::<F8E4M3FN>::align_of(10), 128);
}

/// The builder API that the macros expand to, used directly.
#[test]
fn builder_api() {
    let mut b = StructLayout::builder("Storage");
    let full = b.mem_range::<u64>("full", 4, None);
    let data = b.mem_range::<f32>("data", 64, Some(256));
    let count = b.scalar::<i32>("count", None);
    let l = b.build();
    assert_eq!((full.offset, data.offset, count.offset), (0, 256, 512));
    assert_eq!((l.size_of(), l.align_of()), (768, 256));
    assert_eq!(l.to_string(), "struct Storage (size 768, align 256) { full:32:0, data:256:256, count:4:512 }");

    let mut u = StructLayout::union_builder("U");
    let _ = u.nested("storage", &l, None);
    let _ = u.mem_range::<u8>("bytes", 1000, None);
    let u = u.build();
    assert_eq!((u.size_of(), u.align_of()), (1024, 256));

    let kernel = Kernel::builder("builder_api").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(2));
        let base = l.allocate();
        let (d, c) = (data.at(&base), count.at(&base));
        if_(thread_idx().x.eq(0), || {
            c.store(7);
            d.store(63, c.load().to::<f32>() * 2.0);
            out.store(0, d.load(63));
            out.store(1, d.get_tensor(Layout::col_major(64)).load(63));
        });
    });
    let mut args = vec![KernelArg::f32(vec![0.0; 2])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].as_f32(), [14.0, 14.0]);
    let dump = kernel.dump();
    assert!(dump.contains("shared smem0: 768 x u8 (align 256)"), "{dump}");
    assert!(dump.contains("    struct Storage { full:32:0, data:256:256, count:4:512 }"), "{dump}");
}

/// `SmemAllocator.allocate(Struct, byte_alignment)`: the larger of the two
/// alignments.
#[test]
fn allocation_alignment() {
    let kernel = Kernel::builder("allocation_alignment").block_dim([32, 1, 1]).build(|| {
        let _ = Complex::allocate_aligned(512);
        let _ = StorageB::allocate_aligned(16, 4, 4);
        let _ = Marked::allocate();
    });
    let dump = kernel.dump();
    assert!(dump.contains("shared smem0: 8 x u8 (align 512)"), "{dump}");
    assert!(dump.contains("shared smem1: 2048 x u8 (align 1024)"), "{dump}");
    assert!(dump.contains("shared smem2: 16 x u8 (align 8)"), "{dump}");
}

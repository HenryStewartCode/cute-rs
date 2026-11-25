//! Addresses of shared buffers in the dynamic shared-memory window (kernels
//! over 48 KB, with a dynamic region or with overlays), compiled with every
//! installed `mlir-opt`. LLVM 21's `--expand-strided-metadata` folds
//! `memref.extract_aligned_pointer_as_index` of a `memref.view` into the same
//! op on the view's source, dropping the view's byte offset; the emitter
//! therefore adds each buffer's start to the window's own pointer. Skipped
//! (with a note) when no `mlir-opt` is installed.

use std::path::{Path, PathBuf};

use cute::prelude::*;

/// Every `mlir-opt` this machine has: `$CUTE_MLIR_OPT`, the Homebrew kegs
/// (LLVM 21 and the current one), the pod's LLVM 22, and any on `PATH`. Deduplicated by canonical path.
fn every_mlir_opt() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut candidates: Vec<PathBuf> = std::env::var_os("CUTE_MLIR_OPT").map(PathBuf::from).into_iter().collect();
    let path_dirs: Vec<PathBuf> =
        std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    candidates.extend(path_dirs.iter().map(|d| d.join("mlir-opt")));
    for dir in [
        "/opt/homebrew/opt/llvm@21/bin",
        "/opt/homebrew/opt/llvm@22/bin",
        "/opt/homebrew/opt/llvm@23/bin",
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm@21/bin",
        "/usr/local/opt/llvm/bin",
        "/workspace/llvm-22/bin",
        "/usr/lib/llvm-21/bin",
        "/usr/lib/llvm-22/bin",
        "/usr/lib/llvm-23/bin",
    ] {
        candidates.push(Path::new(dir).join("mlir-opt"));
    }
    for path in candidates {
        let Ok(canonical) = path.canonicalize() else { continue };
        if !found.iter().any(|f| f.canonicalize().ok().as_ref() == Some(&canonical)) {
            found.push(path);
        }
    }
    found
}

/// A 64 KB buffer (past the 48 KB of static shared memory, so the window
/// holds every buffer), then an mbarrier at byte 65536, whose address the
/// kernel initializes, takes as an integer and stores.
fn kernel_with_buffer_after_64k() -> Kernel {
    Kernel::builder("mbarrier_after_64k").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major((1,)));
        let big = alloc_smem::<f32>(16384, 16);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        big.offset(t).store(t.to::<f32>());
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        if_(t.eq(0), || out.store(0, mbar.toint_i32()));
    })
}

#[test]
fn window_buffers_keep_their_start_with_every_mlir_opt() {
    let kernel = kernel_with_buffer_after_64k();
    let layout = kernel.ir().shared_memory_layout();
    assert!(layout.in_window, "the kernel must use the dynamic window");
    assert_eq!(layout.starts.iter().flatten().copied().collect::<Vec<_>>(), vec![0, 65536]);
    let mlir = kernel.to_mlir();
    // The address is the window's pointer plus the start, not a view's pointer.
    assert!(mlir.contains("arith.constant 65536 : i64"), "{mlir}");
    let tools = every_mlir_opt();
    if tools.is_empty() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    }
    for path in tools {
        let tc = Toolchain::new(&path);
        let ptx = kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        // The mbarrier's address: the window symbol plus 65536, whether folded into the
        // operand (`[sym+65536]`) or added to the symbol's address.
        let init = ptx.lines().find(|l| l.contains("mbarrier.init")).unwrap_or_else(|| panic!("{ptx}"));
        let at_offset = |l: &str| l.contains("+65536]") || l.contains("65536;");
        let shown = init.contains("+65536]")
            || ptx.lines().any(|l| l.contains("__cute_dynamic_smem") && at_offset(l))
            || ptx.lines().any(|l| (l.contains("add.s64") || l.contains("add.s32")) && at_offset(l));
        assert!(shown, "{} ({:?}): the mbarrier's address lost its start\n{ptx}", path.display(), tc.llvm_version());
        assert!(!init.contains("[__cute_dynamic_smem]"), "{}: mbarrier at the window's start\n{ptx}", path.display());
    }
}

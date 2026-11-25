#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Validate cute-rs on the GPU this runs on, in one unattended pass, for short rentals of
# hardware the tests otherwise never see (an H100 for sm_90, a B200 for sm_100).
#
#   bash scripts/gpu-validate.sh [--quick] [--no-bench] [--no-full]
#
# Run it from the repository root, on Linux with an NVIDIA driver. It
#   1. installs what is missing: Rust (rustup, into ~/.cargo) and mlir-opt (LLVM 22 from
#      apt.llvm.org, unless CUTE_MLIR_OPT or /workspace/llvm-22 provides one; LLVM_VERSION=23 picks
#      another LLVM);
#   2. builds the tests once, in release, with the `cuda` feature;
#   3. runs the architecture's own suite first (tests/gpu_hopper.rs on sm_90,
#      tests/gpu_blackwell.rs on sm_10x) with CUTE_RS_EXPECT_SM set, so that those tests fail
#      rather than skip if the device or toolchain is not what was expected;
#   4. runs the whole test suite (all GPU tests, plus the simulator and PTX tests);
#   5. runs the timings (the `bench_*` tests) and, when a Python with PyTorch is already
#      installed, cuBLAS on the same shapes for comparison;
#   6. writes everything to gpu-results/<gpu>-<time>/ and packs it as a .tar.gz beside it.
#
# --quick runs only steps 1-3 and the timings; --no-full skips step 4; --no-bench step 5.
# Everything it builds stays on this machine's local disk (target/ and the JIT cache under
# $HOME), so it is safe to run on a fresh pod that will be thrown away.

set -uo pipefail

QUICK=0 BENCH=1 FULL=1
LLVM_VERSION=${LLVM_VERSION:-22} # the LLVM whose mlir-opt is installed when none is found
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 FULL=0 ;;
        --no-bench) BENCH=0 ;;
        --no-full) FULL=0 ;;
        -h|--help) sed -n '3,24p' "$0"; exit 0 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

[ -f Cargo.toml ] && grep -q '^name = "cute-rs"' Cargo.toml || { echo "run this from the cute-rs repository root" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: is this a GPU machine?" >&2; exit 2; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SLUG=$(echo "$GPU_NAME" | tr -cs 'A-Za-z0-9' '-' | sed 's/-*$//')
OUT="$PWD/gpu-results/${SLUG}-sm${CC}-${STAMP}"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

case "$CC" in
    90) SUITE=gpu_hopper ;;
    100|101|103) SUITE=gpu_blackwell ;;
    *) SUITE="" ;;
esac
SUITE=${VALIDATE_SUITE:-$SUITE} # to exercise this script on another GPU
log "cute-rs GPU validation on $GPU_NAME (sm_$CC), commit $(cat .cute-rs-commit 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[ -z "$SUITE" ] && log "note: sm_$CC has no architecture suite here (sm_90 and sm_10x do); running the full suite only"

# ---------------------------------------------------------------- 1. toolchain
# A Rust already on a network volume (RunPod's /workspace) is used as it is.
if [ -z "${CARGO_HOME:-}" ] && ! command -v cargo >/dev/null && [ -x /workspace/.cargo/bin/cargo ]; then
    export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup
fi
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"
if ! command -v cargo >/dev/null; then
    log "installing Rust (rustup, minimal profile, stable)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >"$OUT/rustup.log" 2>&1 \
        || { log "FAILED: rustup install (see rustup.log)"; exit 1; }
fi

if [ -z "${CUTE_MLIR_OPT:-}" ]; then
    for c in "/workspace/llvm-$LLVM_VERSION/bin/mlir-opt" "/usr/lib/llvm-$LLVM_VERSION/bin/mlir-opt"; do
        [ -x "$c" ] && { export CUTE_MLIR_OPT="$c"; break; }
    done
fi
if [ -z "${CUTE_MLIR_OPT:-}" ]; then
    log "installing mlir-opt (LLVM $LLVM_VERSION, apt.llvm.org)"
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
    CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME:-noble}")
    {
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq wget gnupg ca-certificates &&
        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | $SUDO tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null &&
        echo "deb http://apt.llvm.org/$CODENAME/ llvm-toolchain-$CODENAME-$LLVM_VERSION main" | $SUDO tee /etc/apt/sources.list.d/llvm-$LLVM_VERSION.list >/dev/null &&
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "mlir-$LLVM_VERSION-tools" "llvm-$LLVM_VERSION"
    } >"$OUT/apt.log" 2>&1 || { log "FAILED: mlir-opt install (see apt.log)"; exit 1; }
    export CUTE_MLIR_OPT="/usr/lib/llvm-$LLVM_VERSION/bin/mlir-opt"
fi

# The JIT disk cache refuses world-writable directories (network volumes): keep it local.
export CUTE_RS_CACHE_DIR="${CUTE_RS_CACHE_DIR:-$HOME/.cache/cute-rs-validate}"
mkdir -p "$CUTE_RS_CACHE_DIR" && chmod 700 "$CUTE_RS_CACHE_DIR"
export CARGO_TERM_COLOR=never RUST_BACKTRACE=1

{
    echo "== nvidia-smi"; nvidia-smi
    echo "== nvidia-smi -q (clocks, power)"; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE | head -80
    echo "== rustc"; rustc --version; cargo --version
    echo "== mlir-opt ($CUTE_MLIR_OPT)"; "$CUTE_MLIR_OPT" --version | head -3
    echo "== ptxas"; command -v ptxas && ptxas --version | tail -1 || echo "none (the driver JIT-compiles the PTX)"
    echo "== os"; uname -a; head -2 /etc/os-release; nproc; free -g | head -2
} >"$OUT/environment.txt" 2>&1
log "toolchain: $(rustc --version | cut -d' ' -f1-2), mlir-opt $CUTE_MLIR_OPT, ptxas $(command -v ptxas >/dev/null && ptxas --version | tail -1 | awk '{print $NF}' || echo none)"

# ---------------------------------------------------------------- 2. build
log "building the tests (release, --features cuda)"
T0=$(date +%s)
if ! cargo test --release --features cuda --workspace --no-run >"$OUT/build.log" 2>&1; then
    log "FAILED: build (see build.log)"; tail -30 "$OUT/build.log" | tee -a "$SUMMARY"
    tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"; exit 1
fi
log "built in $(( $(date +%s) - T0 )) s"

# A cargo test run's outcome: its result lines and failures, into the summary.
digest() { # log-file label
    local passed failed ignored
    passed=$(grep -Eo '[0-9]+ passed' "$1" | awk '{s+=$1} END {print s+0}')
    failed=$(grep -Eo '[0-9]+ failed' "$1" | awk '{s+=$1} END {print s+0}')
    ignored=$(grep -Eo '[0-9]+ ignored' "$1" | awk '{s+=$1} END {print s+0}')
    log "$2: $passed passed, $failed failed, $ignored ignored ($(grep -c 'skipping' "$1") skip notes)"
    grep -E -- '--- FAILED|^test .* FAILED|panicked at' "$1" | head -40 | sed 's/^/    /' | tee -a "$SUMMARY"
    [ "$failed" -eq 0 ] && ! grep -q 'error: test failed\|could not compile' "$1"
}
STATUS=0

# ---------------------------------------------------------------- 3. architecture suite
if [ -n "$SUITE" ]; then
    log "running $SUITE (CUTE_RS_EXPECT_SM=$CC)"
    T0=$(date +%s)
    CUTE_RS_EXPECT_SM=$CC cargo test --release --features cuda --test "$SUITE" -- --nocapture --test-threads=1 >"$OUT/$SUITE.log" 2>&1
    digest "$OUT/$SUITE.log" "$SUITE ($(( $(date +%s) - T0 )) s)" || STATUS=1
fi

# ---------------------------------------------------------------- 4. full suite
if [ "$FULL" -eq 1 ]; then
    log "running the full suite (cargo test --release --features cuda --workspace --no-fail-fast)"
    T0=$(date +%s)
    cargo test --release --features cuda --workspace --no-fail-fast >"$OUT/full-suite.log" 2>&1
    digest "$OUT/full-suite.log" "full suite ($(( $(date +%s) - T0 )) s)" || STATUS=1
fi

# ---------------------------------------------------------------- 5. timings
if [ "$BENCH" -eq 1 ] && [ -n "$SUITE" ]; then
    log "timings ($SUITE bench_*)"
    cargo test --release --features cuda --test "$SUITE" -- --ignored --nocapture --test-threads=1 bench_ >"$OUT/bench.log" 2>&1 \
        || log "    (a timing failed: see bench.log)"
    grep '^BENCH' "$OUT/bench.log" | sed 's/^BENCH /    cute-rs  /' | tee -a "$SUMMARY"
    PY=""
    for p in python3 python; do command -v "$p" >/dev/null && "$p" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null && { PY=$p; break; }; done
    if [ -n "$PY" ]; then
        log "cuBLAS through PyTorch on the same shapes"
        "$PY" - >"$OUT/cublas.log" 2>&1 <<'EOF'
import torch
def bench(f, reps=10):
    f(); torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); e.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    return best
m = n = k = 8192
a = torch.randn(m, k, device="cuda", dtype=torch.float16)
b = torch.randn(n, k, device="cuda", dtype=torch.float16)
t = bench(lambda: a @ b.T)
print(f"BENCH cublas_f16_f32acc {m}x{n}x{k} {t*1e6:.1f}us {2*m*n*k/t/1e12:.1f} TFLOP/s")
try:
    import torch.nn.functional as F
    for causal in (False, True):
        q, kk, v = (torch.randn(4, 16, 4096, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        t = bench(lambda: F.scaled_dot_product_attention(q, kk, v, is_causal=causal))
        fl = 4 * 4 * 16 * 4096 * 4096 * 64 / (2 if causal else 1)
        print(f"BENCH torch_sdpa b4h16s4096d64{'c' if causal else ''} {t*1e6:.1f}us {fl/t/1e12:.1f} TFLOP/s")
except Exception as e:
    print("sdpa:", e)
EOF
        grep '^BENCH' "$OUT/cublas.log" | sed 's/^BENCH /    torch    /' | tee -a "$SUMMARY"
    else
        log "    (no Python with CUDA PyTorch here: no cuBLAS comparison)"
    fi
fi

log "done: $([ $STATUS -eq 0 ] && echo PASS || echo FAIL)"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT.tar.gz"
exit $STATUS

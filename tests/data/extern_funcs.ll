; External device functions for tests/extern_calls.rs and tests/gpu_extern.rs, assembled
; with llvm-as at test time (cute-rs's own test data, MIT OR Apache-2.0).
target triple = "nvptx64-nvidia-cuda"

; 3 * x
define float @ext_scale(float %x) {
  %r = fmul float %x, 3.0
  ret float %r
}

; p[i] + 1, reading global memory.
define float @ext_load(ptr addrspace(1) %p, i32 %i) {
  %q = getelementptr float, ptr addrspace(1) %p, i32 %i
  %v = load float, ptr addrspace(1) %q, align 4
  %r = fadd float %v, 1.0
  ret float %r
}

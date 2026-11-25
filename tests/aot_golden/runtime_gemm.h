/* runtime_gemm.h: the kernel `runtime_gemm`, exported by cute-rs @VERSION@ (export format 1) for sm_80, PTX ISA 7.0.
 *
 * Generated code: do not edit. Build runtime_gemm.c with the CUDA toolkit's headers and link with -lcuda:
 *
 *   runtime_gemm_Kernel_Module_t module;
 *   runtime_gemm_Kernel_Module_Load(&module);   (a CUresult; module.error says why on failure)
 *   runtime_gemm_Args_t args = { ... };          (one field per kernel parameter)
 *   runtime_gemm_Launch_Config_t config = { {grid}, {block}, {1, 1, 1}, 0, stream };
 *   runtime_gemm_Kernel_Launch(&module, &config, &args);
 *   runtime_gemm_Kernel_Module_Unload(&module);
 */
#ifndef RUNTIME_GEMM_CUTE_RS_EXPORT_H
#define RUNTIME_GEMM_CUTE_RS_EXPORT_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#define RUNTIME_GEMM_CUTE_RS_EXPORT_FORMAT 1
#define RUNTIME_GEMM_CUTE_RS_VERSION "@VERSION@"
#define RUNTIME_GEMM_KERNEL_NAME "runtime_gemm"
#define RUNTIME_GEMM_TARGET "sm_80"
#define RUNTIME_GEMM_PTX_VERSION 70
#define RUNTIME_GEMM_BLOCK_DIM_X 128
#define RUNTIME_GEMM_BLOCK_DIM_Y 1
#define RUNTIME_GEMM_BLOCK_DIM_Z 1
#define RUNTIME_GEMM_NUM_PARAMS 5

#ifdef __cplusplus
extern "C" {
#endif

/* The kernel's parameters, in order. */
typedef struct {
    CUdeviceptr a; /* `a`: Float16 *, 16-byte aligned */
    CUdeviceptr b; /* `b`: Float16 *, 16-byte aligned */
    CUdeviceptr c; /* `c`: Float32 *, 16-byte aligned */
    int32_t n; /* `n`: Int32, a multiple of 128 */
    int32_t k; /* `k`: Int32, a multiple of 32 */
} runtime_gemm_Args_t;

/* A launch: grid and block sizes, the cluster shape (entries of 0 mean 1; clusters need SM90 and CUDA 12),
 * the bytes of the kernel's dynamic shared-memory region, and the stream (NULL for the legacy stream). */
typedef struct {
    unsigned int grid[3];
    unsigned int block[3];
    unsigned int cluster[3];
    unsigned int dynamic_smem_bytes;
    CUstream stream;
} runtime_gemm_Launch_Config_t;

/* The loaded kernel. `error` holds a message when a function returns an error. */
typedef struct {
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdevice device;
    int retained_primary;
    int from_cubin;
    char error[256];
} runtime_gemm_Kernel_Module_t;

/* Load the kernel into the current context (the first device's primary context when none is current):
 * the cubin when the device runs it as a binary, else the PTX, which the driver's JIT compiles. */
CUresult runtime_gemm_Kernel_Module_Load(runtime_gemm_Kernel_Module_t *module);

/* Unload the module (and release the primary context Load retained). */
CUresult runtime_gemm_Kernel_Module_Unload(runtime_gemm_Kernel_Module_t *module);

/* Check the launch against the kernel (block size, clusters, pointer alignment and size, divisibility)
 * and queue it (cuLaunchKernelEx), from a thread whose current context is the module's. Returns
 * CUDA_ERROR_INVALID_VALUE, with module->error set, when a check fails. */
CUresult runtime_gemm_Kernel_Launch(runtime_gemm_Kernel_Module_t *module, const runtime_gemm_Launch_Config_t *config, const runtime_gemm_Args_t *args);

#ifdef __cplusplus
}
#endif

#endif /* RUNTIME_GEMM_CUTE_RS_EXPORT_H */

/* aot_leaves.h: the kernel `aot_leaves`, exported by cute-rs @VERSION@ (export format 1) for sm_89, PTX ISA 7.8.
 *
 * Generated code: do not edit. Build aot_leaves.c with the CUDA toolkit's headers and link with -lcuda:
 *
 *   leaves_Kernel_Module_t module;
 *   leaves_Kernel_Module_Load(&module);   (a CUresult; module.error says why on failure)
 *   leaves_Args_t args = { ... };          (one field per kernel parameter)
 *   leaves_Launch_Config_t config = { {grid}, {block}, {1, 1, 1}, 0, stream };
 *   leaves_Kernel_Launch(&module, &config, &args);
 *   leaves_Kernel_Module_Unload(&module);
 */
#ifndef LEAVES_CUTE_RS_EXPORT_H
#define LEAVES_CUTE_RS_EXPORT_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#define LEAVES_CUTE_RS_EXPORT_FORMAT 1
#define LEAVES_CUTE_RS_VERSION "@VERSION@"
#define LEAVES_KERNEL_NAME "aot_leaves"
#define LEAVES_TARGET "sm_89"
#define LEAVES_PTX_VERSION 78
#define LEAVES_CUBIN_TARGET "sm_89"
#define LEAVES_BLOCK_DIM_X 32
#define LEAVES_BLOCK_DIM_Y 1
#define LEAVES_BLOCK_DIM_Z 1
#define LEAVES_NUM_PARAMS 5

#ifdef __cplusplus
extern "C" {
#endif

/* The kernel's parameters, in order. */
typedef struct {
    CUdeviceptr x; /* `x`: Float32 *, at least 64 elements, 16-byte aligned */
    int32_t n; /* `n`: Int32, a multiple of 4 */
    int64_t wide; /* `wide`: Int64 */
    float alpha; /* `alpha`: Float32 */
    uint8_t s[16]; /* `s`: 16 bytes by value, 8-byte aligned */
} leaves_Args_t;

/* A launch: grid and block sizes, the cluster shape (entries of 0 mean 1; clusters need SM90 and CUDA 12),
 * the bytes of the kernel's dynamic shared-memory region, and the stream (NULL for the legacy stream). */
typedef struct {
    unsigned int grid[3];
    unsigned int block[3];
    unsigned int cluster[3];
    unsigned int dynamic_smem_bytes;
    CUstream stream;
} leaves_Launch_Config_t;

/* The loaded kernel. `error` holds a message when a function returns an error. */
typedef struct {
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdevice device;
    int retained_primary;
    int from_cubin;
    char error[256];
} leaves_Kernel_Module_t;

/* Load the kernel into the current context (the first device's primary context when none is current):
 * the cubin when the device runs it as a binary, else the PTX, which the driver's JIT compiles. */
CUresult leaves_Kernel_Module_Load(leaves_Kernel_Module_t *module);

/* Unload the module (and release the primary context Load retained). */
CUresult leaves_Kernel_Module_Unload(leaves_Kernel_Module_t *module);

/* Check the launch against the kernel (block size, clusters, pointer alignment and size, divisibility)
 * and queue it (cuLaunchKernelEx), from a thread whose current context is the module's. Returns
 * CUDA_ERROR_INVALID_VALUE, with module->error set, when a check fails. */
CUresult leaves_Kernel_Launch(leaves_Kernel_Module_t *module, const leaves_Launch_Config_t *config, const leaves_Args_t *args);

#ifdef __cplusplus
}
#endif

#endif /* LEAVES_CUTE_RS_EXPORT_H */

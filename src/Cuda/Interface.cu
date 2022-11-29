#include "Cuda/Interface.cuh"


namespace apes {

// cudaError_t CudaCheckErrorBase(const cudaError_t err, const char *file, int line) {
// }

cudaError_t CudaCheckErrorBase(const char *file, int line) {
    return CudaCheckErrorBase(cudaGetLastError(), file, line);
}

// void CudaAssertErrorBase(const cudaError_t err, const char *file, int line) {
//     assert(CudaCheckErrorBase(err, file, line) == cudaSuccess);
// }

void CudaAssertErrorBase(const char *file, int line) {
    CudaAssertErrorBase(cudaGetLastError(), file, line);
}

cudaError_t CudaMalloc(void **ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t CudaMalloc(void **ptr, unsigned int size) {
    return cudaMalloc(ptr, size);
}

cudaError_t CudaCopyToDevice(void *target, void const *source, size_t size) {
    return cudaMemcpy(target, source, size, cudaMemcpyHostToDevice);
}

cudaError_t CudaCopyFromDevice(void *target, void const *source, size_t size) {
    return cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost);
}

cudaError_t CudaFree(void *ptr) { return cudaFree(ptr); }

}

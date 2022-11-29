#pragma once

#include "cuda_runtime.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "Manager.h"
#include "fmt/format.h"

namespace apes {
namespace cuda{

using status_t = cudaError_t;

class runtime_error : public std::exception {
    std::string msg;

    public:
        runtime_error(const status_t &err, const char *file, int line)
                : std::exception() {
            msg = fmt::format("CUDA reported runtime_error with message: \"{}\" at {}:{}",
                               cudaGetErrorString(err), file, line);
        }
        runtime_error(const runtime_error &other) noexcept {
            msg = other.msg;
        }
        runtime_error& operator=(const runtime_error& other) noexcept {
            msg = other.msg;
            return *this;
        }

        const char *what() const noexcept {
            return msg.c_str();
        }
};
#define cuda_error(err) (apes::cuda::runtime_error(err, __FILE__, __LINE__))

inline void check_cuda_call(const status_t &err, const char *file, int line) {
    if(err != cudaSuccess) throw runtime_error(err, file, line);
}
#define CheckCudaCall(err) (apes::cuda::check_cuda_call(err, __FILE__, __LINE__))

class Device {
    public:
        Device(int device_id) : id(device_id) {
            cudaGetDeviceProperties(&properties, id);
            peak_mem_bandwidth = 2.0*properties.memoryClockRate
                               * (properties.memoryBusWidth/8)/1.0e6;
        }

        void ListProperties() {
            fmt::print("Device Number: {}\n", id);
            fmt::print("  Device name: {}\n", properties.name);
            fmt::print("  Max Threads: {}\n", properties.maxThreadsPerBlock);
            fmt::print("  Max Block Dimensions: {}\n", fmt::join(properties.maxThreadsDim, ", "));
            fmt::print("  Max Grid Dimensions: {}\n", fmt::join(properties.maxGridSize, ", "));
            fmt::print("  Max Shared Memory (bytes): {}\n", properties.sharedMemPerBlock);
            fmt::print("  Total Global Memory (bytes): {}\n", properties.totalGlobalMem);
            fmt::print("  Total Constant Memory (bytes): {}\n", properties.totalConstMem);
            fmt::print("  Memory Clock Rate (kHz): {}\n", properties.memoryClockRate);
            fmt::print("  Memory Bus Width (bytes): {}\n", properties.memoryBusWidth);
            fmt::print("  PeakMemory Bandwidth (GB/s): {}\n\n", peak_mem_bandwidth);
        }

        size_t GetMaxThreads() const {
            return properties.maxThreadsPerBlock;
        }

    private:
        const int id;
        cudaDeviceProp properties;
        double peak_mem_bandwidth;
};

inline std::vector<Device> GetDevices() noexcept {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::vector<Device> result;
    for(int i = 0; i < nDevices; ++i) {
        result.emplace_back(Device(i));
        result[i].ListProperties();
    }
    return result;
}

inline int GetCurrentDevice() {
    int device;
    CheckCudaCall(cudaGetDevice(&device));
    return device;
}

inline void SetDevice(int device) {
    CheckCudaCall(cudaSetDevice(device));
}

template<typename DataClass, typename... ArgsTypes>
__global__ void ConstructOnGpu(DataClass *gpu_ptr, ArgsTypes... params){
    new (gpu_ptr) DataClass(params...);
}

template<typename DataClass, typename... ArgsTypes>
__global__ void ConstructArrayOnGpu(DataClass *gpu_ptr, size_t nElements, ArgsTypes... params) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int idx = tid;
    while(idx < nElements) {
        new (gpu_ptr + idx) DataClass(params...);
        idx += blockDim.x * gridDim.x;
    }
}

template<typename DataClass, typename... ArgsTypes>
void GenericCopyToGpu(const DataClass *gpu_ptr, ArgsTypes... params) {
    ConstructOnGpu<<<1, 1>>>(gpu_ptr, params...);
}

}//end namespace cuda

static cudaError_t CudaCheckErrorBase(const cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess)
        fmt::print("CUDA reported error with message: \"{}\" at {}:{}\n",
                   cudaGetErrorString(err), file, line); 
    return err;
}
// cudaError_t CudaCheckErrorBase(const char *file, int line);
static void CudaAssertErrorBase(const cudaError_t err, const char *file, int line) {
    // assert(CudaCheckErrorBase(err, file, line) == cudaSuccess);
}
// void CudaAssertErrorBase(const char *file, int line);
#define CudaCheckError(err) (CudaCheckErrorBase(err, __FILE__, __LINE__))
#define CudaAssertError(err) (CudaAssertErrorBase(err, __FILE__, __LINE__))


cudaError_t CudaMalloc(void **ptr, size_t size);
cudaError_t CudaMalloc(void **ptr, unsigned int size);
cudaError_t CudaCopyToDevice(void *target, const void *source, size_t size);
cudaError_t CudaCopyFromDevice(void *target, const void *source, size_t size);
cudaError_t CudaFree(void *ptr);

template<typename T>
T *AllocateOnGpu(const size_t size) {
    T *ptr;
    CudaManager::Instance().Report(size);
    CudaCheckError(CudaMalloc((void **)&ptr, size));
    return ptr;
}

template<typename T>
T *AllocateOnGpu() {
    return AllocateOnGpu<T>(sizeof(T));
}

template<typename T>
void FreeFromGpu(T *ptr) {
    CudaCheckError(CudaFree(ptr));
}

template<typename T>
void ConstantToGpu(T const *const source, T const target, const size_t size) {
    CudaCheckError(cudaMemcpyToSymbol(target, source, size, 0, cudaMemcpyHostToDevice));
}

template<typename T>
void ConstantToGpu(T const *const source, T const target) {
    ConstantToGpu<T>(source, target, sizeof(T));
}

template<typename T>
void CopyToGpu(T const *const source, T *const target, const size_t size) {
    CudaCheckError(CudaCopyToDevice(target, source, size));
}

template<typename T>
void CopyToGpu(T const *const source, T *const target) {
    CopyToGpu<T>(source, target, sizeof(T));
}

template<typename T>
void CopyFromGpu(T const *const source, T *const target, const size_t size) {
     CudaCheckError(CudaCopyFromDevice(target, source, size));
}

template<typename T>
void CopyFromGpu(T const *const source, T *const target) {
    CopyFromGpu<T>(target, source, sizeof(T));
}

template<typename T>
struct MemoryPair {
    std::vector<T> host;
    T *device;
    size_t size;

    MemoryPair(size_t size_) : size{size_} {
        host.resize(size);
        device = AllocateOnGpu<T>(size*sizeof(T));
    }

    template<typename... Args>
    MemoryPair<T>(size_t size_, Args &&...args) : size{size_} {
        for(size_t i = 0; i < size; ++i)
            host.emplace_back(T(args...));
        device = AllocateOnGpu<T>(size*sizeof(T));
    }

    ~MemoryPair() {}

    void ToGpu() {
        CopyToGpu<T>(host.data(), device, size*sizeof(T));
    }

    void FromGpu() {
        CopyFromGpu<T>(device, host.data(), size*sizeof(T));
    }
};

}

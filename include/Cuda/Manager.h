#pragma once

#include <type_traits>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "fmt/format.h"

class CudaManager {
    public:
        static CudaManager &Instance() {
            static CudaManager instance;
            return instance;
        }

        CudaManager(const CudaManager&) = delete;
        CudaManager(CudaManager&&) = delete;
        CudaManager& operator=(const CudaManager&) = delete;
        CudaManager& operator=(CudaManager&&) = delete;

        void Report(std::size_t allocd) {
            allocatedMem += allocd;
            if(totalMem == 0)
                fmt::print(" Allocated  | Total Memory | Free Memory \n");
            cudaMemGetInfo(&freeMem, &totalMem);
            fmt::print("{:11} | {:12} | {:11}\n", allocatedMem, totalMem, freeMem);
        }

    private:
        CudaManager() = default;
        std::size_t freeMem{}, totalMem{}, allocatedMem{};
};

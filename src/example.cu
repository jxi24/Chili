#include "Integrator/Vegas.cuh"
#include "Cuda/Interface.cuh"
#include <iomanip>
#include "thrust/sort.h"
#include "thrust/copy.h"
#include "thrust/binary_search.h"
#include "thrust/adjacent_difference.h"
#include "thrust/iterator/constant_iterator.h"
#include "thrust/iterator/counting_iterator.h"

#include <fstream>

__global__
void EvaluateFunc(double *rans, double *results, double *wgt) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    results[idx] = rans[idx*2]*rans[idx*2+1]*wgt[idx];
}

struct test_functor {
    static constexpr size_t npart = 8;
    static constexpr size_t ndims = 3*npart-4+2;
    static constexpr size_t threads = 1024;
    void operator()(double *x, double *results, double *wgt, size_t nevents) const {
        size_t blocks = nevents/threads;
        EvaluateFunc<<<blocks, threads>>>(x, results, wgt);
        CheckCudaCall(cudaDeviceSynchronize());
    }
};

int main() {
    test_functor test;
    size_t ndims = test.ndims;
    size_t nbins = 1000;
    size_t seed = 123456789;
    apes::cuda::Vegas vegas(ndims, nbins, seed, {});

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vegas.Optimize(test);
    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Optimization Time: " << milliseconds << " ms" << std::endl;
}

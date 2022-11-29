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

struct test_functor {
    static constexpr size_t npart = 8;
    static constexpr size_t ndims = 3*npart-4+2;
    __host__ __device__ double operator()(double *x) const {
        return x[0]*x[1];
    }
};

int main() {
    test_functor h_test, *d_test;
    size_t ndims = h_test.ndims;
    size_t nbins = 1000;
    size_t seed = 123456789;
    apes::cuda::Vegas vegas(ndims, nbins, seed, {});

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Setup function on device
    CheckCudaCall(cudaMalloc(&d_test, sizeof(test_functor))); 
    CheckCudaCall(cudaMemcpy(d_test, &h_test, sizeof(test_functor), cudaMemcpyHostToDevice));

    for(size_t i = 0; i < 10; ++i) {
        cudaEventRecord(start);
        vegas(&h_test);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Time per iteration: " << milliseconds << " ms" << std::endl;
    }
}

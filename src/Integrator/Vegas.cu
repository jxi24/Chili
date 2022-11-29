#include "Integrator/Vegas.cuh"

using apes::cuda::Vegas;

__global__
void apes::cuda::detail::Histogram(AdaptiveMap* map, uint16_t *index, double *results, double *hist, size_t nevents) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = 0; i < map -> dims; ++i) {
        uint16_t ibin = i*map->bins + index[i+map->dims*idx];
        atomicAdd(&hist[ibin], results[idx]*results[idx]);
    }
}

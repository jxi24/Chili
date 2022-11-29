#include "Integrator/AdaptiveMap.cuh"
#include "thrust/sequence.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/binary_search.h"
#include "thrust/distance.h"

#include <iostream>

using apes::cuda::AdaptiveMap;
using apes::cuda::AdaptiveMapHandler;

// Taken from thurst/examples/tiled_range.cu
template<typename Iterator>
class tiled_range {
    public:
        using difference_type = typename thrust::iterator_difference<Iterator>::type;
        
        struct tile_functor : public thrust::unary_function<difference_type, difference_type> {
            difference_type tile_size;

            tile_functor(difference_type tile_size) : tile_size(tile_size) {}

            __host__ __device__
            difference_type operator()(const difference_type& i) const {
                return i % tile_size;
            }
        };

        using CountingIterator = thrust::counting_iterator<difference_type>;
        using TransformIterator = thrust::transform_iterator<tile_functor, CountingIterator>;
        using PermutationIterator = thrust::permutation_iterator<Iterator, TransformIterator>;

        // type of the tiled_range iterator
        using iterator = PermutationIterator;

        // construct repeated_range for the range [first, last)
        tiled_range(Iterator first_, Iterator last_, difference_type tiles_)
            : first(first_), last(last_), tiles(tiles_) {}

        iterator begin(void) const {
            return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
        }

        iterator end(void) const {
            return begin() + tiles * (last - first);
        }

    protected:
        Iterator first;
        Iterator last;
        difference_type tiles;
};

__host__ __device__
AdaptiveMap::AdaptiveMap(const AdaptiveMap &other) {
    hist = other.hist;
    dims = other.dims;
    bins = other.bins;
}

AdaptiveMap::AdaptiveMap(size_t _dims, size_t _bins) : dims(_dims), bins(_bins) {
    hist = new double[dims*(bins+1)];
    for(size_t i = 0; i < dims; ++i) {
        for(size_t j = 0; j < bins; ++j) {
            hist[i*(bins+1)+j] = static_cast<double>(j)/bins;
        }
        hist[i*(bins+1)+bins] = 1.0;
    }
}

__device__
void AdaptiveMap::Sample(double *rans) const {
    // Batch index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = 0; i < dims; ++i) {
        const auto position = rans[i + idx*dims] * static_cast<double>(bins);
        const auto index = static_cast<size_t>(position);
        const auto loc = position - static_cast<double>(index);
        const double left = lower_edge(i, index);
        const double size = width(i, index);

        // Calculate inverse CDF
        rans[i + idx*dims] = left + loc * size;
    }
}

__device__
void AdaptiveMap::GenerateWeight(double *rans, double *wgts) const {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    wgts[idx] = 1.0;
    for(size_t i = 0; i < dims; ++i) {
        const auto index = FindBin(i, rans[i+idx*dims]);
        wgts[idx] = wgts[idx] * width(i, index) * static_cast<double>(bins);
    }
}

__device__
void AdaptiveMap::SampleWeight(double *rans, double *wgts, uint16_t *index) const {
    // Batch index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    wgts[idx] = 1.0;
    for(size_t i = 0; i < dims; ++i) {
        const auto position = rans[i + idx*dims] * static_cast<double>(bins);
        index[idx*dims+i] = static_cast<size_t>(position);
        const auto loc = position - static_cast<double>(index[idx*dims+i]);
        const double left = lower_edge(i, index[idx*dims+i]);
        const double size = width(i, index[idx*dims+i]);

        // Calculate inverse CDF
        rans[i + idx*dims] = left + loc * size;
        wgts[idx] = wgts[idx] * size * static_cast<double>(bins);
    }
}

__host__ __device__
size_t AdaptiveMap::FindBin(size_t dim, double x) const {
    for(size_t i = 1; i < bins+1; ++i) {
        if(x < hist[dim*(bins+1)+i]) return i - 1;
    }
    return bins+1;
}

__host__
void AdaptiveMap::Adapt(double alpha, const std::vector<double> &data) {
    std::vector<double> tmp(bins);
    std::vector<double> new_hist(dims*(bins+1));

    for(size_t i = 0; i < dims; ++i) {
        // Load data into tmp
        tmp.assign(data.begin() + static_cast<int>(i * bins),
                   data.begin() + static_cast<int>((i + 1) * bins));

        // Smooth data by averaging over neighbors
        double previous = tmp[0];
        double current = tmp[1];
        tmp[0] = (previous + current) / 2;
        double norm = tmp[0];

        for(size_t bin = 1; bin < bins - 1; ++bin) {
            const double sum = previous + current;
            previous = current;
            current = tmp[bin + 1];
            tmp[bin] = (sum + current) / 3;
            norm += tmp[bin];
        }
        tmp[bins-1] = (previous + current) / 2;
        norm += tmp[bins-1];

        // If norm is zero, then there is no data to adjust
        if(norm == 0) continue;

        // Compute the importance factor
        double average_bin = 0;
        for(size_t bin = 0; bin < bins; ++bin) {
            if(tmp[bin] != 0) {
                const double r = tmp[bin] / norm;
                const double fac = pow((r - 1.0) / log(r), alpha);
                average_bin += fac;
                tmp[bin] = fac;
            }
        }
        average_bin /= static_cast<double>(bins);

        double cbin = 0;
        size_t ibin = 0;

        // Adjust boundaries
        for(size_t nbin = 1; nbin < bins; ++nbin) {
            for(; cbin < average_bin; ++ibin) cbin += tmp[ibin];

            const double prev = lower_edge(i, ibin-1);
            const double curr = lower_edge(i, ibin);

            cbin -= average_bin;
            const double delta = (curr - prev) * cbin;
            new_hist[i * (bins + 1) + nbin] = curr - delta / tmp[ibin - 1];
        }
        new_hist[i * (bins + 1) + bins] = 1.0;
    }

    std::copy(new_hist.begin(), new_hist.end(), hist);
}

__global__ void apes::cuda::detail::SeedRandom(curandState *state, size_t seed) {
    int id = threadIdx.x+blockIdx.x*blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void apes::cuda::detail::BatchSample(curandState *state, AdaptiveMap *map, double* rans, double* wgts, uint16_t *index) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy state to local memory for efficiency
    curandState localState = state[idx];

    // Generate random numbers
    for(int i = 0; i < map -> dims; ++i) {
        rans[map -> dims*idx + i] = curand_uniform_double(&localState);
    }

    // Generate vegas random numbers and calculate weights
    map -> SampleWeight(rans, wgts, index);

    // Copy state back to global memory
    state[idx] = localState;
}

AdaptiveMapHandler::AdaptiveMapHandler(size_t _ndims, size_t _nbins) : ndims{_ndims}, nbins{_nbins} {
    hist_size = ndims*(nbins+1)*sizeof(double);
    map_size = sizeof(apes::cuda::AdaptiveMap);
    h_map = new apes::cuda::AdaptiveMap(ndims, nbins);
    // Deep copy of map
    CheckCudaCall(cudaMalloc(&d_map, map_size));
    CheckCudaCall(cudaMemcpy(d_map, h_map, map_size, cudaMemcpyHostToDevice));
    CheckCudaCall(cudaMalloc(&d_hist, hist_size));
    CheckCudaCall(cudaMemcpy(d_hist, h_map -> hist, hist_size, cudaMemcpyHostToDevice));
    CheckCudaCall(cudaMemcpy(&(d_map->hist), &d_hist, sizeof(void *), cudaMemcpyHostToDevice));
}


AdaptiveMapHandler::~AdaptiveMapHandler() {
    if(h_map) delete h_map;
    cudaFree(d_hist);
    cudaFree(d_map);
    destroyDeviceArray(devStates);
}

void AdaptiveMapHandler::Seed(size_t seed, size_t nevents, size_t threads) {
    size_t blocks = nevents/threads;
    devStates = createDeviceArray<curandState>(nevents);
    apes::cuda::detail::SeedRandom<<<blocks, threads>>>(devStates, seed);
    CheckCudaCall(cudaDeviceSynchronize());
}

void AdaptiveMapHandler::Sample(double *rans, double *wgts, uint16_t *index, size_t nevents, size_t threads) {
    size_t blocks = nevents/threads;
    apes::cuda::detail::BatchSample<<<blocks, threads>>>(devStates, d_map, rans, wgts, index);
    CheckCudaCall(cudaDeviceSynchronize());
}

void AdaptiveMapHandler::Adapt(double alpha, const std::vector<double> &data) {
    h_map -> Adapt(0.25, data);
    CheckCudaCall(cudaMemcpy(d_hist, h_map->hist, hist_size, cudaMemcpyHostToDevice));
}

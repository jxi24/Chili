#pragma once

#include "stdgpu/vector.cuh"
#include "curand_kernel.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include "Cuda/Interface.cuh"
#include <cstdint>

namespace apes::cuda {

template<typename Iterator, typename OStream>
void print_array(const std::string &name, Iterator first, Iterator last, OStream &stream) {
    using T = typename std::iterator_traits<Iterator>::value_type;

    //stream << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(stream, " "));
    //stream << "\n";
}

struct AdaptiveMap {
    __host__ __device__ AdaptiveMap(const AdaptiveMap&);
    __host__ AdaptiveMap(size_t _dims, size_t _bins);
    ~AdaptiveMap() { if(hist) delete[] hist; }

    // Bin locations
    __host__ __device__
    double lower_edge(size_t dim, size_t bin) const { return hist[dim*(bins+1) + bin]; }
    __host__ __device__
    double upper_edge(size_t dim, size_t bin) const { return hist[dim*(bins+1) + bin + 1]; }
    __host__ __device__
    double width(size_t dim, size_t bin) const { return upper_edge(dim, bin) - lower_edge(dim, bin); }
    __host__ __device__
    uint16_t FindBin(size_t, double) const;

    // Functions
    __device__
    void Sample(double*) const;
    __device__
    void GenerateWeight(double*, double*) const;
    __device__
    void SampleWeight(double*, double*) const;
    __device__
    void Histogram(double*, uint16_t*) const;
    __host__
    void Adapt(double, const std::vector<double>&);

    double *hist;
    size_t dims, bins;
};

namespace detail {
__global__ void SeedRandom(curandState*, size_t);
__global__ void BatchSample(curandState*, AdaptiveMap*, double*, double*);
}

struct AdaptiveMapHandler {
    AdaptiveMap *h_map, *d_map;
    double *d_hist;
    size_t hist_size, map_size, ndims, nbins;
    bool seeded{false};
    curandState *devStates;
    static constexpr size_t _threads = 1024;

    AdaptiveMapHandler(size_t ndims, size_t nbins);
    ~AdaptiveMapHandler();
    void Seed(size_t seed, size_t nevents, size_t threads=_threads);
    void Sample(double *rans, double *wgts, size_t nevents, size_t threads=_threads);
    void Adapt(double alpha, const std::vector<double> &data);
};

}

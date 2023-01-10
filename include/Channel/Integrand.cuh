#ifndef INTEGRAND_CUH
#define INTEGRAND_CUH

#include "Channel/Mapper.cuh"
#include "Cuda/Interface.cuh"
#include "Integrator/Vegas.cuh"

namespace apes::cuda {

template<typename T, typename Integrator, typename Mapper>
struct Channel {
    Integrator *integrator = nullptr;
    Mapper *mapper = nullptr;
    bool on_gpu = true;

    Channel(Integrator *_integrator, Mapper *_mapper, bool _on_gpu=true) : on_gpu(_on_gpu) {
        // Integrator requirements
        static_assert(std::is_invocable_v<decltype(&Integrator::InitializeTrain),
                      Integrator&>,
                      "Integrator must have function InitializeTrain");
        static_assert(std::is_invocable_v<decltype(&Integrator::AddTrainData),
                      Integrator&, double>,
                      "Integrator must have function AddTrainData");
        static_assert(std::is_invocable_v<decltype(&Integrator::Train),
                      Integrator&>,
                      "Integrator must have function Train");
        static_assert(std::is_invocable_v<decltype(&Integrator::Sample),
                      Integrator&, double*>,
                      "Integrator must have function Sample");
        static_assert(std::is_invocable_v_r<decltype(&Integrator::NDims), size_t, Integrator&>,
                      "Integrator must have function NDims");

        // Mapper requirements
        static_assert(std::is_invocable_v<decltype(&Mapper::GeneratePoint),
                      Mapper&, T*, double*>, "Mapper must have function GeneratePoint");
        static_assert(std::is_invocable_v<decltype(&Mapper::GenerateWeight),
                      Mapper&, T*, double*, double*>, "Mapper must have function GenerateWeight");
        static_assert(std::is_invocable_v_r<decltype(&Mapper::NDims), size_t, Mapper&>,
                      "Mapper must return the number of dimensions");

        // Ensure the integrator and the mapper are the same size
        if(_integrator -> NDims() != _mapper -> NDims())
            throw std::runtime_error("Integrator and Mapper have different number of dimensions");

        integrator = _integrator;
        mapper = _mapper;
    }
    ~Channel() {
        if(on_gpu) {
            if(integrator) CheckCudaCall(cudaFree(integrator));
            if(mapper) CheckCudaCall(cudaFree(mapper));
        } else {
            if(integrator) delete integrator;
            if(mapper) delete mapper;
        }
    }

    size_t NDims() const { return mapper -> NDims(); }
};

template<typename T, typename Integrator, typename Mapper>
__global__
void InitializeTrainGPU(Channel<T, Integrator, Mapper> *channels) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    channels[idx] -> InitializeTrain();
}

template<typename T, typename Integrator, typename Mapper>
__global__
void AddTrainDataGPU(Channel<T, Integrator, Mapper> *channels, size_t *ichannel, double *val2, size_t npts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < npts) channels[ichannel[idx]] -> AddTrainData(val2);
}

template<typename Functor, typename T, typename Integrator, typename Mapper>
class Integrand {
    public:
        Integrand() = default;
        Integrand(Functor func) : m_func{std::move(func)} {
            // Ensure the functor is valid
            static_assert(std::is_invocable_v<decltype(&Functor::Evaluate),
                          Functor&, T*, double*>, "Functor must have a function Evaluate"); 
            static_assert(std::is_invocable_v<decltype(&Functor::PreProcess),
                          Functor&, T*, bool*>, "Functor must have a function PreProcess"); 
            static_assert(std::is_invocable_v<decltype(&Functor::PostProcess),
                          Functor&, T*, double*, bool*>, "Functor must have a function PostProcess"); 
        }

        // Function Utilities
        void Evaluate(T* data, double* result) const { func.Evaluate(data, result); }
        void PreProcess(T* data, bool* valid) const { func.PreProcess(data, valid); }
        void PostProcess(T* data, double* result, bool* valid) const { func.PostProcess(data, result, valid); }

        // Channel Utilities
        void AddChannel(Channel<T, Integrator, Mapper> channel) {
            if(channels.size() != 0) {
                if(channels[0].NDims() != channel.NDims())
                    throw std::runtime_error("Integrand: Channels have different dimensions");
                if(channels[0].on_gpu != channel.on_gpu)
                    throw std::runtime_error("Integrand: Channels on different devices");
            } else {
                on_gpu = channel.on_gpu;
            }
            channels.push_back(std::move(channel));
        }
        void RemoveChannel(int idx) { channels.erase(channels.begin() + idx); }
        size_t NChannels() const { return channels.size(); }
        size_t NDims() const { return channels[0].NDims(); }

        // Train integrator
        void InitializeTrain() {
                for(size_t i = 0; i < npts; ++i) {
                    channel[ichannel[i]].AddTrainData(val2[i]);
                }
            if(on_gpu) {
                InitializeTrainGPU<T, Integrator, Mapper><<<channels.size(), 1>>>(channels.data());
            } else {
                for(auto &channel : channels) {
                    channel.integrator -> InitializeTrain();
                }
            }
        }
        void AddTrainData(size_t *ichannel, double *val2, size_t npts) {
            if(on_gpu) {
                constexpr size_t nthreads = 1024;
                AddTrainDataGPU<T, Integrator, Mapper><<<nthreads, npts/nthreads>>>(channels.data(), ichannel, val2, npts);
            } else {
                for(size_t i = 0; i < npts; ++i) {
                    channels[ichannel[i]].integrator -> AddTrainData(val2[i]);
                }
            }
        }
        void Train() {
            if(on_gpu) {
            } else {
                for(auto &channel : channels) {
                    channel.integrator -> Train();
                }
            }
        }

    private:
        Functor m_func;
        std::vector<Channel<T, Integrator, Mapper>> channels;
        bool on_gpu = true;
};

}

#endif

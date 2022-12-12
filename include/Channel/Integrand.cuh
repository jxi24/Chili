#ifndef INTEGRAND_CUH
#define INTEGRAND_CUH

#include "Channel/Mapper.cuh"
#include "Integrator/Vegas.cuh"

namespace apes::cuda {

template<typename T, typename Integrator, typename Mapper>
struct Channel {
    Integrator *integrator;
    Mapper *mapper;

    __host__ __device__
    Channel(Integrator *_integrator, Mapper *_mapper) {
        // Integrator requirements
        static_assert(std::is_invocable_v<decltype(&Integrator::Train),
                      Integrator&, double*, double*>,
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

    size_t NDims() const { return mapping -> NDims(); }
};

template<typename Functor, typename T>
class Integrand {
    public:
        Integrand() = default;
        Integrand(Functor func) : m_func{std::move(func)} {}

        // Function Utilities
        void operator()(T*, double*) const;
        void PreProcess(T*, bool*) const;
        void PostProcess(T*, double*, bool*) const;

        // Channel Utilities
        void AddChannel(Channel<T> channel) {
            if(channels.size() != 0)
                if(channels[0].NDims() != channel.NDims())
                    throw std::runtime_error("Integrand: Channels have different dimensions");
            channels.push_back(std::move(channel));
        }
        void RemoveChannel(int idx) { channels.erase(channels.begin() + idx); }
        size_t NChannels() const { return channels.size(); }
        size_t NDims() const { return channels[0].NDims(); }

        // Train integrator
        void InitializeTrain() {

        }
};

}

#endif

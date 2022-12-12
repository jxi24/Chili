#ifndef MULTICHANNEL_CUH
#define MULTICHANNEL_CUH

#include "Integrator/Vegas.cuh"
#include "Channel/Integrand.cuh"

namespace apes::cuda {

struct MultiChannelSummary {
    using StatsData = summary_stats_data<double>;   

    std::vector<StatsData> results;
    std::vector<double> best_weights;
    StatsData sum_results;

    StatsData Result() const { return sum_results; }
};

struct MultiChannelParams {
    size_t ncalls{ncalls_default}, niterations{nint_default};
    double rtol{rtol_default};
    size_t nrefine{nrefine_default};
    double beta{beta_default}, min_alpha{min_alpha_default};
    size_t iteration{};

    static constexpr size_t ncalls_default{10000}, nint_default{10};
    static constexpr double rtol_default{1e-2};
    static constexpr size_t nrefine_default{1};
    static constexpr double beta_default{0.25}, min_alpha_default{1e-5};
    static constexpr size_t nparams = 7;
};

class MultiChannel {
    public:
        MultiChannel() = default;
        MultiChannel(size_t, size_t, MultiChannelParams);

        // Utilities
        size_t Dimensions() const { return ndims; }
        size_t NChannels() const { return channel_weights.size(); }
        MultiChannelParams Parameters() const { return params; }
        MultiChannelParams &Parameters() { return params; }

        // Optimization and event generation
        template<typename T>
        void operator()(Integrand<T>&);
        template<typename T>
        void Optimize(Integrand<T>&);

        // Getting results
        MultiChannelSummary Summary();

        // YAML interface
        friend YAML::convert<apes::cuda::MultiChannel>;

    private:
        void Adapt(const std::vector<double>&);
        void TrainChannels();
        template<typename T>
        void RefineChannels(Integrand<T> &func) {
            params.iteration = 0;
            params.ncalls *= 2;
            for(auto &channel : func.Channels()) {
                if(channel.integrator.Grid().Bins() < 200)
                    channel.integrator.Refine();
            }
        }
        void PrintIteration() const;
        void MaxDifference(const std::vector<double>&);

        size_t ndims{};
        MultiChannelParams params{};
        std::vector<double> channel_weights, best_weights;
        double min_diff{std::numeric_limits<double>::infinity()};
        MultiChannelSummary summary;
};

template<typename T>
void apes::cuda::MultiChannel::operator()(Integrand<T> &func) {
    size_t nchannels = channel_weights.size();
    thrust::device_vector<double> rans(ndims*params.ncalls);
    thrust::device_vector<T> point(ndims*params.ncalls);
    thrust::device_vector<double> densities(nchannels*params.ncalls);
    thrust::device_vector<double> train_data(nchannels);
    thrust::device_vector<size_t> ichannel(params.ncalls);
    
    func.InitializeTrain();
}

}

#endif

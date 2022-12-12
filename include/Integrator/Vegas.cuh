#ifndef VEGAS_CUH
#define VEGAS_CUH

#include "Integrator/AdaptiveMap.cuh"
#include "Integrator/Statistics.cuh"

namespace apes::cuda {

struct VegasParams {
    size_t ncalls{ncalls_default};
    double rtol{rtol_default}, atol{atol_default}, alpha{alpha_default};
    size_t niterations{nitn_default};

    static constexpr size_t nitn_default = 10, ncalls_default = 1024*4096, nrefine_default = 5;
    static constexpr double alpha_default = 1.5, rtol_default = 1e-4, atol_default = 1e-4;
    static constexpr size_t nparams = 6;
};

struct VegasSummary {
    using StatsData = summary_stats_data<double>;   
    std::vector<StatsData> results;
    StatsData sum_results;

    StatsData Result() const { return sum_results; }
};

namespace detail {

__global__ void Histogram(AdaptiveMap*, double*, double*, double*, size_t);

}

class Vegas {
    public:
        Vegas(size_t ndims, size_t nbins, size_t seed, VegasParams _params) : m_map_handler(ndims, nbins), m_params{std::move(_params)} {
            m_map_handler.Seed(seed, m_params.ncalls);
        }

        template<typename Functor>
        void operator()(Functor func, size_t threads=1024) {
            static_assert(std::is_invocable_v<Functor, double*, double*, double*, size_t>, "Functor must be invocable with signature void(double*, double*, double*, size_t)"); 
            static_assert(Functor::ndims > 0, "Functor has to have at least one dimension");
            thrust::device_vector<double> rans(m_map_handler.h_map->dims*m_params.ncalls);
            thrust::device_vector<double> results(m_params.ncalls);
            thrust::device_vector<double> weights(m_params.ncalls);

            m_map_handler.Sample(rans.data().get(), weights.data().get(), m_params.ncalls);
            func(rans.data().get(), results.data().get(), weights.data().get(), m_params.ncalls);

            summary_stats_unary_op<double> unary_op;
            summary_stats_binary_op<double> binary_op;
            summary_stats_data<double> init;
            init.initialize();
            summary_stats_data<double> result = thrust::transform_reduce(results.begin(), results.end(), unary_op, init, binary_op);
            m_summary.results.push_back(result);
            m_summary.sum_results = binary_op(m_summary.sum_results, result);

            size_t blocks = m_params.ncalls/threads;
            thrust::device_vector<double> train_data(m_map_handler.h_map->dims*m_map_handler.h_map->bins);
            detail::Histogram<<<blocks, threads>>>(m_map_handler.d_map, rans.data().get(), results.data().get(),
                                                   train_data.data().get(), m_params.ncalls);
            std::vector<double> h_train_data(train_data.size());
            thrust::copy(train_data.begin(), train_data.end(), h_train_data.begin());
            m_map_handler.Adapt(m_params.alpha, h_train_data);

        }
        template<typename Functor>
        void Optimize(Functor func, size_t=1024) {
            double abs_err = std::numeric_limits<double>::max(), rel_err = std::numeric_limits<double>::max();
            while((abs_err > m_params.atol && rel_err > m_params.rtol) || m_summary.results.size() < m_params.niterations) {
                (*this)(func);
                summary_stats_data<double> current = m_summary.Result();
                abs_err = current.error();
                rel_err = abs_err / std::abs(current.mean);

                PrintIteration();
            }
        }
        void Adapt(const std::vector<double>&);

        void PrintIteration() const {
            std::cout << fmt::format("{:3d}   {:^8.5e} +/- {:^8.5e}    {:^8.5e} +/- {:^8.5e}",
                    m_summary.results.size(), m_summary.results.back().mean, m_summary.results.back().error(),
                    m_summary.Result().mean, m_summary.Result().error()) << std::endl;
        }
    private:
        AdaptiveMapHandler m_map_handler;
        VegasParams m_params;
        VegasSummary m_summary{};
};

}

#endif

#ifndef MULTICHANNEL_HH
#define MULTICHANNEL_HH

#include "Integrator/Vegas.hh"
#include "Channel/Integrand.hh"
#include <optional>
#include <functional>

#ifdef ENABLE_MPI
#include "Tools/MPI.hh"
#endif

namespace apes {

struct MultiChannelSummary {
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
    double refine_size{refine_size_default};
    size_t max_bins{max_bins_default};
    size_t iteration{};
    bool   should_optimize{should_optimize_default};
    static constexpr size_t ncalls_default{10000}, nint_default{10};
    static constexpr double rtol_default{1e-5};
    static constexpr size_t nrefine_default{1};
    static constexpr double beta_default{0.25}, min_alpha_default{1e-5};
    static constexpr double refine_size_default{1.5};
    static constexpr size_t max_bins_default{200};
    static constexpr size_t nparams = 8;
    static constexpr bool   should_optimize_default{true};
};

class MultiChannel {
    public:
        using ref_vector = std::reference_wrapper<std::vector<double>>;
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
        void Optimize(Integrand<T>&,std::ostream & =std::cout);

        // Manual event generation mode
        template<typename T>
        size_t GeneratePoint(const Integrand<T> &func, std::vector<T>& point) const;
        template<typename T>
        double GenerateWeight(const Integrand<T> &func,
                              std::vector<T>& point, size_t ichannel,
                              std::optional<ref_vector> densities = std::nullopt,
                              std::optional<ref_vector> rans = std::nullopt) const;

        // Manual Training mode
        template<typename T>
        void AddTrainData(Integrand<T> &func, size_t ichannel, double val, double wgt,
                          std::vector<double> &train_data, const std::vector<double> &rans,
                          const std::vector<double> &densities);
        template<typename T>
        void Train(Integrand<T> &func, std::vector<double> &train_data);
        void UpdateResults(StatsData&);

        // Getting results
        MultiChannelSummary Summary(std::ostream &str);

        // YAML interface
        friend YAML::convert<apes::MultiChannel>;

    private:
        void Adapt(const std::vector<double>&);
        void TrainChannels();
        template<typename T>
        void RefineChannels(Integrand<T> &func) {
            params.iteration = 0;
            params.ncalls = static_cast<size_t>(static_cast<double>(params.ncalls) * params.refine_size);
            // params.ncalls = static_cast<size_t>(pow(static_cast<double>(params.ncalls), params.refine_size));
            for(auto &channel : func.Channels()) {
                if(channel.integrator.Grid().Bins() < params.max_bins)
                    channel.integrator.Refine();
            }
        }
        void PrintIteration(std::ostream &str) const;
        void MaxDifference(const std::vector<double>&);

        size_t ndims{};
        MultiChannelParams params{};
        std::vector<double> channel_weights, best_weights;
        double min_diff{lim::infinity()};
        MultiChannelSummary summary;
};

template<typename T>
size_t apes::MultiChannel::GeneratePoint(const Integrand<T> &func,
                                       std::vector<T>& point) const {
  // Generate single point without any of the surroundings
  std::vector<double> rans(ndims);
  Random::Instance().Generate(rans);
  size_t ichannel = Random::Instance().SelectIndex(channel_weights);
  func.GeneratePoint(ichannel, rans, point);
  return ichannel;
}

template<typename T>
double apes::MultiChannel::GenerateWeight(const Integrand<T> &func,
                      std::vector<T>& point, size_t ichannel, std::optional<ref_vector> densities,
                      std::optional<ref_vector> rans) const {
    if(densities and rans) {
        return func.GenerateWeight(channel_weights, point, ichannel, densities->get(), rans->get());
    } else {
        // Compute weight for a given point
        size_t nchannels = channel_weights.size();
        std::vector<double> _densities(nchannels);
        std::vector<double> _rans{};
        return func.GenerateWeight(channel_weights, point, ichannel, _densities, _rans);
    }
}

template<typename T>
void apes::MultiChannel::AddTrainData(Integrand<T> &func, size_t ichannel, double val, double wgt,
                                      std::vector<double> &train_data, const std::vector<double> &rans,
                                      const std::vector<double> &densities) {
    double val2 = val*val;
    func.AddTrainData(ichannel, val2, rans);

    size_t nchannels = train_data.size();
    for(size_t j = 0; j < nchannels; ++j) {
        train_data[j] += val2 * wgt / densities[j];
        spdlog::debug("val2 = {}, wgt = {}, densities[{}] = {}, tain_data[{}] = {}",
                val2, wgt, j, densities[j], j, train_data[j]);
    }
}

template<typename T>
void apes::MultiChannel::Train(Integrand<T> &func, std::vector<double> &train_data) {
#ifdef ENABLE_MPI
    // Combine mpi results
    auto &mpi = MPIHandler::Instance();
    mpi.AllReduce<std::vector<double>, Add>(train_data);
    for(auto &channel : func.Channels())
        mpi.AllReduce<std::vector<double>, Add>(channel.train_data);
#endif
      Adapt(train_data);
      func.Train();
      MaxDifference(train_data);
}

template<typename T>
void apes::MultiChannel::operator()(Integrand<T> &func) {
    size_t nchannels = channel_weights.size();
    std::vector<double> rans(ndims);
    std::vector<T> point(ndims);
    std::vector<double> densities(nchannels);
    std::vector<double> train_data(nchannels);

    StatsData results;
    func.InitializeTrain();

#ifdef ENABLE_MPI
    auto &mpi = MPIHandler::Instance();
    size_t ncalls = params.ncalls/mpi.Size();
#else
    size_t ncalls = params.ncalls;
#endif

    for(size_t i = 0; i < ncalls; ++i) {
        // Generate random point and returns the used channel
        size_t ichannel = GeneratePoint(func, point);

        // Preprocess event to determine if a valid point was created (i.e. cuts)
        if(!func.PreProcess()(point)) {
            --i;
            results += 0;
            continue;
        }

        // Evaluate the function at this point
        double wgt = GenerateWeight(func, point, ichannel, densities, rans);
        double f = func(point);
        double val = wgt == 0 ? 0 : f*wgt;

        if(std::isnan(val)){
          std::cerr << "Encountered nan in integration: "
		    << "f = " << f << ", w = " << wgt << std::endl;
	  val = 0;
        }

        if(params.should_optimize) AddTrainData(func, ichannel, val, wgt, train_data, rans, densities);

        // Postprocess point (i.e. write out and unweighting)
        if(!func.PostProcess()(point, val)) {
            // If rejected generate another point
            i--;
            val = 0;
        }

        results += val;
    }
    if(params.should_optimize) Train(func, train_data);
    UpdateResults(results);
}

template<typename T>
void apes::MultiChannel::Optimize(Integrand<T> &func,std::ostream &ostr) {
    double rel_err = lim::max();

    while((rel_err > params.rtol) && summary.results.size() < params.niterations) {
        (*this)(func);
        StatsData current = summary.sum_results;
        rel_err = current.Error() / std::abs(current.Mean());

#ifdef ENABLE_MPI
        if(MPIHandler::Instance().Rank() == 0)
#endif
        PrintIteration(ostr);
        if(++params.iteration == params.nrefine) RefineChannels(func);
    }
}

}

namespace YAML {

template<>
struct convert<apes::MultiChannelSummary> {
    static Node encode(const apes::MultiChannelSummary &rhs) {
        Node node;
        node["NEntries"] = rhs.results.size();
        for(const auto &entry : rhs.results)
            node["Entries"].push_back(entry);

        node["NChannels"] = rhs.best_weights.size();
        for(const auto &weight : rhs.best_weights)
            node["ChannelWeights"].push_back(weight);

        return node;
    }

    static bool decode(const Node &node, apes::MultiChannelSummary &rhs) {
        // Get the number of entries and ensure that is the number of entries
        auto nentries = node["NEntries"].as<size_t>();
        if(node["Entries"].size() != nentries) return false;

        // Load the entries and keep track of the sum
        for(const auto &entry : node["Entries"]) {
            rhs.results.push_back(entry.as<apes::StatsData>());
            rhs.sum_results += rhs.results.back();
        }

        // Get the number of channels and ensure that is the number of channels
        auto nchannels = node["NChannels"].as<size_t>();
        if(node["ChannelWeights"].size() != nchannels) return false;

        // Load the best weights
        for(const auto &weight : node["ChannelWeights"])
            rhs.best_weights.push_back(weight.as<double>());

        return true;
    }
};

template<>
struct convert<apes::MultiChannelParams> {
    static Node encode(const apes::MultiChannelParams &rhs) {
        Node node;

        node["NCalls"] = rhs.ncalls;
        node["NIterations"] = rhs.niterations;
        node["rtol"] = rhs.rtol;
        node["nrefine"] = rhs.nrefine;
        node["beta"] = rhs.beta;
        node["min_alpha"] = rhs.min_alpha;
        node["refine_size"] = rhs.refine_size;
        node["max_bins"] = rhs.max_bins;
        node["iteration"] = rhs.iteration;

        return node;
    }

    static bool decode(const Node &node, apes::MultiChannelParams &rhs) {
        if(node.size() != rhs.nparams) return false;

        rhs.ncalls = node["NCalls"].as<size_t>();
        rhs.niterations = node["NIterations"].as<size_t>();
        rhs.rtol = node["rtol"].as<double>();
        rhs.nrefine = node["nrefine"].as<size_t>();
        rhs.beta = node["beta"].as<double>();
        rhs.min_alpha = node["min_alpha"].as<double>();
        rhs.refine_size = node["refine_size"].as<double>();
        rhs.max_bins = node["max_bins"].as<size_t>();
        rhs.iteration = node["iteration"].as<size_t>();

        return true;
    }
};

}

#endif

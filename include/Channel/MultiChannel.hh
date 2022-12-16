#ifndef MULTICHANNEL_HH
#define MULTICHANNEL_HH

#include "Integrator/Vegas.hh"
#include "Channel/Integrand.hh"

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

    static constexpr size_t ncalls_default{10000}, nint_default{20};
    static constexpr double rtol_default{1e-5};
    static constexpr size_t nrefine_default{1};
    static constexpr double beta_default{0.25}, min_alpha_default{1e-5};
    static constexpr double refine_size_default{1.5};
    static constexpr size_t max_bins_default{200};
    static constexpr size_t nparams = 8;
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
        void PrintIteration() const;
        void MaxDifference(const std::vector<double>&);

        size_t ndims{};
        MultiChannelParams params{};
        std::vector<double> channel_weights, best_weights;
        double min_diff{lim::infinity()};
        MultiChannelSummary summary;
};

template<typename T>
void apes::MultiChannel::operator()(Integrand<T> &func) {
    size_t nchannels = channel_weights.size();
    std::vector<double> rans(ndims);
    std::vector<T> point(ndims);
    std::vector<double> densities(nchannels);
    std::vector<double> train_data(nchannels);

    StatsData results;
    func.InitializeTrain();

    for(size_t i = 0; i < params.ncalls; ++i) {
        // Generate needed random numbers
        Random::Instance().Generate(rans);

        // Select a channel
        size_t ichannel = Random::Instance().SelectIndex(channel_weights);

        // Map the point based on the channel
        func.GeneratePoint(ichannel, rans, point);

        // Preprocess event to determine if a valid point was created (i.e. cuts)
        if(!func.PreProcess()(point)) {
            --i;
            results += 0;
            continue;
        }
        // Evaluate the function at this point
        double wgt = func.GenerateWeight(channel_weights, point, densities);
        double val = wgt == 0 ? 0 : func(point)*wgt;
        // if(std::isnan(wgt)){
        //   std::cout << "nan encountered" << std::endl;
        //   for(auto p : point)
        //     std::cout << p << std::endl;
        //   std::cout << func(point) << std::endl;
        // }

        double val2 = val * val;
        func.AddTrainData(ichannel, val2);

        if(val2 != 0) {
            for(size_t j = 0; j < nchannels; ++j) {
                train_data[j] += val2 * wgt / densities[j];
            }
        }

        // Postprocess point (i.e. write out and unweighting)
        if(!func.PostProcess()(point, val)) {
            // If rejected generate another point
            i--;
            val = 0;
        }

        results += val;
    }
    Adapt(train_data);
    func.Train();
    MaxDifference(train_data);
    results.n_nonzero = params.ncalls;
    summary.results.push_back(results);
    summary.sum_results += results;
}

template<typename T>
void apes::MultiChannel::Optimize(Integrand<T> &func) {
    double rel_err = lim::max();

    while((rel_err > params.rtol) && summary.results.size() < params.niterations) {
        (*this)(func);
        StatsData current = summary.sum_results;
        rel_err = current.Error() / std::abs(current.Mean());

        PrintIteration();
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

template<>
struct convert<apes::MultiChannel> {
    static Node encode(const apes::MultiChannel &rhs) {
        Node node;
        node["NDims"] = rhs.ndims;
        node["NChannels"] = rhs.best_weights.size();
        node["Parameters"] = rhs.params;
        node["Summary"] = rhs.summary;
        return node;
    }

    static bool decode(const Node &node, apes::MultiChannel &rhs) {
        if(node.size() != 4) return false; 

        rhs.ndims = node["NDims"].as<size_t>();
        rhs.summary = node["Summary"].as<apes::MultiChannelSummary>();
        rhs.params = node["Parameters"].as<apes::MultiChannelParams>();

        auto nchannels = node["NChannels"].as<size_t>();
        if(rhs.summary.best_weights.size() != nchannels) return false; 
        rhs.channel_weights = rhs.summary.best_weights;
        rhs.best_weights = rhs.summary.best_weights;
        return true;
    }
};

}

#endif

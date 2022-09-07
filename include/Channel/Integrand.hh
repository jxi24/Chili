#ifndef INTEGRAND_HH
#define INTEGRAND_HH

#include "Channel/Mapper.hh"
#include "Integrator/Vegas.hh"

namespace apes {

template<typename T>
struct Channel {
    Vegas integrator;
    std::unique_ptr<Mapper<T>> mapping;
    double weight{};
    std::vector<double> train_data;
    std::vector<double> rans;

    size_t NDims() const { return mapping -> NDims(); }
};

template<typename T>
class Integrand {
    public:
        Integrand() = default;
        Integrand(Func<T> func) : m_func{std::move(func)} {}

        // Function Utilities
        double operator()(const std::vector<T> &point) const { return m_func(point); }
        const Func<T> &Function() const { return m_func; }
        Func<T> &Function() { return m_func; }
        const std::function<bool(const std::vector<T>&, double)> &PostProcess() const { return m_post; }
        std::function<bool(const std::vector<T>&, double)> &PostProcess() { return m_post; }
        const std::function<bool(const std::vector<T>&)> &PreProcess() const { return m_pre; }
        std::function<bool(const std::vector<T>&)> &PreProcess() { return m_pre; }

        // Channel Utilities
        void AddChannel(Channel<T> channel) { 
            if(channels.size() != 0)
                if(channels[0].NDims() != channel.NDims())
                    throw std::runtime_error("Integrand: Channels have different dimensions");
            channels.push_back(std::move(channel)); 
        }
        void RemoveChannel(int idx) { channels.erase(channels.begin() + idx); }
        std::vector<Channel<T>> Channels() const { return channels; }
        std::vector<Channel<T>> &Channels() { return channels; }
        Channel<T> GetChannel(size_t idx) const { return channels[idx]; }
        Channel<T> &GetChannel(size_t idx) { return channels[idx]; }
        size_t NChannels() const { return channels.size(); }
        size_t NDims() const { return channels[0].NDims(); }

        // Train integrator
        void InitializeTrain() {
            for(auto &channel : channels) {
                const auto grid = channel.integrator.Grid();
                channel.train_data.resize(grid.Dims()*grid.Bins());
            }
        }
        void AddTrainData(size_t channel, const double val2) {
            const auto grid = channels[channel].integrator.Grid();
            for(size_t j = 0; j < grid.Dims(); ++j) 
                channels[channel].train_data[j * grid.Bins() + grid.FindBin(j, channels[channel].rans[j])] += val2;
        }
        void Train() {
            for(auto &channel : channels) {
                if(std::all_of(channel.train_data.begin(), channel.train_data.end(),
                               [](double i) { return i == 0; })) continue;
                channel.integrator.Adapt(channel.train_data);
                std::fill(channel.train_data.begin(), channel.train_data.end(), 0);
            }
        }

        // Interface to MultiChannel integration
        void GeneratePoint(size_t channel, std::vector<double> &rans, std::vector<T> &point) const {
            channels[channel].integrator.Grid()(rans);
            channels[channel].mapping -> GeneratePoint(point, rans); 
        }
        double GenerateWeight(const std::vector<double> &wgts, const std::vector<T> &point,
                              std::vector<double> &densities) {
            double weight{};
            std::vector<double> rans;
            for(size_t i = 0; i < NChannels(); ++i) {
                densities[i] = channels[i].mapping -> GenerateWeight(point, rans);
                channels[i].rans = rans;
                double vw = channels[i].integrator.GenerateWeight(rans);
                weight += wgts[i] * densities[i] / vw;
            }
            return 1.0 / weight;
        }

        // YAML interface
        friend YAML::convert<apes::Integrand<T>>;

    private:
        std::vector<Channel<T>> channels;
        Func<T> m_func{};
        std::function<bool(const std::vector<T>&, double)> m_post;
        std::function<bool(const std::vector<T>&)> m_pre;
};

}

namespace YAML {

template<typename T>
struct convert<apes::Channel<T>> {
    static Node encode(const apes::Channel<T> &rhs) {
        Node node;
        node["Integrator"] = rhs.integrator;
        // node["Mapper"] = rhs.mapping -> ToYAML();
        return node;
    }

    static bool decode(const Node &node, apes::Channel<T> &rhs) {
        if(node.size() != 2) return false;
        rhs.integrator = node["Integrator"].as<apes::Vegas>();
        // TODO: Figure out how to load a saved integrator
        return true;
    }
};

// Dummy YAML encoding / decoding for testing basics
template<>
struct convert<apes::Channel<double>> {
    static Node encode(const apes::Channel<double> &rhs) {
        Node node;
        node["Integrator"] = rhs.integrator;
        // node["Mapper"] = rhs.mapping -> ToYAML();
        return node;
    }

    static bool decode(const Node &node, apes::Channel<double> &rhs) {
        if(node.size() != 2) return false;
        rhs.integrator = node["Integrator"].as<apes::Vegas>();
        return true;
    }
};

template<typename T>
struct convert<apes::Integrand<T>> {
    static Node encode(const apes::Integrand<T> &rhs) {
        Node node;
        node["NChannels"] = rhs.channels.size();
        for(const auto & channel : rhs.channels) {
            node["Channels"].push_back(channel);
        }
        return node;
    }

    static bool decode(const Node &node, apes::Integrand<T> &rhs) {
        if(node.size() != 2) return false;

        auto nchannels = node["NChannels"].as<size_t>();
        if(node["Channels"].size() != nchannels) return false;
        for(const auto & channel : node["Channels"])
            rhs.channels.push_back(channel.as<apes::Channel<T>>());

        return true;
    }
};

}

#endif

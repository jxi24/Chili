#ifndef INTEGRAND_HH
#define INTEGRAND_HH

#include "Channel/Mapper.hh"
#include "Integrator/Vegas.hh"

namespace chili {

template<typename T>
struct Channel {
    Vegas integrator;
    std::unique_ptr<Mapper<T>> mapping;
    double weight{};
    std::vector<double> train_data;
    std::vector<double> rans;

    size_t NDims() const { return mapping -> NDims(); }
    bool Serialize(std::ostream &out) const {
        integrator.Grid().Serialize(out);
        Mapper<T>::Serialize(out, *mapping.get());
        return true;
    }
    bool Deserialize(std::istream &in) {
        AdaptiveMap map;
        map.Deserialize(in);
        integrator = Vegas(map, {});
        std::unique_ptr<Mapper<T>> mapper;
        Mapper<T>::Deserialize(in, mapper);
        mapping = std::move(mapper);
        return true;
    }
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
                channel.train_data.clear();
                channel.train_data.resize(grid.Dims()*grid.Bins());
            }
        }
        void AddTrainData(size_t channel, const double val2, const std::vector<double> &rans) {
            const auto grid = channels[channel].integrator.Grid();
            for(size_t j = 0; j < grid.Dims(); ++j) {
                if(j * grid.Bins() + grid.FindBin(j, rans[j]) > channels[channel].train_data.size())
                    spdlog::info("{}, {}, {}, {}", channels[channel].train_data.size(), j * grid.Bins() + grid.FindBin(j, rans[j]), rans[j], grid.FindBin(j, rans[j]));
                channels[channel].train_data[j * grid.Bins() + grid.FindBin(j, rans[j])] += val2;
            }
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

        double GenerateWeight(const std::vector<double> &wgts, const std::vector<T> &point, size_t ichannel,
                              std::vector<double> &densities, std::vector<double> &rans) const {
            double weight{};
            std::vector<double> _rans(point.size());
            for(size_t i = 0; i < NChannels(); ++i) {
                densities[i] = channels[i].mapping -> GenerateWeight(point, _rans);
                if(i == ichannel) rans = _rans;
                double vw = channels[i].integrator.GenerateWeight(_rans);
                weight += wgts[i] / densities[i] / vw;
            }
            return 1.0 / weight;
        }

        // YAML interface
        friend YAML::convert<chili::Integrand<T>>;

    private:
        std::vector<Channel<T>> channels;
        Func<T> m_func{};
        std::function<bool(const std::vector<T>&, double)> m_post;
        std::function<bool(const std::vector<T>&)> m_pre;
};

}

#endif

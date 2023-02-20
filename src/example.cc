#include "Channel/ChannelNode.hh"
#include "Channel/Integrand.hh"
#include "Channel/MultiChannel.hh"
#include "Integrator/AdaptiveMap.hh"
#include "Model/Model.hh"
#include "Channel/Channel.hh"
#include "Tools/JetCluster.hh"
#include <iostream>

std::vector<int> combine(int i, int j) { 
    if((i == 21 && j == 23) || (i == 23 && j == 21)) return {};
    if(i == 21 && j == 21) return {21};
    if(i == -j) {
        return {21, 23};
    } else if (i != j) {
        if(i == 21 || i == 23) {
            return {j};
        } else if(j == 21 || j == 23) {
            return {i};
        }
    }
    return {};
}

bool PreProcess(const std::vector<chili::FourVector> &mom) {
    if(std::isnan(mom[0][0])) {
        spdlog::info("Failed inital state");
        return false;
    }
    if((mom[0]+mom[1]).Mass2() > 13000*13000) {
        spdlog::info("Failed s limit");
        return false;
    }
    chili::JetCluster cluster(0.4);
    auto jets = cluster(mom);
    if(jets.size() < mom.size()-2) return false;
    return true;
    for(size_t i = 2; i < mom.size(); ++i) {
        if(mom[i].Pt() < 30) return false;
        if(std::abs(mom[i].Rapidity()) > 5) {
            spdlog::info("Failed y cut for {}: y = {}", i, mom[i].Rapidity());
            return false;
        }
        for(size_t j = i+1; j < mom.size(); ++j) {
            if(mom[i].DeltaR(mom[j]) < 0.4) {
                spdlog::info("Failed R cut for {},{}: DR = {}", i, j, mom[i].DeltaR(mom[j]));
                return false;
            }
        }
    }
    return true;
}

bool PostProcess(const std::vector<chili::FourVector>&, double) { return true; }

int main() {
    chili::Model model(combine);
    model.Mass(1) = 0;
    model.Mass(-1) = 0;
    model.Mass(2) = 0;
    model.Mass(-2) = 0;
    model.Mass(3) = 0;
    model.Mass(-3) = 0;
    model.Mass(23) = 100;
    model.Mass(21) = 0;
    model.Width(1) = 0;
    model.Width(-1) = 0;
    model.Width(2) = 0;
    model.Width(-2) = 0;
    model.Width(3) = 0;
    model.Width(-3) = 0;
    model.Width(23) = 2.5;
    model.Width(21) = 0;
    model.Mass(5) = 172;
    model.Width(5) = 5;
    model.Mass(-5) = 172;
    model.Width(-5) = 5;

    spdlog::set_level(spdlog::level::info);

    std::vector<std::vector<int>> processes{
        {2, -2, 1, -1},
        {-1, -2, 1, 2},
        {1, -1, 2, -2, 2, -2},
        {1, -1, 2, -2, 3, -3},
        {1, -1, 1, -1, 21, 21, 21},
        {1, -1, 2, -2, 21, 21, 21}
    };

    // Construct channels
    // for(const auto &process : processes) {
    //     auto mappings = chili::ConstructChannels(13000, process, model, 0);
    //     if(mappings.size() == 0)
    //         throw std::logic_error(fmt::format("Failed process {{{}}}",
    //                                            fmt::join(process.begin(), process.end(), ", ")));
    // }

    auto mappings = chili::ConstructChannels(13000, {21, 21, 21, 21}, model, 0);

    // Setup integrator
    chili::Integrand<chili::FourVector> integrand;
    for(auto &mapping : mappings) {
        chili::Channel<chili::FourVector> channel;
        channel.mapping = std::move(mapping);
        // Initializer takes the number of integration dimensions
        // and the number of bins for vegas to start with
        chili::AdaptiveMap map(channel.mapping -> NDims(), 2);
        // Initializer takes adaptive map and settings (found in struct VegasParams)
        channel.integrator = chili::Vegas(map, chili::VegasParams{});
        integrand.AddChannel(std::move(channel));
    }

    // Initialize the multichannel integrator
    // Takes the number of dimensions, the number of channels, and options
    // The options can be found in the struct MultiChannelParams
    chili::MultiChannel integrator{integrand.NDims(), integrand.NChannels(), {}};

    // To integrate a function you need to pass it in and tell it to optimize
    // Summary will print out a summary of the results including the values of alpha
    auto func = [&](const std::vector<chili::FourVector> &) {
        return 1; // (pow(p[0]*p[2], 2)+pow(p[0]*p[3], 2))/pow(p[2]*p[3], 2);
    };
    integrand.Function() = func;
    integrand.PreProcess() = PreProcess;
    integrand.PostProcess() = PostProcess;
    spdlog::info("Starting optimization");
    integrator.Optimize(integrand);
    integrator.Summary(std::cout);

    spdlog::info("Saving trained integrator");
    size_t idx = 0;
    for(const auto &channel : integrand.Channels()) {
        std::string filename = fmt::format("channel_{}.bin", idx++);
        std::ofstream output;
        output.open(filename, std::ios::binary | std::ios::out);
        channel.Serialize(output);
        output.close();
    }

    spdlog::info("Finished saving");

    chili::Channel<chili::FourVector> channel;
    std::string filename = "channel_0.bin";
    std::ifstream output;
    output.open(filename, std::ios::binary | std::ios::in);
    channel.Deserialize(output);
    output.close();

    // integrator(integrand); // Generate events

    return 0;
}

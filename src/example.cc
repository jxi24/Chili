#include "Channel/ChannelNode.hh"
#include "Channel/Integrand.hh"
#include "Channel/MultiChannel.hh"
#include "Model/Model.hh"
#include "Channel/Channel.hh"
#include <iostream>

std::vector<int> combine(int i, int j) { 
    if((i == 21 && j == 23) || (i == 23 && j == 21)) return {};
    if(i == 21 && j == 21) return {21};
    if(i == 1 && j == -1) return {23};
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

bool PreProcess(const std::vector<apes::FourVector> &mom) {
    for(size_t i = 2; i < mom.size(); ++i) {
        if(mom[i].Pt() < 1) return false;
        if(std::abs(mom[i].Rapidity()) > 5) return false;
        for(size_t j = i+1; j < mom.size(); ++j) {
            if(mom[i].DeltaR(mom[j]) < 0.4) return false;
        }
    }
    return true;
}

bool PostProcess(const std::vector<apes::FourVector>&, double) { return true; }

int main() {
    apes::Model model(combine);
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

    // spdlog::set_level(spdlog::level::trace);

    // Construct channels
    auto mappings = apes::ConstructChannels(13000, {2, -2, 1, -1}, model, 1);
    // auto mappings = apes::ConstructChannels(13000, {-2, -1, 2, 1}, model, 1);
    std::cout << mappings.size() << std::endl;

    // Setup integrator
    apes::Integrand<apes::FourVector> integrand;
    for(auto &mapping : mappings) {
        apes::Channel<apes::FourVector> channel;
        channel.mapping = std::move(mapping);
        // Initializer takes the number of integration dimensions
        // and the number of bins for vegas to start with
        apes::AdaptiveMap map(channel.mapping -> NDims(), 2);
        // Initializer takes adaptive map and settings (found in struct VegasParams)
        channel.integrator = apes::Vegas(map, apes::VegasParams{});
        integrand.AddChannel(std::move(channel));
    }

    // Initialize the multichannel integrator
    // Takes the number of dimensions, the number of channels, and options
    // The options can be found in the struct MultiChannelParams
    apes::MultiChannel integrator{integrand.NDims(), integrand.NChannels(), {}};

    // To integrate a function you need to pass it in and tell it to optimize
    // Summary will print out a summary of the results including the values of alpha
    auto func = [&](const std::vector<apes::FourVector> &) {
        return 1.0;
    };
    integrand.Function() = func;
    integrand.PreProcess() = PreProcess;
    integrand.PostProcess() = PostProcess;
    integrator.Optimize(integrand);
    integrator.Summary();
    integrator(integrand); // Generate events

    return 0;
}

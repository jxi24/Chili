#include "Channel/ChannelNode.hh"
#include "Channel/Integrand.hh"
#include "Channel/MultiChannel.hh"
#include "Model/Model.hh"
#include "Channel/Channel.hh"
#include <iostream>

std::vector<int> combine(int, int) { return {21}; }

int main() {
    apes::Model model(combine);
    model.Mass(21) = 0;

    // Construct channels
    auto mappings = apes::ConstructChannels({21, 21, 21, 21, 21}, model);
    std::cout << mappings.size() << std::endl;

    // Setup integrator
    std::vector<double> masses{0, 0, 0, 0, 0};
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
    apes::MultiChannel integrator{integrand.NDims(), integrand.NChannels(), {1000, 2}};

    // To integrate a function you need to pass it in and tell it to optimize
    // Summary will print out a summary of the results including the values of alpha
    auto func = [&](const std::vector<apes::FourVector> &mom, const double &) {
        return mom[0].Mass();
    };
    integrand.Function() = func;
    integrator.Optimize(integrand);
    integrator.Summary();

    return 0;
}

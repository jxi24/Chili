#include "Channel/ChannelNode.hh"
#include "Channel/Integrand.hh"
#include "Channel/MultiChannel.hh"
#include "Model/Model.hh"
#include "Channel/Channel.hh"
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

    // Construct channels
    auto mappings = apes::ConstructChannels(13000, {2, -2, 1, -1, 21, 21}, model, 2);
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
    apes::MultiChannel integrator{integrand.NDims(), integrand.NChannels(), {1000, 2}};

    // To integrate a function you need to pass it in and tell it to optimize
    // Summary will print out a summary of the results including the values of alpha
    auto func = [&](const std::vector<apes::FourVector> &, const double &) {
        return 1;
    };
    integrand.Function() = func;
    integrator.Optimize(integrand);
    integrator.Summary();
    integrator(integrand); // Generate events

    return 0;
}

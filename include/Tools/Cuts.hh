#pragma once

#include "Channel/ChannelUtils.hh"
#include <cmath>
#include <map>
#include <unordered_map>
#include <vector>

#include <iostream>

namespace apes {

struct Cuts {
    std::vector<double> sexternal;
    std::map<std::pair<int, int>, double> deltaR;
    std::unordered_map<unsigned int, double> smin;
    std::unordered_map<unsigned int, double> ptmin;
    std::unordered_map<unsigned int, double> etamax;
};

inline double SMax(double sqrts, const Cuts &cuts, unsigned int idx) {
    double result = sqrts;
    for(unsigned int i = 0; i < cuts.sexternal.size(); ++i) {
        if(!BitIsSet(idx, i+2)) {
            result -= sqrt(cuts.sexternal[i]);
        }
    }
    return result*result;
}

}

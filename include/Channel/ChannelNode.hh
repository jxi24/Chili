#pragma once

// #include "Channel/Channel.hh"
#include "Model/Model.hh"
#include "Tools/Cuts.hh"
#include "fmt/format.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sstream>
#include <vector>

namespace apes {

class FSMapper;

// TODO: Remove channel node from everywhere
struct ChannelNode {
    unsigned int m_pid;
    unsigned int m_idx;
    std::vector<std::shared_ptr<ChannelNode>> m_children;
};


struct ParticleInfo {
    unsigned int idx{};
    int pid{};
    double mass{}, width{};
    bool Deserialize(std::istream &in);
    bool Serialize(std::ostream &out) const;
};
using Current = std::unordered_map<unsigned int, std::set<ParticleInfo>>;
using DecayProds = std::pair<ParticleInfo, ParticleInfo>;
using DecayMap = std::map<ParticleInfo, DecayProds>;
using DecayChain = std::map<ParticleInfo, std::set<DecayProds>, std::greater<ParticleInfo>>;

struct DataFrame {
    std::set<int> pid;
    std::vector<unsigned int> avail_currents;
    std::vector<unsigned int> currents{};
    unsigned int idx_sum{};
    size_t schans{};
};

struct ChannelDescription {
    std::vector<ParticleInfo> info;
    DecayMap decays{};
    bool Deserialize(std::istream &in);
    bool Serialize(std::ostream &out) const;
};
using ChannelVec = std::vector<ChannelDescription>;

ChannelVec AddDecays(unsigned int, const ChannelDescription&, const DecayChain&);

bool operator<(const ParticleInfo &a, const ParticleInfo &b);
bool operator==(const ParticleInfo &a, const ParticleInfo &b);
inline bool operator!=(const ParticleInfo &a, const ParticleInfo &b) {
    return !(a == b);
}
inline bool operator>(const ParticleInfo &a, const ParticleInfo &b) {
    return b < a && b != a;
}
bool operator<(const ChannelDescription &a, const ChannelDescription &b);

inline std::string ToString(ParticleInfo info) {
    return fmt::format("Particle({}, {}, {}, {})",
                       info.idx, info.pid, info.mass, info.width);
}

inline std::string ToString(ChannelDescription channel) {
    std::vector<std::string> particles;
    for(const auto &info : channel.info) particles.push_back(ToString(info));
    std::vector<std::string> decays;
    for(const auto &decay : channel.decays) {
        decays.push_back(fmt::format("Decay({} -> ({}, {}))",
                                     ToString(decay.first), ToString(decay.second.first),
                                     ToString(decay.second.second)));
    }
    return fmt::format("ChannelDescription({}, {})",
                       fmt::join(particles.begin(), particles.end(), ", "),
                       fmt::join(decays.begin(), decays.end(), ", "));
}

std::string ToString(DecayChain chain);

std::vector<std::unique_ptr<FSMapper>> ConstructChannels(double sqrts, const std::vector<int> &flavs, const Model &model, size_t smax=2);
std::vector<std::unique_ptr<FSMapper>> ConstructChannels(double sqrts, const std::vector<int> &flavs, const Model &model, Cuts& cuts, size_t smax=2);

}

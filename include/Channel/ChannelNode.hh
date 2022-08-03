#pragma once

#include "Channel/Channel.hh"
#include "Model/Model.hh"
#include <map>
#include <memory>
#include <vector>

namespace apes {

struct Current {
    std::vector<int> pid;
    unsigned int idx;
    std::vector<double> mass, width;
    std::shared_ptr<Current> left{nullptr}, right{nullptr};
};

struct SChannelChain {
    int pid;
    unsigned int idx;
    double mass, width;
    std::vector<std::shared_ptr<SChannelChain>> left{}, right{};
};

struct DataFrame {
    int pid;
    std::vector<std::shared_ptr<Current>> avail_currents;
    std::vector<std::shared_ptr<Current>> currents{};
    unsigned int idx_sum{};
    size_t schans{};
};

struct ChannelDescription {
    std::vector<unsigned int> idxs; 
    std::map<unsigned int, std::pair<unsigned int, unsigned int>> decays{};
};

struct ChannelElm {
    std::vector<Current> currents;
};

struct ChannelNode {
    unsigned int m_pid;
    unsigned int m_idx;
    std::vector<std::shared_ptr<ChannelNode>> m_children;
};

inline std::string ToString(Current *cur) {
    std::string lhs = cur -> left ? ToString(cur -> left.get()) : "";
    std::string rhs = cur -> right ? ToString(cur -> right.get()) : "";
    std::string result = fmt::format("Cur({}, [{}]:-> {}, {})",
                                     cur -> idx, fmt::join(cur -> pid.begin(),
                                                           cur -> pid.end(), ","), lhs, rhs);
    return result;
}

inline std::string ToString(SChannelChain *chain) {
    std::string lhs = "";
    std::string rhs = "";
    for(const auto &left : chain -> left) {
        lhs += ToString(left.get());
    }
    for(const auto &right : chain -> right) {
        rhs += ToString(right.get());
    }
    if(lhs == "" && rhs == "") 
        return fmt::format("Cur({}, [{}])",
                           chain -> idx, chain -> pid);

    return fmt::format("Cur({}, [{}]: -> {}, {})",
                       chain -> idx, chain -> pid, lhs, rhs);
}

std::vector<std::unique_ptr<FSMapper>> ConstructChannels(const std::vector<int> &flavs, const Model &model, size_t smax=2);
using DecayChain = std::unordered_map<size_t, std::vector<std::shared_ptr<apes::SChannelChain>>>;
std::unique_ptr<FSMapper> ConstructChannel(const std::vector<unsigned int> &ch_descr,
                                           const std::vector<int> &flavs,
                                           const Model &model,
                                           const DecayChain &chain);

}

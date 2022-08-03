#include "Channel/ChannelUtils.hh"
#include "Channel/ChannelNode.hh"
#include <iostream>
#include <bitset>
#include <stack>

using apes::ChannelElm;
using apes::Current;
using apes::BitIsSet;
using apes::BitsAreSet;
using apes::NextPermutation;
using apes::FSMapper;
using CurrentVec = std::unordered_map<size_t, std::shared_ptr<Current>>;
using ChannelVec = std::vector<std::shared_ptr<ChannelElm>>;
using apes::DecayChain;

std::string WriteCurrent(size_t cur) {
    std::stringstream ss;
    ss << cur << "(" << std::bitset<5>(cur) << ") ";
    return ss.str();
}

std::vector<std::unique_ptr<FSMapper>> apes::ConstructChannels(const std::vector<int> &flavs, const Model &model, size_t smax) {
    CurrentVec currentComponents;
    DecayChain decayChain;

    // Setup initial states
    std::vector<double> masses;
    for(size_t i = 0; i < flavs.size(); ++i) {
        auto current = std::make_shared<Current>();
        current -> pid = {flavs[i]};
        current -> idx = 1 << i;
        current -> mass = {model.Mass(flavs[i])};
        masses.push_back(flavs[i]);
        currentComponents[current -> idx] = current;

        auto chain = std::make_shared<SChannelChain>();
        chain -> pid = flavs[i];
        chain -> idx = 1 << i;
        chain -> mass = model.Mass(flavs[i]);
        decayChain[current -> idx].push_back(chain);
    }

    // Recursion over all set particles
    // TODO: Clean this up to make it easier to read
    for(unsigned int nset = 2; nset < flavs.size(); ++nset) {
        unsigned int cur = (1u << nset) - 1;
        // Combine all currents
        while(cur < (1u << (flavs.size())) && NSetBits(cur) - 1 <= smax) {
            auto set = BitsAreSet(cur, static_cast<unsigned int>(flavs.size()));
            for(unsigned int iset = 1; iset < nset; ++iset) {
                unsigned int idx = (1u << iset) - 1;
                while(idx < (1u << (nset - 1))) {
                    unsigned int subCur1 = 0;
                    for(unsigned int i = 0; i < flavs.size(); ++i)
                        subCur1 += set[i]*((idx >> i) & 1);
                    auto subCur2 = cur ^ subCur1;
                    // Skip over initial state legs
                    if(BitIsSet(subCur1, 0) || BitIsSet(subCur2, 0)) break;
                    if(BitIsSet(subCur1, 1) || BitIsSet(subCur2, 1)) break;

                    // Check that currents are allowed
                    if(currentComponents.find(subCur1) == currentComponents.end()) {
                        idx = NextPermutation(idx);
                        continue;
                    }
                    if(currentComponents.find(subCur2) == currentComponents.end()) {
                        idx = NextPermutation(idx);
                        continue;
                    }

                    // Create new channel component
                    auto combined = model.Combinable(currentComponents[subCur1] -> pid[0],
                                                     currentComponents[subCur2] -> pid[0]);
                    if(combined.size() == 0) {
                        idx = NextPermutation(idx);
                        continue;
                    }
                            
                    auto current = std::make_shared<Current>();
                    current -> left = currentComponents[subCur1];
                    current -> right = currentComponents[subCur2];
                    current -> idx = cur;
                    for(const auto & elm : combined) {
                        current -> pid.push_back(elm);
                        current -> mass.push_back(model.Mass(elm));

                        auto chain = std::make_shared<SChannelChain>();
                        chain -> left = decayChain[subCur1];
                        chain -> right = decayChain[subCur2];
                        chain -> pid = elm;
                        chain -> idx = cur;
                        chain -> mass = model.Mass(elm);
                        decayChain[cur].push_back(chain);
                    }
                    currentComponents[cur] = current;
                    idx = NextPermutation(idx);
                }
            }
            cur = NextPermutation(cur);
        }
    }

    std::vector<std::shared_ptr<Current>> avail_currents;
    for(const auto & chelm : currentComponents) {
        avail_currents.push_back(chelm.second);
    }

    std::stack<DataFrame> s;
    size_t max_id = (1 << flavs.size()) - 1;
    s.push({currentComponents[2] -> pid[0], {avail_currents}});
    std::vector<std::vector<unsigned int>> channels;
    while(!s.empty()) {
        DataFrame top = s.top();
        s.pop();
        if(top.idx_sum > max_id) continue;
        if(top.idx_sum == max_id) {
            std::vector<unsigned int> channel;
            for(const auto &c : top.currents) {
                channel.push_back(c -> idx);
            }
            channels.push_back(channel);
            continue;
        }
        if(top.avail_currents.empty()) continue;
        for(size_t i = 0; i < top.avail_currents.size(); ++i) {
            DataFrame d = top;
            d.schans += apes::NSetBits(top.avail_currents[i]->idx)-1;
            if(d.schans > smax) {
                d.schans = top.schans;
                d.avail_currents.erase(d.avail_currents.begin()+static_cast<int>(i));
                continue;
            }
            if(!d.currents.empty()) {
                if(d.currents.back() > top.avail_currents[i]) continue;
            }
            if(HaveCommonBitSet(d.idx_sum, top.avail_currents[i]->idx)) continue;
            auto combined = model.Combinable(d.pid, top.avail_currents[i] -> pid[0]);
            if(combined.size() == 0) continue;
            d.pid = combined[0];
            d.idx_sum += top.avail_currents[i]->idx;
            d.currents.push_back(top.avail_currents[i]);
            d.avail_currents.erase(d.avail_currents.begin()+static_cast<int>(i));
            s.push(d);
        }
    }

    // Convert channel descriptions to mappings
    for(auto ch_descr : channels) {
        ConstructChannel(ch_descr, flavs, model, decayChain);
    }

    return {};
} 

std::unique_ptr<FSMapper> apes::ConstructChannel(const std::vector<unsigned int> &ch_descr,
                                                 const std::vector<int> &flavs,
                                                 const Model &model) {
    // Create vector of masses
    for(auto idx : ch_descr) {
        if(!IsPower2(idx)) {
            std::cout << "Flavor " << idx << ": ";
            for(const auto &decay : chain.at(idx)) {
                // std::cout << ToString(decay.get()) << std::endl;
                std::cout << decay -> pid << ", ";
            }
            std::cout << "\b\b\n";
        } else {
            std::cout << "Flavor " << idx << ": " << flavs[Log2(idx)] << std::endl;
        }
    }

    return std::make_unique<FSMapper>();
}

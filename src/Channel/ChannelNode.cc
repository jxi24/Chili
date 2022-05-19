#include "Channel/ChannelUtils.hh"
#include "Channel/ChannelNode.hh"
#include <iostream>

using apes::ChannelNode;
using apes::SetBit;
using apes::SetBits;
using apes::NextPermutation;
using apes::FSMapper;
using ChannelMap = std::map<size_t, std::vector<std::shared_ptr<ChannelNode>>>;

std::vector<std::unique_ptr<FSMapper>> apes::ConstructChannels(const std::vector<int> &flavs, const Model &model) {
    ChannelMap channelComponents;

    // Setup initial states
    std::vector<double> masses;
    for(size_t i = 0; i < flavs.size(); ++i) {
        auto node = std::make_shared<ChannelNode>();
        node -> m_pid = flavs[i];
        node -> m_idx = 1 << i;
        node -> m_mass = model.Mass(flavs[i]);
        masses.push_back(flavs[i]);
        channelComponents[(1 << i)].push_back(node);
    }

    // Recursion over all set particles
    // TODO: Clean this up to make it easier to read
    for(unsigned int nset = 2; nset < flavs.size(); ++nset) {
        unsigned int cur = (1u << nset) - 1;
        // Combine all currents
        while(cur < (1u << (flavs.size()))) {
            auto set = SetBits(cur, static_cast<unsigned int>(flavs.size()));
            for(unsigned int iset = 1; iset < nset; ++iset) {
                unsigned int idx = (1u << iset) - 1;
                while(idx < (1ul << (nset - 1))) {
                    unsigned int subCur1 = 0;
                    for(unsigned int i = 0; i < flavs.size(); ++i)
                        subCur1 += set[i]*((idx >> i) & 1);
                    auto subCur2 = cur ^ subCur1;
                    // Skip over fixed leg
                    if(SetBit(subCur1, 0) || SetBit(subCur2, 0)) break;
                    // Create new channel component
                    for(const auto &subChan1 : channelComponents[subCur1]) {
                        for(const auto &subChan2 : channelComponents[subCur2]) {
                            auto combined = model.Combinable(subChan1 -> m_pid,
                                                             subChan2 -> m_pid);
                            if(combined.size() == 0) continue;
                            
                            // Last one to be combined
                            if(cur == (1ul << flavs.size()) - 2) {
                                auto node = std::make_shared<ChannelNode>();
                                node -> m_left = subChan1;
                                node -> m_right = subChan2;
                                node -> m_pid = flavs[0];
                                node -> m_idx = cur;
                                node -> m_mass = model.Mass(flavs[0]);
                                channelComponents[cur].push_back(node);
                            } else {
                                for(const auto & elm : combined) {
                                    auto node = std::make_shared<ChannelNode>();
                                    node -> m_left = subChan1;
                                    node -> m_right = subChan2;
                                    node -> m_pid = elm;
                                    node -> m_idx = cur;
                                    node -> m_mass = model.Mass(elm);
                                    channelComponents[cur].push_back(node);
                                }
                            }
                        }
                    }
                    idx = NextPermutation(idx);
                }
            }
            cur = NextPermutation(cur);
        }
    }

    unsigned int lid = (1u << flavs.size()) - 2;
    if(channelComponents.find(lid) == channelComponents.end()) throw;
    std::vector<std::unique_ptr<FSMapper>> channels;
    for(const auto &cur : channelComponents[lid]) {
        auto channel = std::make_unique<FSMapper>(flavs.size(), masses);
        channel -> InitializeChannel(cur);
        std::cout << channel -> ToString() << std::endl;
        channels.push_back(std::move(channel));
    }

    return channels;
}

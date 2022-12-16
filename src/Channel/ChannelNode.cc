#include "Channel/ChannelUtils.hh"
#include "Channel/ChannelNode.hh"
#include "Channel/Channel.hh"
#include "Tools/ChannelElements.hh"
#include "Tools/Cuts.hh"
#include "spdlog/spdlog.h"
#include <iostream>
#include <bitset>
#include <stack>

using apes::FSMapper;

std::string WriteCurrent(size_t cur) {
    std::stringstream ss;
    ss << cur << "(" << std::bitset<5>(cur) << ") ";
    return ss.str();
}

// TODO: Refactor code to be less messy
std::vector<std::unique_ptr<FSMapper>> apes::ConstructChannels(double sqrts, const std::vector<int> &flavs, const Model &model, size_t smax) {
    Current currentComponents;
    DecayChain decayChain;
    Cuts cuts;
    // TODO: Make this nicer with alias to jet particle

    for(auto f1 : flavs){
      for(auto f2 : flavs) {
        // do we really need this between all outgoing particles?
        // not sure if this is correct for the leptons
        cuts.deltaR[{f1, f2}] = 0.4;
      }
    }

    // Setup initial states
    for(size_t i = 0; i < flavs.size(); ++i) {
        ParticleInfo info;
        info.idx = 1 << i;
        info.pid = flavs[i];
        info.mass = model.Mass(info.pid);
        info.width = model.Width(info.pid);
        currentComponents[info.idx].insert(info);
        cuts.smin[info.idx] = info.mass*info.mass;
        // TODO: Make this read in
        cuts.ptmin[info.idx] = 30;
        cuts.etamax[info.idx] = 99;
        if(i > 1) cuts.sexternal.push_back(info.mass*info.mass);
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
                    std::set<int> combined;
                    for(const auto &cur1 : currentComponents[subCur1]) {
                        for(const auto &cur2 : currentComponents[subCur2]) {
                            auto tmp = model.Combinable(cur1.pid, cur2.pid);
                            combined.insert(tmp.begin(), tmp.end());
                        }
                    }
                    if(combined.size() == 0) {
                        idx = NextPermutation(idx);
                        continue;
                    }
                            
                    for(const auto & elm : combined) {
                        ParticleInfo info;
                        info.pid = elm;
                        info.idx = cur;
                        info.mass = model.Mass(elm);
                        info.width = model.Width(elm);
                        currentComponents[cur].insert(info);
                    }
                    cuts.ptmin[cur] = 0;
                    cuts.etamax[cur] = 99;
                    auto lparts = currentComponents[subCur1];
                    auto rparts = currentComponents[subCur2];
                    for(const auto &linfo : lparts) {
                        for(const auto &rinfo : rparts) {
                            auto allowed = model.Combinable(linfo.pid, rinfo.pid);
                            if(allowed.size() != 0) {
                                for(const auto &elm : allowed) {
                                    ParticleInfo info;
                                    info.pid = elm;
                                    info.idx = cur;
                                    info.mass = model.Mass(elm);
                                    info.width = model.Width(elm);
                                    if(linfo < rinfo) decayChain[info].insert({linfo, rinfo});
                                    else decayChain[info].insert({rinfo, linfo});
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

    spdlog::trace("Available currents:");
    std::vector<unsigned int> avail_currents;
    for(const auto & chelm : currentComponents) {
        avail_currents.push_back(chelm.first);
        spdlog::trace("  - {}", chelm.first);
    }

    std::stack<DataFrame> s;
    size_t max_id = (1 << flavs.size()) - 1;
    std::set<int> pids{currentComponents[1].begin() -> pid};
    s.push({pids, {avail_currents}});
    std::set<ChannelDescription> channels;
    while(!s.empty()) {
        DataFrame top = s.top();
        s.pop();
        if(top.idx_sum > max_id) continue;
        if(top.idx_sum == max_id) {
            // Collect all t-channel particles
            std::vector<ChannelDescription> channel_old(1);
            for(const auto &c : top.currents) {
                // Remove initial state particles
                if(c == 1 || c == 2) continue;
                std::vector<ParticleInfo> infos;
                for(const auto &cinfo : currentComponents[c]) {
                    auto pid = cinfo.pid;
                    ParticleInfo info{c, pid, model.Mass(pid), model.Width(pid)};
                    infos.push_back(info);
                }
                std::vector<ChannelDescription> channel_new;
                for(const auto &chan : channel_old) {
                    for(const auto &info : infos) {
                        ChannelDescription channel = chan;
                        channel.info.push_back(info);
                        std::sort(channel.info.begin(), channel.info.end(), std::greater<>());
                        channel_new.push_back(channel);
                    }
                }
                channel_old = channel_new;
            }
            channels.insert(channel_old.begin(), channel_old.end());
            continue;
        }
        if(top.avail_currents.empty()) continue;
        for(size_t i = 0; i < top.avail_currents.size(); ++i) {
            DataFrame d = top;
            d.schans += apes::NSetBits(top.avail_currents[i])-1;
            if(d.schans > smax) {
                d.schans = top.schans;
                d.avail_currents.erase(d.avail_currents.begin()+static_cast<int>(i));
                continue;
            }
            // if(!d.currents.empty()) {
            //     // Ensure that one and only one particle is combined with an initial state
            //     // Since the s-channel particles have already been combined
            //     if(!(d.currents.back() & 3) && !(top.avail_currents[i] & 3)) continue;
            // }
            if(HaveCommonBitSet(d.idx_sum, top.avail_currents[i])) continue;
            std::set<int> combined;
            for(const auto &pid1 : d.pid) {
                for(const auto &info : currentComponents[top.avail_currents[i]]) {
                    auto tmp = model.Combinable(pid1, info.pid);
                    combined.insert(tmp.begin(), tmp.end());
                }
            }
            if(combined.size() == 0) continue;
            d.pid = combined;
            d.idx_sum += top.avail_currents[i];
            d.currents.push_back(top.avail_currents[i]);
            d.avail_currents.erase(d.avail_currents.begin()+static_cast<int>(i));
            s.push(d);
        }
    }

    // Extend channels by adding in decays
    for(const auto &decay : decayChain) {
        spdlog::trace("Decay: {}", ToString(decay.first));
        std::set<ChannelDescription> new_channels;
        for(const auto &channel : channels) {
            spdlog::trace("Channel: {}", ToString(channel));
            bool add_decay = std::binary_search(channel.info.begin(), channel.info.end(), decay.first, std::greater<>());
            for(const auto &prods : channel.decays) {
                add_decay |= prods.second.first == decay.first || prods.second.second == decay.first;
            }
            spdlog::trace("Allowed? {}", add_decay);
            if(add_decay) {
                for(const auto &decay_prods : decay.second) {
                    auto chan = channel;
                    chan.decays[decay.first] = decay_prods;
                    new_channels.insert(chan);
                }
            } else new_channels.insert(channel);
        }
        channels = new_channels;
    }

    // Convert channel descriptions to mappings
    std::vector<std::unique_ptr<FSMapper>> mappings;
    for(auto ch_descr : channels) {
        spdlog::trace("Channel: {}", ToString(ch_descr));
        mappings.emplace_back(std::make_unique<FSMapper>(sqrts, flavs.size(), ch_descr, cuts));
    }

    return mappings;
}

bool apes::operator<(const ParticleInfo &a, const ParticleInfo &b) {
    if(a.idx == b.idx) {
        if(a.mass == b.mass) {
            return a.pid < b.pid;
        }
        return a.mass < b.mass;
    }
    return a.idx < b.idx;
}

bool apes::operator==(const ParticleInfo &a, const ParticleInfo &b) {
    return a.mass == b.mass && a.idx == b.idx && a.pid == b.pid;
}

bool apes::operator<(const ChannelDescription &a, const ChannelDescription &b) {
    if(a.info.size() == b.info.size()) {
        for(size_t i = 0; i < a.info.size(); ++i) {
            if(a.info[i] < b.info[i]) return true;
            else if(a.info[i] > b.info[i]) return false;
        }
        return false;
    }
    return a.info.size() < b.info.size();
}

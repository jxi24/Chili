#include "Channel/ChannelUtils.hh"
#include "Channel/ChannelNode.hh"
#include "Tools/ChannelElements.hh"
#include "Tools/Cuts.hh"
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
std::vector<std::unique_ptr<FSMapper>> apes::ConstructChannels(const std::vector<int> &flavs, const Model &model, size_t smax) {
    Current currentComponents;
    DecayChain decayChain;
    Cuts cuts;
    // TODO: Make this nicer with alias to jet particle
    cuts.deltaR[{-1, -1}] = 0.4;
    cuts.deltaR[{-1, 1}] = 0.4;
    cuts.deltaR[{-1, 2}] = 0.4;
    cuts.deltaR[{-1, -2}] = 0.4;
    cuts.deltaR[{-1, 21}] = 0.4;
    cuts.deltaR[{1, -1}] = 0.4;
    cuts.deltaR[{1, 1}] = 0.4;
    cuts.deltaR[{1, 2}] = 0.4;
    cuts.deltaR[{1, -2}] = 0.4;
    cuts.deltaR[{1, 21}] = 0.4;
    cuts.deltaR[{2, -1}] = 0.4;
    cuts.deltaR[{2, 1}] = 0.4;
    cuts.deltaR[{2, 2}] = 0.4;
    cuts.deltaR[{2, -2}] = 0.4;
    cuts.deltaR[{2, 21}] = 0.4;
    cuts.deltaR[{-2, -1}] = 0.4;
    cuts.deltaR[{-2, 1}] = 0.4;
    cuts.deltaR[{-2, 2}] = 0.4;
    cuts.deltaR[{-2, -2}] = 0.4;
    cuts.deltaR[{-2, 21}] = 0.4;
    cuts.deltaR[{21, -1}] = 0.4;
    cuts.deltaR[{21, 1}] = 0.4;
    cuts.deltaR[{21, 2}] = 0.4;
    cuts.deltaR[{21, -2}] = 0.4;
    cuts.deltaR[{21, 21}] = 0.4;

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
        cuts.etamax[info.idx] = 5;
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

    std::vector<unsigned int> avail_currents;
    for(const auto & chelm : currentComponents) {
        avail_currents.push_back(chelm.first);
    }

    std::stack<DataFrame> s;
    size_t max_id = (1 << flavs.size()) - 1;
    std::set<int> pids;
    for(const auto &info : currentComponents[2]) {
        pids.insert(info.pid);
    }
    s.push({pids, {avail_currents}});
    std::vector<ChannelDescription> channels;
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
                        channel_new.push_back(channel);
                    }
                }
                channel_old = channel_new;
            }
            channels.insert(channels.end(), channel_old.begin(), channel_old.end());
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
            if(!d.currents.empty()) {
                if(d.currents.back() > top.avail_currents[i]) continue;
            }
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
        ChannelVec new_channels;
        for(const auto &channel : channels) {
            bool add_decay = std::binary_search(channel.info.begin(), channel.info.end(), decay.first);
            for(const auto &prods : channel.decays) {
                add_decay |= prods.second.first == decay.first || prods.second.second == decay.first;
            }
            if(add_decay) {
                for(const auto &decay_prods : decay.second) {
                    auto chan = channel;
                    chan.decays[decay.first] = decay_prods;
                    new_channels.push_back(chan);
                }
            } else new_channels.push_back(channel);
        }
        channels = new_channels;
    }

    // Convert channel descriptions to mappings
    std::vector<std::unique_ptr<FSMapper>> mappings;
    for(auto ch_descr : channels) {
        auto mapping = ConstructChannel(ch_descr, cuts);
        // if(mapping) mappings.push_back(std::move(mapping));
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

std::unique_ptr<FSMapper> apes::ConstructChannel(ChannelDescription channel, const Cuts& cuts) {
    // Sort channel based on masses then on indices then on pids
    std::sort(channel.info.begin(), channel.info.end());
    spdlog::info("{}", ToString(channel));

    // TODO: Read from an input card
    double sqrts = 13000;

    // Create lambda to generate masses for decays
    size_t iran = 0, iran2 = 0;
    auto gen_decays = [&](std::unordered_map<unsigned int, double> &masses2, const std::vector<double> &rans) {
        spdlog::info("Masses:");
        for(const auto &decay : channel.decays) {
            const auto part1 = decay.second.first;
            const auto part2 = decay.second.second;
            double smin1 = pow(sqrt(masses2[part1.idx])
                              + sqrt(masses2[part2.idx]), 2);
            double deltaR = cuts.deltaR.at({part1.pid, part2.pid});
            double smin2 = cuts.ptmin.at(part1.idx)*cuts.ptmin.at(part2.idx)*2/M_PI/M_PI*pow(deltaR, 2);
            double smin = std::max(smin1, smin2);
            double smax = SMax(sqrts, cuts, decay.first.idx);
            spdlog::info("smin1 = {}, smin2 = {}", smin1, smin2);
            if(std::abs(decay.first.mass) < 1e-16) {
                masses2[decay.first.idx] = MasslessPropMomenta(smin, smax, rans[iran++]); 
            } else {
                masses2[decay.first.idx] = MassivePropMomenta(decay.first.mass, decay.first.width,
                                                              smin, smax, rans[iran++]); 
            }
            spdlog::info("  {}: {}", decay.first.idx, sqrt(masses2[decay.first.idx]));
        }
    };
    auto wgt_decays = [&](const std::unordered_map<unsigned int, FourVector> &mom, std::vector<double> &rans) {
        double wgt = 1;
        for(const auto &decay : channel.decays) {
            const auto part1 = decay.second.first;
            const auto part2 = decay.second.second;
            double smin1 = pow(mom.at(part1.idx).Mass()
                              + mom.at(part2.idx).Mass(), 2);
            if(std::isnan(smin1)) smin1 = 0;
            double deltaR = cuts.deltaR.at({part1.pid, part2.pid});
            double smin2 = cuts.ptmin.at(part1.idx)*cuts.ptmin.at(part2.idx)*2/M_PI/M_PI*pow(deltaR, 2);
            double smin = std::max(smin1, smin2);
            double smax = SMax(sqrts, cuts, decay.first.idx);
            spdlog::info("smin1 = {}, smin2 = {}", smin1, smin2);
            if(std::abs(decay.first.mass) < 1e-16) {
                wgt *= MasslessPropWeight(smin, smax, mom.at(decay.first.idx).Mass2(), rans[iran2++]); 
            } else {
                wgt *= MassivePropWeight(decay.first.mass, decay.first.width,
                                         smin, smax, mom.at(decay.first.idx).Mass2(), rans[iran2++]); 
            }
        }
        return wgt;
    };

    // Handle t-channel momentum generation
    const double ptmax = sqrts/2.0;
    // TODO: get cut definitions from an input card
    std::vector<double> ptmin;
    for(size_t i = 0; i < channel.info.size(); ++i) {
        double val = 5;
        if(cuts.ptmin.at(channel.info[i].idx) != 0)
            val = cuts.ptmin.at(channel.info[i].idx);
        ptmin.push_back(val);
    }
    auto gen_tchan = [&](std::unordered_map<unsigned int, FourVector> &mom, const std::vector<double> &rans,
                         const std::unordered_map<unsigned int, double> &masses2) -> void {
        FourVector psum{};
        // Handle the first n-1 outgoing t-channel particles
        for(size_t i = 0; i < channel.info.size() - 1; ++i) {
            const double ran = rans[iran++];
            double pt = 2*ptmin[i]*ptmax*ran/(2*ptmin[i]+ptmax*(1-ran));
            double etamax = sqrts/2/pt; 
            etamax = log(etamax+sqrt(etamax*etamax - 1));
            etamax = std::min(etamax, cuts.etamax.at(channel.info[i].idx));
            double y = etamax*(2*rans[iran++]-1);
            double sinhy = sinh(y);
            double coshy = sqrt(1+sinhy*sinhy);
            double phi = 2*M_PI*rans[iran++];
            double mt = sqrt(pt*pt + masses2.at(channel.info[i].idx));
            mom[channel.info[i].idx] = {mt*coshy, pt*cos(phi), pt*sin(phi), mt*sinhy};
            psum += mom[channel.info[i].idx];
        }

        // Handle initial states and last t-channel momentum
        double mjets2 = psum.Mass2();
        double ybar = 0.5*log((psum.E() + psum.Pz())/(psum.E() - psum.Pz()));
        double ptsum2 = psum.Pt2();
        double m2 = masses2.at(channel.info.back().idx);
        double plstarsq = (pow(sqrts*sqrts-m2-mjets2, 2)
                           -4*(mjets2*m2+ptsum2*sqrts*sqrts))/(4*sqrts*sqrts);
        double plstar = sqrt(plstarsq);
        double Estar = sqrt(plstarsq+ptsum2+mjets2);
        double y5starmax = 0.5*log((Estar+plstar)/(Estar-plstar));

        double etamax = ybar+y5starmax;
        double etamin = ybar-y5starmax;
        double dely = etamax-etamin;
        double ycm = etamin+rans[iran++]*dely;
        double sinhy = sinh(ycm);
        double coshy = sqrt(1+sinhy*sinhy);
        
        double sumpst = ptsum2+pow(psum.Pz()*coshy-psum.E()*sinhy, 2);
        double q0st = sqrt(m2+sumpst);
        double rshat = q0st + sqrt(mjets2+sumpst);
        mom[3] = {rshat*coshy, 0, 0, rshat*sinhy};

        std::array<double, 2> xx;
        xx[1] = (mom[3].E()+mom[3].Pz())/sqrts;
        xx[2] = (mom[3].E()-mom[3].Pz())/sqrts;

        mom[channel.info.back().idx] = mom[3] - psum;
        mom[1] = {-xx[1]*sqrts/2, 0, 0, -xx[1]*sqrts/2};
        mom[2] = {-xx[2]*sqrts/2, 0, 0, xx[2]*sqrts/2};
    };
    auto wgt_tchan = [&](const std::unordered_map<unsigned int, FourVector> &mom, std::vector<double> &rans) -> double {
        double wgt = 2*M_PI;
        FourVector psum{};
        for(size_t i = 0; i < channel.info.size() - 1; ++i) {
            wgt /= 16/pow(M_PI, 3);
            double pt = mom.at(channel.info[i].idx).Pt();
            wgt *= pt*ptmax/2/ptmin[i]/(2*ptmin[i]+ptmax)*pow(2*ptmin[i]+pt, 2);
            rans[iran2++] = pt*(ptmax+2*ptmin[i])/(ptmax*(pt+2*ptmin[i]));
            double etamax = sqrts/2/pt; 
            etamax = log(etamax+sqrt(etamax*etamax - 1));
            etamax = std::min(etamax, cuts.etamax.at(channel.info[i].idx));
            wgt *= 2*etamax*2*M_PI;
            double y = mom.at(channel.info[i].idx).Rapidity();
            double phi = mom.at(channel.info[i].idx).Phi();
            rans[iran2++] = (y/etamax + 1)/2;
            rans[iran2++] = phi/2/M_PI;
            psum += mom.at(channel.info[i].idx);
        }
        // Handle initial states and last t-channel momentum
        double mjets2 = psum.Mass2();
        double ybar = 0.5*log((psum.E() + psum.Pz())/(psum.E() - psum.Pz()));
        double ptsum2 = psum.Pt2();
        double m2 = mom.at(channel.info.back().idx).Mass2();
        double plstarsq = (pow(sqrts*sqrts-m2-mjets2, 2)
                           -4*(mjets2*m2+ptsum2*sqrts*sqrts))/(4*sqrts*sqrts);
        double plstar = sqrt(plstarsq);
        double Estar = sqrt(plstarsq+ptsum2+mjets2);
        double y5starmax = 0.5*log((Estar+plstar)/(Estar-plstar));

        double etamax = ybar+y5starmax;
        double etamin = ybar-y5starmax;
        double dely = etamax-etamin;
        wgt *= dely;
        double ycm = (mom.at(1) + mom.at(2)).Rapidity();
        rans[iran2++] = (ycm-etamin)/dely;

        return wgt;
    };

    auto decay = [&](ParticleInfo idxab, ParticleInfo idxa, ParticleInfo idxb,
                     std::unordered_map<unsigned int, FourVector> &mom, const std::vector<double> &rans,
                     const std::unordered_map<unsigned int, double> &masses2) -> void {
        auto decay_impl = [&](ParticleInfo ab, ParticleInfo a, ParticleInfo b, auto &impl) -> void {
            double ran1 = rans[iran++]; 
            double ran2 = rans[iran++];
            mom[a.idx] = FourVector();
            mom[b.idx] = FourVector();
            SChannelMomenta(mom[ab.idx], masses2.at(a.idx), masses2.at(b.idx), mom[a.idx], mom[b.idx],
                            ran1, ran2,-1,1);
            if(!IsPower2(a.idx)) {
                auto decays = channel.decays.at(a);
                impl(a, decays.first, decays.second, impl);
            }
            if(!IsPower2(b.idx)) {
                auto decays = channel.decays.at(b);
                impl(b, decays.first, decays.second, impl);
            }
        };
        decay_impl(idxab, idxa, idxb, decay_impl);
    };
    auto decay_wgt = [&](ParticleInfo idxa, ParticleInfo idxb,
                         const std::unordered_map<unsigned int, FourVector> &mom, std::vector<double> &rans) -> double {
        auto decay_wgt_impl = [&](ParticleInfo a, ParticleInfo b, auto &impl) -> double {  
            double ran1, ran2;
            double wgt = SChannelWeight(mom.at(a.idx), mom.at(b.idx), ran1, ran2);
            rans[iran2++] = ran1;
            rans[iran2++] = ran2;
            if(!IsPower2(a.idx)) {
                auto decays = channel.decays.at(a);
                wgt *= impl(decays.first, decays.second, impl);
            }
            if(!IsPower2(b.idx)) {
                auto decays = channel.decays.at(b);
                wgt *= impl(decays.first, decays.second, impl);
            }
            return wgt;
        };
        return decay_wgt_impl(idxa, idxb, decay_wgt_impl);
    };

    // Handle s-channel decays momentum generation
    auto gen_schan = [&](std::unordered_map<unsigned int, FourVector> &mom, const std::vector<double> &rans,
                         const std::unordered_map<unsigned int, double> &masses2) -> void {
        for(size_t i = 0; i < channel.info.size(); ++i) {
            if(!IsPower2(channel.info[i].idx)) {
                auto decays = channel.decays.at(channel.info[i]);
                decay(channel.info[i], decays.first, decays.second, mom, rans, masses2);
            }
        }
    };
    auto wgt_schan = [&](const std::unordered_map<unsigned int, FourVector> &mom, std::vector<double> &rans) -> double {
        double wgt = 1;
        for(size_t i = 0; i < channel.info.size(); ++i) {
            if(!IsPower2(channel.info[i].idx)) {
                auto decays = channel.decays.at(channel.info[i]);
                wgt *= decay_wgt(decays.first, decays.second, mom, rans);
            }
        }
        return wgt;
    };

    // Combine all lambdas and put into a mapper
    auto gen_pts = [&](std::vector<FourVector> &mom, const std::vector<double> &rans) -> void {
        iran = 0;
        std::unordered_map<unsigned int, double> masses2;
        std::unordered_map<unsigned int, FourVector> tmp_mom;
        for(size_t i = 0; i < cuts.sexternal.size(); ++i) {
            masses2[1 << (i + 2)] = cuts.sexternal[i];
        }
        gen_decays(masses2, rans); 
        gen_tchan(tmp_mom, rans, masses2);
        gen_schan(tmp_mom, rans, masses2);
        for(size_t i = 0; i < mom.size(); ++i) {
            mom[i] = tmp_mom[1 << i];
        }
        for(const auto &p : tmp_mom) {
            spdlog::info("mom[{}] = {}", p.first, p.second);
        }
    };
    auto gen_wgt = [&](const std::vector<FourVector> &mom, std::vector<double> &rans) -> double {
        iran2 = 0;
        double wgt = 1.0;
        std::unordered_map<unsigned int, FourVector> tmp_mom;
        for(size_t i = 0; i < mom.size(); ++i) {
            tmp_mom[1 << i] = mom[i];
        }
        for(const auto &part : channel.decays) {
            tmp_mom[part.first.idx] = tmp_mom[part.second.first.idx] + tmp_mom[part.second.second.idx];
        }
        wgt *= wgt_decays(tmp_mom, rans);
        wgt *= wgt_tchan(tmp_mom, rans);
        wgt *= wgt_schan(tmp_mom, rans);
        return 1.0/wgt;
    };

    // test
    std::vector<FourVector> mom(cuts.sexternal.size() + 2);
    std::vector<double> rans{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1234};
    std::vector<double> rans2(3*(mom.size()-2)-2);
    gen_pts(mom, rans);
    gen_wgt(mom, rans2);

    for(size_t i = 0; i < rans2.size(); ++i) 
        spdlog::info("rans[{}] = {}", i, rans2[i]);

    return std::make_unique<FSMapper>();
}

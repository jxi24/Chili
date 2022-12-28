#include "Channel/Channel.hh"
#include "Tools/ChannelElements.hh"

using apes::FSMapper;
using std::isnan;

void FSMapper::GeneratePoint(std::vector<FourVector> &mom, const std::vector<double> &rans) {
    mom.resize(m_ntot);
    iran = 0;
    SparseMass masses2;
    SparseMom tmp_mom;
    for(size_t i = 0; i < m_nout; ++i) {
        masses2[1 << (i + 2)] = m_cuts.sexternal[i];
    }
    GenDecays(masses2, rans);
    GenTChan(tmp_mom, rans, masses2);
    GenSChan(tmp_mom, rans, masses2);
    for(size_t i = 0; i < mom.size(); ++i) {
        mom[i] = tmp_mom[1 << i];
    }
    spdlog::trace("Channel = {}", ToString(m_channel));
    Mapper<FourVector>::Print(__PRETTY_FUNCTION__, mom, rans);
}

double FSMapper::GenerateWeight(const std::vector<FourVector> &mom, std::vector<double> &rans) {
    rans.resize(NDims());
    iran = 0;
    double wgt = 1.0;
    SparseMom tmp_mom;
    for(size_t i = 0; i < mom.size(); ++i) {
        tmp_mom[1 << i] = mom[i];
    }
    for(const auto &part : m_channel.decays) {
        tmp_mom[part.first.idx] = tmp_mom[part.second.first.idx] + tmp_mom[part.second.second.idx];
    }
    tmp_mom[3] = tmp_mom[1] + tmp_mom[2];
    double wgtdecay = WgtDecays(tmp_mom, rans);
    double wgttchan = WgtTChan(tmp_mom, rans);
    double wgtschan = WgtSChan(tmp_mom, rans);
    wgt = wgtdecay*wgttchan*wgtschan;
    spdlog::trace("Channel = {}", ToString(m_channel));
    Mapper<FourVector>::Print(__PRETTY_FUNCTION__, mom, rans);
    spdlog::trace("  Weight = {}", wgt);
    return wgt;
}

void FSMapper::GenDecays(SparseMass &masses2, const std::vector<double> &rans) {
    for(const auto &decay : m_channel.decays) {
        const auto part1 = decay.second.first;
        const auto part2 = decay.second.second;
        double smin1 = pow(sqrt(masses2[part1.idx])
                          + sqrt(masses2[part2.idx]), 2);
        double deltaR = m_cuts.deltaR.find({part1.pid, part2.pid}) == m_cuts.deltaR.end() 
            ? 0 : m_cuts.deltaR.at({part1.pid, part2.pid});
        double smin2 = m_cuts.ptmin.at(part1.idx)*m_cuts.ptmin.at(part2.idx)*2/M_PI/M_PI*pow(deltaR, 2);
        double smin = std::max(std::max(smin1, smin2), m_cuts.smin[part1.idx|part2.idx]);
        double smax = SMax(m_sqrts, m_cuts, decay.first.idx);
        if(std::abs(decay.first.mass) < 1e-16) {
            masses2[decay.first.idx] = MasslessPropMomenta(smin, smax, rans[iran++]); 
        } else {
            masses2[decay.first.idx] = MassivePropMomenta(decay.first.mass, decay.first.width,
                                                          smin, smax, rans[iran++]); 
        }
    }
}

double FSMapper::WgtDecays(const SparseMom &mom, std::vector<double> &rans) {
    double wgt = 1;
    for(const auto &decay : m_channel.decays) {
        const auto part1 = decay.second.first;
        const auto part2 = decay.second.second;
        double smin1 = pow(mom.at(part1.idx).Mass()
                          + mom.at(part2.idx).Mass(), 2);
        if(std::isnan(smin1)) smin1 = 0;
        double deltaR = m_cuts.deltaR.at({part1.pid, part2.pid});
        double smin2 = m_cuts.ptmin.at(part1.idx)*m_cuts.ptmin.at(part2.idx)*2/M_PI/M_PI*pow(deltaR, 2);
        double smin = std::max(std::max(smin1, smin2), m_cuts.smin[part1.idx|part2.idx]);
        double smax = SMax(m_sqrts, m_cuts, decay.first.idx);
        if(std::abs(decay.first.mass) < 1e-16) {
            wgt *= MasslessPropWeight(smin, smax, mom.at(decay.first.idx).Mass2(), rans[iran++]); 
        } else {
            wgt *= MassivePropWeight(decay.first.mass, decay.first.width,
                                     smin, smax, mom.at(decay.first.idx).Mass2(), rans[iran++]); 
        }
        wgt /= (2*M_PI);
    }
    return wgt;
}

void FSMapper::GenTChan(SparseMom &mom, const std::vector<double> &rans, const SparseMass &masses2) {
    FourVector psum{};
    // Handle the first n-1 outgoing t-channel particles
    for(size_t i = 0; i < m_channel.info.size() - 1; ++i) {
        const double ran = rans[iran++];
        double pt{};
        if(!IsPower2(m_channel.info[i].idx) || m_ptmin[i]==0)
            pt = 2*m_ptmin[i]*m_ptmax*ran/(2*m_ptmin[i]+m_ptmax*(1-ran));
	else {
            double hmin = 1/m_ptmax;
            double hmax = 1/m_ptmin[i];
            pt = 1/(hmin+ran*(hmax-hmin));
        }
        double etamax = m_sqrts/2/pt; 
        etamax = log(etamax+sqrt(etamax*etamax - 1));
        etamax = std::min(etamax, m_cuts.etamax.at(m_channel.info[i].idx));
        double y = etamax*(2*rans[iran++]-1);
        double sinhy = sinh(y);
        double coshy = sqrt(1+sinhy*sinhy);
        double phi = 2*M_PI*rans[iran++];
        double mt = sqrt(pt*pt + masses2.at(m_channel.info[i].idx));
        mom[m_channel.info[i].idx] = {mt*coshy, pt*cos(phi), pt*sin(phi), mt*sinhy};
        psum += mom[m_channel.info[i].idx];
    }

    // Handle initial states and last t-channel momentum
    double mjets2 = psum.Mass2();
    double yjets = psum.Rapidity();
    double ptj2 = psum.Pt2();
    double m2 = masses2.at(m_channel.info.back().idx);
    double qt = sqrt(m2+ptj2);
    double mt = sqrt(mjets2+ptj2);
    double ymin = -log(m_sqrts/qt*(1-mt/m_sqrts*exp(-yjets)));
    double ymax = log(m_sqrts/qt*(1-mt/m_sqrts*exp(yjets)));
    double dely = ymax-ymin;
    double yv = ymin+rans[iran++]*dely;
    double sinhy = sinh(yv);
    double coshy = sqrt(1+sinhy*sinhy);
    mom[m_channel.info.back().idx] = {qt*coshy, -psum[1], -psum[2], qt*sinhy};
    double pp = (mom[m_channel.info.back().idx]+psum).PPlus();
    double pm = (mom[m_channel.info.back().idx]+psum).PMinus();
    mom[1] = {-pp/2, 0, 0, -pp/2};
    mom[2] = {-pm/2, 0, 0, pm/2};
}

double FSMapper::WgtTChan(const SparseMom &mom, std::vector<double> &rans) {
    double wgt = 2*M_PI;
    FourVector psum{};
    for(size_t i = 0; i < m_channel.info.size() - 1; ++i) {
        wgt *= 1.0/(16*pow(M_PI, 3));
        double pt = mom.at(m_channel.info[i].idx).Pt();
        if(!IsPower2(m_channel.info[i].idx) || m_ptmin[i]==0) {
            wgt *= pt*m_ptmax/2/m_ptmin[i]/(2*m_ptmin[i]+m_ptmax)*pow(2*m_ptmin[i]+pt, 2);
            rans[iran++] = pt*(m_ptmax+2*m_ptmin[i])/(m_ptmax*(pt+2*m_ptmin[i]));
	} else {
            double hmin = 1/m_ptmax;
            double hmax = 1/m_ptmin[i];
            rans[iran++] = (1/pt-hmin)/(hmax-hmin);
            wgt *= (hmax-hmin)*pt*pt*pt;
        }
        double etamax = m_sqrts/2/pt; 
        etamax = log(etamax+sqrt(etamax*etamax - 1));
        etamax = std::isnan(etamax) ? 99 : etamax;
        etamax = std::min(etamax, m_cuts.etamax.at(m_channel.info[i].idx));
        wgt *= 2*etamax*2*M_PI;
        double y = mom.at(m_channel.info[i].idx).Rapidity();
        double phi = mom.at(m_channel.info[i].idx).Phi();
        rans[iran++] = (y/etamax + 1)/2;
        rans[iran++] = phi/2/M_PI;
        psum += mom.at(m_channel.info[i].idx);
    }
    // Handle initial states and last t-channel momentum
    double mjets2 = psum.Mass2();
    double yjets = psum.Rapidity();
    double ptj2 = psum.Pt2();
    double m2 = mom.at(m_channel.info.back().idx).Mass2();
    double qt = sqrt(m2+ptj2);
    double mt = sqrt(mjets2+ptj2);
    double yv = mom.at(m_channel.info.back().idx).Rapidity();
    double ymin = -log(m_sqrts/qt*(1-mt/m_sqrts*exp(-yjets)));
    double ymax = log(m_sqrts/qt*(1-mt/m_sqrts*exp(yjets)));
    double dely = ymax-ymin;
    if (std::isnan(ymax) || std::isnan(ymin)) return 0;
    wgt *= dely/m_sqrts/m_sqrts;
    rans[iran++] = (yv-ymin)/dely;

    return wgt;
}

void FSMapper::DecayParts(ParticleInfo ab, ParticleInfo a, ParticleInfo b,
                          SparseMom &mom, const std::vector<double> &rans, const SparseMass &masses2) {
    double ran1 = rans[iran++];
    double ran2 = rans[iran++];
    mom[a.idx] = FourVector();
    mom[b.idx] = FourVector();
    SChannelMomenta(mom[ab.idx], masses2.at(a.idx), masses2.at(b.idx), mom[a.idx], mom[b.idx],
                    ran1, ran2, -1, 1);
    if(!IsPower2(a.idx)) {
        auto decays = m_channel.decays.at(a);
        DecayParts(a, decays.first, decays.second, mom, rans, masses2);
    }
    if(!IsPower2(b.idx)) {
        auto decays = m_channel.decays.at(b);
        DecayParts(b, decays.first, decays.second, mom, rans, masses2);
    }
}

double FSMapper::WgtDecayParts(ParticleInfo a, ParticleInfo b, const SparseMom &mom, std::vector<double> &rans) {
    double ran1, ran2;
    double wgt = SChannelWeight(mom.at(a.idx), mom.at(b.idx), ran1, ran2);
    rans[iran++] = ran1;
    rans[iran++] = ran2;
    if(!IsPower2(a.idx)) {
        auto decays = m_channel.decays.at(a);
        wgt *= WgtDecayParts(decays.first, decays.second, mom, rans);
    }
    if(!IsPower2(b.idx)) {
        auto decays = m_channel.decays.at(b);
        wgt *= WgtDecayParts(decays.first, decays.second, mom, rans);
    }
    return wgt;
}

void FSMapper::GenSChan(SparseMom &mom, const std::vector<double> &rans, const SparseMass &masses2) {
    for(size_t i = 0; i < m_channel.info.size(); ++i) {
        if(!IsPower2(m_channel.info[i].idx)) {
            auto decays = m_channel.decays.at(m_channel.info[i]);
            DecayParts(m_channel.info[i], decays.first, decays.second, mom, rans, masses2);
        }
    }
}

double FSMapper::WgtSChan(const SparseMom &mom, std::vector<double> &rans) {
    double wgt = 1; 
    for(size_t i = 0; i < m_channel.info.size(); ++i) {
        if(!IsPower2(m_channel.info[i].idx)) {
            auto decays = m_channel.decays.at(m_channel.info[i]);
            wgt *= WgtDecayParts(decays.first, decays.second, mom, rans);
        }
    }
    return wgt;
}

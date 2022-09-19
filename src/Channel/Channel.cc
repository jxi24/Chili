#include "Channel/Channel.hh"
#include "Tools/ChannelElements.hh"

using apes::FSMapper;

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
    wgt *= WgtDecays(tmp_mom, rans);
    wgt *= WgtTChan(tmp_mom, rans);
    wgt *= WgtSChan(tmp_mom, rans);
    Mapper<FourVector>::Print(__PRETTY_FUNCTION__, mom, rans);
    spdlog::trace("  Weight = {}", wgt);
    return 1.0/wgt;
}

void FSMapper::GenDecays(SparseMass &masses2, const std::vector<double> &rans) {
    for(const auto &decay : m_channel.decays) {
        const auto part1 = decay.second.first;
        const auto part2 = decay.second.second;
        double smin1 = pow(sqrt(masses2[part1.idx])
                          + sqrt(masses2[part2.idx]), 2);
        double deltaR = m_cuts.deltaR.at({part1.pid, part2.pid});
        double smin2 = m_cuts.ptmin.at(part1.idx)*m_cuts.ptmin.at(part2.idx)*2/M_PI/M_PI*pow(deltaR, 2);
        double smin = std::max(smin1, smin2);
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
        double smin = std::max(smin1, smin2);
        double smax = SMax(m_sqrts, m_cuts, decay.first.idx);
        if(std::abs(decay.first.mass) < 1e-16) {
            wgt *= MasslessPropWeight(smin, smax, mom.at(decay.first.idx).Mass2(), rans[iran++]); 
        } else {
            wgt *= MassivePropWeight(decay.first.mass, decay.first.width,
                                     smin, smax, mom.at(decay.first.idx).Mass2(), rans[iran++]); 
        }
    }
    return wgt;
}

void FSMapper::GenTChan(SparseMom &mom, const std::vector<double> &rans, const SparseMass &masses2) {
    FourVector psum{};
    // Handle the first n-1 outgoing t-channel particles
    for(size_t i = 0; i < m_channel.info.size() - 1; ++i) {
        const double ran = rans[iran++];
        double pt = 2*m_ptmin[i]*m_ptmax*ran/(2*m_ptmin[i]+m_ptmax*(1-ran));
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
    double ybar = 0.5*log((psum.E() + psum.Pz())/(psum.E() - psum.Pz()));
    double ptsum2 = psum.Pt2();
    double m2 = masses2.at(m_channel.info.back().idx);
    double plstarsq = (pow(m_sqrts*m_sqrts-m2-mjets2, 2)
                       -4*(mjets2*m2+ptsum2*m_sqrts*m_sqrts))/(4*m_sqrts*m_sqrts);
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
    xx[1] = (mom[3].E()+mom[3].Pz())/m_sqrts;
    xx[2] = (mom[3].E()-mom[3].Pz())/m_sqrts;

    mom[m_channel.info.back().idx] = mom[3] - psum;
    mom[1] = {-xx[1]*m_sqrts/2, 0, 0, -xx[1]*m_sqrts/2};
    mom[2] = {-xx[2]*m_sqrts/2, 0, 0, xx[2]*m_sqrts/2};
}

double FSMapper::WgtTChan(const SparseMom &mom, std::vector<double> &rans) {
    double wgt = 2*M_PI;
    FourVector psum{};
    for(size_t i = 0; i < m_channel.info.size() - 1; ++i) {
        wgt /= 16/pow(M_PI, 3);
        double pt = mom.at(m_channel.info[i].idx).Pt();
        wgt *= pt*m_ptmax/2/m_ptmin[i]/(2*m_ptmin[i]+m_ptmax)*pow(2*m_ptmin[i]+pt, 2);
        rans[iran++] = pt*(m_ptmax+2*m_ptmin[i])/(m_ptmax*(pt+2*m_ptmin[i]));
        double etamax = m_sqrts/2/pt; 
        etamax = log(etamax+sqrt(etamax*etamax - 1));
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
    double ybar = 0.5*log((psum.E() + psum.Pz())/(psum.E() - psum.Pz()));
    double ptsum2 = psum.Pt2();
    double m2 = mom.at(m_channel.info.back().idx).Mass2();
    double plstarsq = (pow(m_sqrts*m_sqrts-m2-mjets2, 2)
                       -4*(mjets2*m2+ptsum2*m_sqrts*m_sqrts))/(4*m_sqrts*m_sqrts);
    double plstar = sqrt(plstarsq);
    double Estar = sqrt(plstarsq+ptsum2+mjets2);
    double y5starmax = 0.5*log((Estar+plstar)/(Estar-plstar));

    double etamax = ybar+y5starmax;
    double etamin = ybar-y5starmax;
    double dely = etamax-etamin;
    wgt *= dely;
    double ycm = (mom.at(1) + mom.at(2)).Rapidity();
    rans[iran++] = (ycm-etamin)/dely;

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

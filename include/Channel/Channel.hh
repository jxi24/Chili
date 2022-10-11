#pragma once

#include "Channel/Mapper.hh"
#include "Tools/FourVector.hh"
#include "Channel/ChannelNode.hh"

namespace apes {

using SparseMom = std::unordered_map<unsigned int, FourVector>;
using SparseMass = std::unordered_map<unsigned int, double>;

struct ChannelNode;

class FSMapper : public Mapper<FourVector> {
    public:
        FSMapper(double sqrts, size_t ntot, ChannelDescription channel, Cuts cuts) 
            : m_sqrts{sqrts}, m_ptmax{sqrts/2.0}, m_ntot{ntot}, m_nout{ntot-2}, m_channel{channel}, m_cuts{cuts} {
                for(size_t i = 0; i < m_channel.info.size(); ++i) {
                    double val = 5;
                    if(m_cuts.ptmin.at(m_channel.info[i].idx) != 0) {
                        val = m_cuts.ptmin.at(m_channel.info[i].idx);
                        spdlog::info("here, val = {}", val);
                    }
                    m_ptmin.push_back(val);
                }
                for(size_t i = 0; i < m_ptmin.size(); ++i) {
                    spdlog::info("{}: ptmin = {}, idx = {}", i, m_ptmin[i], m_channel.info[i].pid);
                }
        }
        void GeneratePoint(std::vector<FourVector>&, const std::vector<double>&) override;
        double GenerateWeight(const std::vector<FourVector>&, std::vector<double>&) override;
        size_t NDims() const override { return 3*m_nout - 4 + 2; }
        void WriteChannel() const;

    private:
        void GenDecays(SparseMass&, const std::vector<double>&);
        double WgtDecays(const SparseMom&, std::vector<double>&);
        void GenTChan(SparseMom&, const std::vector<double>&,
                      const SparseMass&);
        double WgtTChan(const SparseMom&, std::vector<double>&);
        void DecayParts(ParticleInfo, ParticleInfo, ParticleInfo,
                        SparseMom&, const std::vector<double>&, const SparseMass&);
        double WgtDecayParts(ParticleInfo, ParticleInfo, const SparseMom&, std::vector<double>&);
        void GenSChan(SparseMom&, const std::vector<double>&, const SparseMass&);
        double WgtSChan(const SparseMom&, std::vector<double>&);

        double m_sqrts, m_ptmax;
        size_t m_ntot, m_nout;
        ChannelDescription m_channel;
        Cuts m_cuts;
        size_t iran{};
        std::vector<double> m_ptmin;
};

}

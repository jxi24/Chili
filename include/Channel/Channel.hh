#pragma once

#include "Channel/Mapper.hh"
#include "Tools/Factory.hh"
#include "Tools/FourVector.hh"
#include "Channel/ChannelNode.hh"

namespace apes {

using SparseMom = std::unordered_map<unsigned int, FourVector>;
using SparseMass = std::unordered_map<unsigned int, double>;

struct ChannelNode;

class FSMapper : public Mapper<FourVector>, Registrable<Mapper<FourVector>, FSMapper> {
    public:
        FSMapper() = default;
        FSMapper(double sqrts, size_t ntot, ChannelDescription channel, Cuts cuts) 
            : m_sqrts{sqrts}, m_ptmax{sqrts/2.0}, m_ntot{ntot}, m_nout{ntot-2}, m_channel{channel}, m_cuts{cuts} {
                for(size_t i = 0; i < m_channel.info.size(); ++i) {
                    double val = 1;
                    if(m_cuts.ptmin.at(m_channel.info[i].idx) != 0) {
                        val = m_cuts.ptmin.at(m_channel.info[i].idx);
                    }
                    m_ptmin.push_back(val);
                }
                for(size_t i = 0; i < m_ptmin.size(); ++i) {
                    spdlog::trace("{}: ptmin = {}, idx = {}", i, m_ptmin[i], m_channel.info[i].pid);
                }
        }
        void GeneratePoint(std::vector<FourVector>&, const std::vector<double>&) override;
        double GenerateWeight(const std::vector<FourVector>&, std::vector<double>&) override;
        size_t NDims() const override { return 3*m_nout - 4 + 2; }
        void WriteChannel() const;
        static std::string Name() { return "FSMapper"; }
        static std::unique_ptr<Mapper<FourVector>> _Deserialize(std::istream &in) {
            auto mapper = std::make_unique<FSMapper>();
            mapper -> m_channel.Deserialize(in);
            in.read(reinterpret_cast<char*>(&mapper -> m_sqrts), sizeof(mapper -> m_sqrts));
            in.read(reinterpret_cast<char*>(&mapper -> m_ptmax), sizeof(mapper -> m_ptmax));
            in.read(reinterpret_cast<char*>(&mapper -> m_ntot), sizeof(mapper -> m_ntot));
            in.read(reinterpret_cast<char*>(&mapper -> m_nout), sizeof(mapper -> m_nout));
            size_t size;
            in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
            mapper -> m_ptmin.resize(size);
            for(auto &pt : mapper -> m_ptmin) {
                in.read(reinterpret_cast<char*>(&pt), sizeof(pt));
            }
            // TODO: Deserialize the cuts??
            return mapper;
        }

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
        SparseMom tmp_mom;

        // Serialization
        std::string GetName() const override { return FSMapper::Name(); }
        bool _Serialize(std::ostream &out) const override;
};

}

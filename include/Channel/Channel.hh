#pragma once

#include "Channel/Mapper.hh"
#include "Tools/vec4.h"
#include "Tools/recursion.h"

namespace apes {

struct ChannelNode;

class Channel : public Mapper<Mom<double>> {
    public:
        Channel(std::vector<int> flavs, const MatrixElement &amp) : m_n{flavs.size()}, m_nout{m_n-2} {
            for(const auto &flav : flavs) {
                m_s.push_back(amp.masses[static_cast<size_t>(std::abs(flav))]);
            }
            m_p.resize(1 << m_n);
        }
        bool InitializeChannel(std::shared_ptr<ChannelNode>);
        void GeneratePoint(std::vector<Mom<double>>&, const std::vector<double>&) override;
        double GenerateWeight(const std::vector<Mom<double>>&, std::vector<double>&) override;
        size_t NDims() const override { return 3*m_nout - 4; }
        void WriteChannel() const;

        // Print out methods to debug
        std::string ToString() {
            auto lid = static_cast<unsigned int>((1 << m_n) - 2);
            auto result = PrintPoint(m_nodes.get(), lid, 0);
            return result.substr(0, result.size()-1);
        }
        std::string PrintPoint(ChannelNode *cur, unsigned int lid, unsigned int depth) const;
        std::string PrintSPoint(ChannelNode *node) const;

    private:
        constexpr static unsigned int m_rid = 2, m_lid = 1;
        constexpr static double m_salpha = 0.9, m_scutoff = 1000;
        constexpr static double m_amct=1.0, m_alpha=0.9, m_ctmax=1.0, m_ctmin=-1.0;
        ChannelNode *LocateNode(ChannelNode *node, unsigned int id);
        void BuildPoint(ChannelNode* node, unsigned int lid, const std::vector<double>&, unsigned int depth=0);
        void BuildSPoint(ChannelNode* node, const std::vector<double>&);
        double BuildWeight(ChannelNode* node, unsigned int lid, std::vector<double>&, unsigned int depth=0);
        double BuildSWeight(ChannelNode* node, std::vector<double>&);
        double SCut(unsigned int);
        unsigned int SId(unsigned int);
        double PropMomenta(ChannelNode*, unsigned int, double, double, double);
        double PropWeight(ChannelNode*, unsigned int, double, double, double, double&);
        void TChannelMomenta(ChannelNode*, unsigned int, unsigned int, unsigned int, const std::vector<double>&);
        void SChannelMomenta(ChannelNode*, unsigned int, unsigned int, unsigned int, const std::vector<double>&);
        double TChannelWeight(ChannelNode*, unsigned int, unsigned int, unsigned int, std::vector<double>&);
        double SChannelWeight(ChannelNode*, unsigned int, unsigned int, unsigned int, std::vector<double>&);
        void FillMomenta(ChannelNode*);

        size_t m_n, m_nout;
        std::vector<double> m_s;
        std::string m_name;
        std::shared_ptr<ChannelNode> m_nodes{};
        std::vector<Mom<double>> m_p;
        size_t iran{};
};

}

#ifndef JET_CLUSTER_HH
#define JET_CLUSTER_HH

#include "Tools/FourVector.hh"

namespace chili {

class JetCluster {
    public:
        JetCluster(double r) : m_r{r} {}
        std::vector<FourVector> operator()(const std::vector<FourVector> &parts);

    private:
        double m_r;

        double R2(const FourVector &p, const FourVector &q) const;
        double Q2i(const FourVector &p) const { return p.Pt2(); }
        double Q2ij(const FourVector &p, const FourVector &q) const {
            return std::min(p.Pt2(), q.Pt2())*R2(p, q);
        }
};

}

#endif

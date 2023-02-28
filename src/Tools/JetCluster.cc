#include "Tools/JetCluster.hh"
#include "Tools/FourVector.hh"
#include "spdlog/spdlog.h"

using chili::FourVector;
using chili::JetCluster;

std::vector<FourVector> JetCluster::operator()(const std::vector<FourVector> &parts) {
    std::vector<FourVector> tmp(parts.begin()+2, parts.end()), jets;
    if(parts.size() <= 1) return tmp;

    for(const auto &part : tmp) {
        spdlog::trace("Part = {}", part);
    }

    size_t ii = 0, jj = 0, n = tmp.size();
    std::vector<size_t> imap(n);
    for(size_t i = 0; i < n; ++i) imap[i] = i;
    std::vector<std::vector<double>> kt2ij(n, std::vector<double>(n));
    double dmin = std::numeric_limits<double>::max();
    for(size_t i = 0; i < n; ++i) {
        double di = kt2ij[i][i] = Q2i(tmp[i]);
        if(di < dmin) {
            dmin = di;
            ii = i;
            jj = i;
        }
        for(size_t j = 0; j < i; ++j) {
            spdlog::trace("i = {}, j = {}, n = {}", i, j, n);
            double dij = kt2ij[i][j] = Q2ij(tmp[i], tmp[j]);
            if(dij < dmin) {
                dmin = dij;
                ii = i;
                jj = j;
            }
        }
    }
    while(n != static_cast<size_t>(-1)) {
        spdlog::trace("Q[{} -> {}] = {} <- {} {}",
                      n, n-1, sqrt(dmin), tmp[imap[jj]], tmp[imap[ii]]);
        if(ii != jj) tmp[imap[jj]] += tmp[imap[ii]];
        --n;
        if(n == static_cast<size_t>(-1)) break;
        if(ii == jj) jets.push_back(tmp[imap[jj]]);
        for(size_t i = ii; i < n; ++i) imap[i] = imap[i+1];
        size_t jjx = imap[jj];
        kt2ij[jjx][jjx] = Q2i(tmp[jjx]);
        for(size_t j = 0; j < jj; ++j) kt2ij[jjx][imap[j]]=Q2ij(tmp[jjx],tmp[imap[j]]);
        for(size_t i = jj+1; i < n; ++i) kt2ij[imap[i]][jjx]=Q2ij(tmp[jjx],tmp[imap[i]]);
        ii=jj=0;
        dmin = kt2ij[imap[0]][imap[0]];
        for(size_t i = 0; i < n; ++i) {
            size_t ix = imap[i];
            double di = kt2ij[ix][ix];
            if(di < dmin) {
                dmin = di;
                ii = i;
                jj = i;
            }
            for(size_t j = 0; j < i; ++j) {
                size_t jx = imap[j];
                double dij = kt2ij[ix][jx];
                if(dij < dmin) {
                    dmin = dij;
                    ii = i;
                    jj = j;
                }
            }
        }
    }
    return jets;
}

double JetCluster::R2(const FourVector &p, const FourVector &q) const {
    const double dy = p.Rapidity() - q.Rapidity();
    const double dphi = p.Phi() - q.Phi();
    return (dy*dy + dphi*dphi)/m_r/m_r;
}

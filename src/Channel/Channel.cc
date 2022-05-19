#include "Channel/Channel.hh"
#include "Channel/ChannelUtils.hh"
#include "Channel/ChannelNode.hh"

using apes::FSMapper;
using apes::ChannelNode;

double FSMapper::SCut(unsigned int id) {
    if(id&3) {
        id = (1u << m_n) - 1u - id; 
    }
    double result = 0;
    for(unsigned int i = 0; i < m_n; ++i) {
        if(apes::SetBit(id, i)) {
            result += sqrt(m_s[i]);
        }
    }
    return result*result;
}

unsigned int FSMapper::SId(unsigned int id) {
    return (id&3)==3?(1u<<m_n)-1u-id:id;
}

void FSMapper::FillMomenta(ChannelNode *node) {
    m_p[node -> m_idx] = FourVector{};
    for(unsigned int i = 0; i < m_n; ++i) {
        if(apes::SetBit(node -> m_idx, i)) {
            m_p[node -> m_idx] += m_p[(1 << i)];
        }
    }
    if(!apes::IsPower2(node -> m_left -> m_idx)) FillMomenta(node -> m_left.get());
    if(!apes::IsPower2(node -> m_right -> m_idx)) FillMomenta(node -> m_right.get());
}

std::string FSMapper::PrintPoint(ChannelNode *cur, unsigned int lid, unsigned int depth) const {
    std::string result = "";
    if(depth == m_n-2) return result;
    unsigned int aid = cur -> m_idx;
    unsigned int bid = cur -> m_left -> m_idx;
    unsigned int cid = cur -> m_right -> m_idx;
    if(bid == lid) std::swap(aid, bid);
    else if(cid == lid) std::swap(aid, cid);
    if((cid&(lid|m_rid))==(lid|m_rid) || (aid&m_rid && bid&m_rid)) {
        std::swap(bid, cid);
        std::swap(cur -> m_left, cur -> m_right);
    }
    if(cid == m_rid) {
        if(!apes::IsPower2(bid)) {
            result += PrintSPoint(cur -> m_left.get());
            return result;
        } else {
            return result;
        }
    }
    result += fmt::format("TChannel({} ({}), {} ({}), {} ({})),",
                          aid, cur -> m_pid,
                          bid, cur -> m_left -> m_pid,
                          cid, cur -> m_right -> m_pid);
    if(!apes::IsPower2(cid)) result += PrintPoint(cur -> m_right.get(), cid, depth+1);
    if(!apes::IsPower2(bid)) {
        result += PrintSPoint(cur -> m_left.get());
    }
    return result;
}


std::string FSMapper::PrintSPoint(ChannelNode *node) const {
    unsigned int cid = node -> m_idx;
    unsigned int aid = node -> m_left -> m_idx;
    unsigned int bid = node -> m_right -> m_idx;
    std::string result = fmt::format("SChannel({} ({}), {} ({}), {} ({})),",
                                     cid, node -> m_pid,
                                     aid, node -> m_left -> m_pid,
                                     bid, node -> m_right -> m_pid);
    if(!apes::IsPower2(aid)) result += PrintSPoint(node -> m_left.get());
    if(!apes::IsPower2(bid)) result += PrintSPoint(node -> m_right.get());
    return result;
}

void FSMapper::GeneratePoint(std::vector<FourVector> &p, const std::vector<double> &rans) {
    iran = 0;
    m_p[1] = p[0];
    m_p[2] = p[1];
    auto lid = static_cast<unsigned int>((1 << m_n) - 2);
    m_p[lid] = p[0];
    BuildPoint(m_nodes.get(), lid, rans, 0);
    for(size_t i = 2; i < m_n; ++i) {
        p[i] = m_p[1 << i];
    }
    // Mapper::Print(__PRETTY_FUNCTION__, p, rans);
}

double FSMapper::GenerateWeight(const std::vector<FourVector> &p, std::vector<double> &rans) {
    if(rans.size() != NDims()) rans.resize(NDims());
    iran = 0;
    for(size_t i = 0; i < m_n; ++i) m_p[1 << i] = p[i];
    FillMomenta(m_nodes.get());
    auto lid = static_cast<unsigned int>((1 << m_n) - 2);
    m_p[lid] = p[0];
    double wgt = BuildWeight(m_nodes.get(), lid, rans, 0);
    if(wgt != 0) wgt = 1.0/wgt/pow(2.0*M_PI, 3.0*static_cast<double>(m_nout)-4);
    return 1.0/wgt;
}

void FSMapper::BuildPoint(ChannelNode *cur, unsigned int lid, const std::vector<double> &rans, unsigned int depth) {
    if(depth == m_n-2) {
        return;
    }
    unsigned int aid = cur -> m_idx;
    unsigned int bid = cur -> m_left -> m_idx;
    unsigned int cid = cur -> m_right -> m_idx;
    if(bid == lid) std::swap(aid, bid);
    else if(cid == lid) std::swap(aid, cid);
    if((cid&(lid|m_rid))==(lid|m_rid) || (aid&m_rid && bid&m_rid)) {
        std::swap(bid, cid);
        std::swap(cur -> m_left, cur -> m_right);
    }
    if(cid == m_rid) {
        if(!apes::IsPower2(bid)) {
            BuildSPoint(cur -> m_left.get(), rans);
            return;
        } else {
            return;
        }
    }
    TChannelMomenta(cur, aid, bid, cid, rans); 
    BuildPoint(cur -> m_right.get(), cid, rans, depth+1);
}

void FSMapper::BuildSPoint(ChannelNode *node, const std::vector<double> &rans) {
    unsigned int cid = node -> m_idx;
    unsigned int aid = node -> m_left -> m_idx;
    unsigned int bid = node -> m_right -> m_idx;
    SChannelMomenta(node, aid, bid, cid, rans);
    if(!apes::IsPower2(aid)) BuildSPoint(node -> m_left.get(), rans);
    if(!apes::IsPower2(bid)) BuildSPoint(node -> m_right.get(), rans);
}

double FSMapper::BuildWeight(ChannelNode *cur, unsigned int lid, std::vector<double> &rans, unsigned int depth) {
    double wgt = 1.0;
    if(depth == m_n-2) {
        return wgt;
    }
    unsigned int aid = cur -> m_idx;
    unsigned int bid = cur -> m_left -> m_idx;
    unsigned int cid = cur -> m_right -> m_idx;
    if(bid == lid) std::swap(aid, bid);
    else if(cid == lid) std::swap(aid, cid);
    if((cid&(lid|m_rid))==(lid|m_rid) || (aid&m_rid && bid&m_rid)) {
        std::swap(bid, cid);
        std::swap(cur -> m_left, cur -> m_right);
    }
    if(cid == m_rid) {
        if(!apes::IsPower2(bid)) {
            wgt *= BuildSWeight(cur -> m_left.get(), rans);
            return wgt;
        } else {
            return wgt;
        }
    }
    wgt *= TChannelWeight(cur, aid, bid, cid, rans); 
    wgt *= BuildWeight(cur -> m_right.get(), cid, rans, depth+1);
    return wgt;
}

double FSMapper::BuildSWeight(ChannelNode *node, std::vector<double> &rans) {
    double wgt = 1.0;
    unsigned int cid = node -> m_idx;
    unsigned int aid = node -> m_left -> m_idx;
    unsigned int bid = node -> m_right -> m_idx;
    wgt *= SChannelWeight(node, aid, bid, cid, rans);
    if(!apes::IsPower2(aid)) wgt *= BuildSWeight(node -> m_left.get(), rans);
    if(!apes::IsPower2(bid)) wgt *= BuildSWeight(node -> m_right.get(), rans);
    return wgt;
}

// TODO: handle propagators
// double FSMapper::PropMomenta(ChannelNode *node, unsigned int id, double smin, double smax, double ran) {
double FSMapper::PropMomenta(ChannelNode *, unsigned int , double , double , double ) {
    return 1;
    // auto foundNode = LocateNode(node, id);
    // if(!foundNode) {
    //     spdlog::trace("ThresholdMomenta({}, {}, {}, {}) = {}",
    //                   0.01, smin, smax, ran,
    //                   CE.ThresholdMomenta(m_salpha, 0.01, smin, smax, ran));
    //     return CE.ThresholdMomenta(m_salpha, 0.01, smin, smax, ran);
    // }
    // auto flav = foundNode -> m_pid;
    // if(flav.Mass() == 0) {
    //     spdlog::trace("MasslessPropMomenta({}, {}, {}) = {}",
    //                   smin, smax, ran,
    //                   CE.MasslessPropMomenta(m_salpha, smin, smax, ran));
    //     return CE.MasslessPropMomenta(m_salpha, smin, smax, ran);
    // } else {
    //     spdlog::trace("MassivePropMomenta({}, {}, {}, {}, {}) = {}",
    //                   flav.Mass(), flav.Width(), smin, smax, ran,
    //                   CE.MassivePropMomenta(flav.Mass(), flav.Width(), 1, smin, smax, ran));
    //     if(smax < flav.Mass()*flav.Mass()/m_scutoff) return smin + ran*(smax-smin);
    //     return CE.MassivePropMomenta(flav.Mass(), flav.Width(), 1, smin, smax, ran);
    // }
}

// TODO: handle propagators
// double FSMapper::PropWeight(ChannelNode *node, unsigned int id, double smin, double smax, double s, double &ran) {
double FSMapper::PropWeight(ChannelNode *, unsigned int , double , double , double , double &) {
    return 1;
    // auto foundNode = LocateNode(node, id);
    // if(!foundNode) {
    //     double wgt = CE.ThresholdWeight(m_salpha, 0.01, smin, smax, s, ran);
    //     spdlog::trace("ThresholdWeight = {}", wgt);
    //     return wgt;
    // }
    // auto flav = foundNode -> m_pid;
    // double wgt = 1.0;
    // if(flav.Mass() == 0) {
    //     wgt = CE.MasslessPropWeight(m_salpha, smin, smax, s, ran);
    //     spdlog::trace("MasslessPropWeight = {}", wgt);
    // } else {
    //     if(smax < flav.Mass()*flav.Mass()/m_scutoff) {
    //         wgt = 1.0/(smax-smin);
    //         ran = (s-smin)/(smax-smin);
    //     } else {
    //         wgt = CE.MassivePropWeight(flav.Mass(), flav.Width(), 1, smin, smax, s, ran);
    //     }
    //     spdlog::trace("MassivePropWeight = {}, m = {}, g = {}, smin = {}, smax = {}, s = {}",
    //                   wgt, flav.Mass(), flav.Width(), smin, smax, s);
    // }

    // return wgt;
}

void FSMapper::TChannelMomenta(ChannelNode *node, unsigned int aid, unsigned int bid, unsigned int cid,
                              const std::vector<double> &rans) {
    unsigned int pid = aid - (m_rid+bid);
    const std::string name = "TChannelMomenta(" + std::to_string(aid) + ", "
                                                + std::to_string(bid) + ", "
                                                + std::to_string(cid) + ", "
                                                + std::to_string(pid) + ")";
    double se = SCut(bid), sp = SCut(pid);
    double rtsmax = (m_p[aid]+m_p[m_rid]).Mass();
    if(!apes::IsPower2(bid)) {
        double smin = se, smax = pow(rtsmax - sqrt(sp), 2);
        se = PropMomenta(node, bid, smin, smax, rans[iran++]); 
    }
    if(!apes::IsPower2(pid)) {
        double smin = sp, smax = pow(rtsmax - sqrt(se), 2);
        sp = PropMomenta(node, pid, smin, smax, rans[iran++]);
    }
    // TODO: Look up correct mass
    // double mass = ATOOLS::Flavour((kf_code)(LocateNode(node, pid+m_rid) -> m_pid)).Mass(); 
    // CE.TChannelMomenta(m_p[aid], m_p[m_rid], m_p[bid], m_p[pid], se, sp, mass,
    //                    m_alpha, m_ctmax, m_ctmin, m_amct, 0, rans[iran], rans[iran+1]);
    // iran += 2;
    // m_p[cid] = m_p[aid] - m_p[bid];
    // spdlog::trace("{}", name);
    // spdlog::trace("  m_p[{}] = {}, m = {}", aid, m_p[aid], m_p[aid].Mass());
    // spdlog::trace("  m_p[{}] = {}, m = {}", m_rid, m_p[m_rid], m_p[m_rid].Mass());
    // spdlog::trace("  m_p[{}] = {}, m = {}", bid, m_p[bid], m_p[bid].Mass());
    // spdlog::trace("  m_p[{}] = {}, m = {}", pid, m_p[pid], m_p[pid].Mass());
    // spdlog::trace("  se = {}, sp = {}", se, sp);
    // spdlog::trace("  m[0]^2 = {}, m[1]^2 = {}, m[2]^2 = {}, m[3]^2 = {}", m_s[0], m_s[1], m_s[2], m_s[3]);
    // spdlog::trace("  iran = {}", iran);
}

double FSMapper::TChannelWeight(ChannelNode *node, unsigned int aid, unsigned int bid, unsigned int cid,
                               std::vector<double> &rans) {
    unsigned int pid = aid - (m_rid+bid);
    double wgt = 1.0;
    // aid = (1 << m_n) - 1 - aid;
    m_p[pid] = m_p[aid]+m_p[m_rid]-m_p[bid];
    double se = SCut(bid), sp = SCut(pid);
    double rtsmax = (m_p[aid] + m_p[m_rid]).Mass();
    const std::string name = "TChannelWeight(" + std::to_string(aid) + ", "
                                               + std::to_string(bid) + ", "
                                               + std::to_string(cid) + ", "
                                               + std::to_string(pid) + ")";
    spdlog::trace("{}", name);
    if(!apes::IsPower2(bid)) {
        double smin = se, smax = pow(rtsmax - sqrt(sp), 2);
        wgt *= PropWeight(node, bid, smin, smax, se=m_p[bid].Mass2(), rans[iran++]);
        spdlog::trace("  smin = {}, smax = {}", smin, smax);
    }
    if(!apes::IsPower2(pid)) {
        double smin = sp, smax = pow(rtsmax - sqrt(se), 2);
        wgt *= PropWeight(node, pid, smin, smax, sp=m_p[pid].Mass2(), rans[iran++]);
        spdlog::trace("  smin = {}, smax = {}", smin, smax);
    }
    // TODO: Look up correct mass
    // double mass = ATOOLS::Flavour((kf_code)(LocateNode(node, pid+m_rid) -> m_pid)).Mass(); 
    // wgt *= CE.TChannelWeight(m_p[aid], m_p[m_rid], m_p[bid], m_p[pid], mass,
    //                          m_alpha, m_ctmax, m_ctmin, m_amct, 0, rans[iran], rans[iran+1]);
    // iran+=2;
    // m_p[cid] = m_p[aid] - m_p[bid];
    // spdlog::trace("  m_p[{}] = {}", aid, m_p[aid]);
    // spdlog::trace("  m_p[{}] = {}", m_rid, m_p[m_rid]);
    // spdlog::trace("  m_p[{}] = {}", bid, m_p[bid]);
    // spdlog::trace("  m_p[{}] = {}", pid, m_p[pid]);
    // spdlog::trace("  mass = {}", mass);
    // spdlog::trace("  se = {}, sp = {}", se, sp);
    // spdlog::trace("  iran = {}", iran);
    // spdlog::trace("  wgt = {}", wgt);
    return wgt;
}

void FSMapper::SChannelMomenta(ChannelNode *node, unsigned int aid, unsigned int bid, unsigned int cid,
                              const std::vector<double> &rans) {
    unsigned int lid = SId(aid), rid = SId(bid);
    const std::string name = "SChannelMomenta(" + std::to_string(cid) + ", "
                                                + std::to_string(aid) + ", "
                                                + std::to_string(bid) + ")";
    double rts = m_p[cid].Mass(), sl = SCut(lid), sr = SCut(rid);
    if(!apes::IsPower2(lid)) {
        double smin = sl, smax = pow(rts - sqrt(sr), 2);
        spdlog::trace("smin = {}, smax = {}, rts = {}, sr = {}", smin, smax, rts, sr);
        sl = PropMomenta(node, lid, smin, smax, rans[iran++]);
    }
    if(!apes::IsPower2(rid)) {
        double smin = sr, smax = pow(rts - sqrt(sl), 2);
        spdlog::trace("smin = {}, smax = {}, rts = {}, sl = {}", smin, smax, rts, sl);
        sr = PropMomenta(node, rid, smin, smax, rans[iran++]);
    }
    // TODO: Translate momenta generation
    // CE.Isotropic2Momenta(m_p[cid], sl, sr, m_p[lid], m_p[rid], rans[iran], rans[iran+1], m_ctmin, m_ctmax);
    // iran+=2;
    // m_p[(1<<m_n)-1-aid] = m_p[aid];
    // m_p[(1<<m_n)-1-bid] = m_p[bid];
    // spdlog::trace("{}", name);
    // spdlog::trace("  m_p[{}] ({}) = {}, m = {}", cid, node -> m_pid, m_p[cid], m_p[cid].Mass());
    // spdlog::trace("  m_p[{}] ({}) = {}, m = {}", lid, node -> m_left -> m_pid, m_p[lid], m_p[lid].Mass());
    // spdlog::trace("  m_p[{}] ({}) = {}, m = {}", rid, node -> m_right -> m_pid, m_p[rid], m_p[rid].Mass());
    // spdlog::trace("  sl = {}, sr = {}", sl, sr);
    // spdlog::trace("  iran = {}", iran);
}

double FSMapper::SChannelWeight(ChannelNode *node, unsigned int aid, unsigned int bid, unsigned int cid,
                               std::vector<double> &rans) {
    double wgt = 1.0;
    unsigned int lid = SId(aid), rid = SId(bid);
    double rts = m_p[cid].Mass(), sl = SCut(lid), sr = SCut(rid);
    if(!apes::IsPower2(lid)) {
        double smin = sl, smax = pow(rts - sqrt(sr), 2);
        wgt *= PropWeight(node, lid, smin, smax, sl=m_p[lid].Mass2(), rans[iran++]);
    }
    if(!apes::IsPower2(rid)) {
        double smin = sr, smax = pow(rts - sqrt(sl), 2);
        wgt *= PropWeight(node, rid, smin, smax, sr=m_p[rid].Mass2(), rans[iran++]);
    }
    // TODO: Implement the weight generation
    // wgt *= CE.Isotropic2Weight(m_p[lid], m_p[rid], rans[iran], rans[iran+1], m_ctmin, m_ctmax);
    // iran+=2;
    // const std::string name = "SChannelWeight(" + std::to_string(cid) + ", "
    //                                            + std::to_string(aid) + ", "
    //                                            + std::to_string(bid) + ")";
    // spdlog::trace("{}", name);
    // spdlog::trace("  m_p[{}] ({}) = {}", cid, node -> m_pid, m_p[cid]);
    // spdlog::trace("  m_p[{}] ({}) = {}", lid, node -> m_left -> m_pid, m_p[lid]);
    // spdlog::trace("  m_p[{}] ({}) = {}", rid, node -> m_right -> m_pid, m_p[rid]);
    // spdlog::trace("  sl = {}, sr = {}", sl, sr);
    // spdlog::trace("  iran = {}", iran);
    // spdlog::trace("  wgt = {}", wgt);
    return wgt;
}

ChannelNode *FSMapper::LocateNode(ChannelNode *node, unsigned int id) {
  if (node->m_idx == id)
    return node;
  if (node->m_left) {
    auto result = LocateNode(node->m_left.get(), id);
    if (!result && node->m_right) {
      result = LocateNode(node->m_right.get(), id);
    }
    return result;
  }
  return nullptr;
}

bool FSMapper::InitializeChannel(std::shared_ptr<ChannelNode> node) {
    m_nodes = node;
    return true;
}

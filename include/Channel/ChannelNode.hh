#pragma once

#include "Channel/Channel.hh"
#include "Model/Model.hh"
#include "Process/Process.hh"
#include <map>
#include <memory>
#include <vector>

namespace apes {

struct ChannelNode {
    std::shared_ptr<ChannelNode> m_left = nullptr, m_right = nullptr;
    int m_pid;
    double m_mass;
    unsigned int m_idx;
};

std::vector<std::unique_ptr<Channel>> ConstructChannels(const Process &proc, const Model &model);

}

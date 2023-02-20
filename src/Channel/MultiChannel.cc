#include "Channel/MultiChannel.hh"
#include <ctime>

chili::MultiChannel::MultiChannel(size_t dims, size_t nchannels, MultiChannelParams params_)
                                     : ndims{std::move(dims)}, params{std::move(params_)} {
    for(size_t i = 0; i < nchannels; ++i) {
        channel_weights.push_back(1.0/static_cast<double>(nchannels));
    }
}

void chili::MultiChannel::UpdateResults(StatsData &results) {
#ifdef ENABLE_MPI
    // Combine mpi results
    auto &mpi = MPIHandler::Instance();
    mpi.AllReduce<StatsData, StatsAdd>(results);
#endif
    summary.results.push_back(results);
    summary.sum_results += results;
}

void chili::MultiChannel::Adapt(const std::vector<double> &train) {
    std::vector<double> new_weights(channel_weights.size());

    spdlog::debug("MultiChannel::Adapt:");
    double sum_wgts = 0;
    for(size_t i = 0; i < new_weights.size(); ++i) {
        new_weights[i] = channel_weights[i] * pow(train[i], params.beta);
        sum_wgts += new_weights[i];
    	spdlog::debug("sum_wgts = {}, train[{}] = {}", sum_wgts, i, train[i]);
    }

    double new_sum = 0;
    for(auto &wgt : new_weights) {
        if(wgt == 0) continue;
        wgt /= sum_wgts;
        wgt = std::max(wgt, params.min_alpha);
        new_sum += wgt;
    	spdlog::debug("new_wgts = {}", new_sum);
    }

    size_t idx = 0;
    for(auto &wgt : new_weights) {
        wgt /= new_sum;
        spdlog::debug("  Channel {}: {}", idx++, wgt);
    }

    channel_weights = new_weights;
}

void chili::MultiChannel::MaxDifference(const std::vector<double> &train) {
    double max = 0;

    for(size_t i = 0; i < train.size() - 1; ++i) {
        const double wi = train[i];
        for(size_t j = i + 1; j < train.size(); ++j) {
            const double wj = train[j];

            max = std::max(max, std::abs(wi - wj));
        }
    }

    if(max < min_diff) {
        best_weights = channel_weights;
        min_diff = max;
    }
}

chili::MultiChannelSummary chili::MultiChannel::Summary(std::ostream &str) {
    summary.best_weights = best_weights;
    str << "Final integral = "
              << fmt::format("{:^8.5e} +/- {:^8.5e} ({:^8.5e} %)",
                             summary.Result().Mean(), summary.Result().Error(),
                             summary.Result().Error() / summary.Result().Mean()*100) << std::endl;
    str << fmt::format("Cut Efficiency: {}", summary.Result().Efficiency()) << std::endl;
    str << "Channel weights:\n";
    for(size_t i = 0; i < best_weights.size(); ++i) {
        str << "  alpha(" << i << ") = " << best_weights[i] << "\n";
    }
    return summary;
}

void chili::MultiChannel::PrintIteration(std::ostream &str) const {
  std::time_t t = std::time(0);
  std::tm* now = std::localtime(&t);
  double calls=static_cast<double>(summary.results.back().Calls());
  double finitecalls=static_cast<double>(summary.results.back().FiniteCalls());
  str << fmt::format("XS: {:3d}   {:^8.5e} +/- {:^8.5e} ({:^2.3f}%)   Points: {:3d}/{:3d} ({:^2.2f}%)   Total: {:3d}/{:3d}   Time: {:02d}:{:02d}:{:02d}",
            summary.results.size(),
                           summary.Result().Mean(), summary.Result().Error(),summary.Result().Error()/summary.Result().Mean()*100,
                           summary.results.back().FiniteCalls(),summary.results.back().Calls(),finitecalls/calls*100,
                           summary.Result().FiniteCalls(), summary.Result().Calls(),
                           now->tm_hour,now->tm_min,now->tm_sec) << std::endl;
}

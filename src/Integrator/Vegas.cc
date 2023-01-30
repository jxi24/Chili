#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <stdexcept>
#include <utility>

#include "Integrator/Vegas.hh"
#include "Tools/MPI.hh"

#include "fmt/format.h"
#include "spdlog/spdlog.h"


using apes::Vegas;
using apes::VegasSummary;

void Vegas::operator()(const Func<double> &func) {
    std::vector<double> rans(grid.Dims());
    std::vector<double> train_data(grid.Dims()*grid.Bins());

    StatsData results;

#ifdef ENABLE_MPI
    size_t ncalls = params.ncalls/MPIHandler::Instance().Size();
#else
    size_t ncalls = params.ncalls;
#endif

    for(size_t i = 0; i < ncalls; ++i) {
        Random::Instance().Generate(rans);

        double wgt = grid(rans);
        double val = func(rans)*wgt;
        double val2 = val * val;

        results += val;

        for(size_t j = 0; j < grid.Dims(); ++j) {
            train_data[j * grid.Bins() + grid.FindBin(j, rans[j])] += val2; 
        }
    }

#ifdef ENABLE_MPI
    MPIHandler::Instance().AllReduce<std::vector<double>, Add>(train_data);
    MPIHandler::Instance().AllReduce<StatsData, StatsAdd>(results);
#endif

    grid.Adapt(params.alpha, train_data);
    summary.results.push_back(results);
    summary.sum_results += results;
}

void Vegas::Optimize(const Func<double> &func) {
    double abs_err = lim::max(), rel_err = lim::max();
    size_t irefine = 0;
    while ((abs_err > params.atol && rel_err > params.rtol) || summary.results.size() < params.ninterations) {
        (*this)(func);
        StatsData current = summary.Result();
        abs_err = current.Error();
        rel_err = abs_err / std::abs(current.Mean());

        PrintIteration();
        if(++irefine == params.nrefine) {
            Refine();
            irefine = 0;
        }
    }
}

double Vegas::GenerateWeight(const std::vector<double> &rans) const {
    return grid.GenerateWeight(rans); 
}

void Vegas::Adapt(const std::vector<double> &train_data) {
    grid.Adapt(params.alpha, train_data);
}

void Vegas::Refine() {
    grid.Split();
    params.ncalls *= 2;
}

VegasSummary Vegas::Summary() const {
    std::cout << "Final integral = "
              << fmt::format("{:^8.5e} +/- {:^8.5e} ({:^8.5e} %)",
                             summary.Result().Mean(), summary.Result().Error(),
                             summary.Result().Error() / summary.Result().Mean()*100) << std::endl;
    return summary;
}

void Vegas::PrintIteration() const {
    std::cout << fmt::format("{:3d}   {:^8.5e} +/- {:^8.5e}    {:^8.5e} +/- {:^8.5e}",
            summary.results.size(), summary.results.back().Mean(), summary.results.back().Error(),
            summary.Result().Mean(), summary.Result().Error()) << std::endl;
}

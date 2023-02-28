#pragma once

#include "Interfaces/Python.hh"
#include "nanobind/nanobind.h"
#include "nanobind/tensor.h"
#include "nanobind/stl/unique_ptr.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace chili::nanobind {

template<typename Device>
using input = nb::tensor<double, nb::shape<nb::any, nb::any>, nb::c_contig, Device>;

template<typename Device>
using output = nb::tensor<double, nb::shape<nb::any>, nb::c_contig, Device>;

template<typename Device>
class Interface {
    public:
        Interface(const std::string &args) { integrand = chili::python::ConstructIntegrand(args); }
        output<Device> Evaluate(input<Device> in_tensor) {
            std::vector<double> out(in_tensor.shape(0));
            auto &mapping = integrand -> GetChannel(0).mapping;
            for(size_t ibatch = 0; ibatch < in_tensor.shape(0); ++ibatch) {
                std::vector<double> rans(in_tensor.shape(1));
                for(size_t irand = 0; irand < in_tensor.shape(1); ++irand) {
                    rans[irand] = in_tensor(ibatch, irand);
                }
                std::vector<chili::FourVector> point;
                mapping -> GeneratePoint(point, rans);

                if(!integrand->PreProcess()(point)) {
                    out[ibatch] = 0;
                    continue;
                }

                double weight = mapping -> GenerateWeight(point, rans);
                double val = weight == 0 ? 0 : integrand -> operator()(point)*weight;

                if(!integrand->PostProcess()(point, val)) {
                    val = 0;
                }

                out[ibatch] = val;
            }
            size_t shape[1] = {out.size()};
            return nb::tensor<double, nb::shape<nb::any>, nb::c_contig, Device>(out.data(), 1, shape);
        }
    private:
        std::unique_ptr<Integrand<FourVector>> integrand;
};

}

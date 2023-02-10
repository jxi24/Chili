#include "fmt/format.h"
#include "Interfaces/nanobind.hh"

NB_MODULE(apes_interface, m) {
    nb::class_<apes::nanobind::Interface<nb::device::cpu>>(m, "ApesCPU")
        .def(nb::init<const std::string&>())
        .def("evaluate", &apes::nanobind::Interface<nb::device::cpu>::Evaluate);
}

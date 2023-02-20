#include "fmt/format.h"
#include "Interfaces/nanobind.hh"

NB_MODULE(chili_interface, m) {
    nb::class_<chili::nanobind::Interface<nb::device::cpu>>(m, "ChiliCPU")
        .def(nb::init<const std::string&>())
        .def("evaluate", &chili::nanobind::Interface<nb::device::cpu>::Evaluate);
}

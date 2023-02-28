#pragma once

#include "Channel/Integrand.hh"
#include "Tools/FourVector.hh"

namespace chili::python {

std::unique_ptr<Integrand<FourVector>> ConstructIntegrand(const std::string&);

}

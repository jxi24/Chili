#pragma once

#include "Channel/Integrand.hh"
#include "Tools/FourVector.hh"

namespace apes::python {

std::unique_ptr<Integrand<FourVector>> ConstructIntegrand(const std::string&);

}

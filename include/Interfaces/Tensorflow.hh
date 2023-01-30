#pragma once

#include "Channel/Integrand.hh"
#include "Tools/FourVector.hh"

namespace apes::tensorflow {
Integrand<FourVector> ConstructIntegrand(const std::string &);
}

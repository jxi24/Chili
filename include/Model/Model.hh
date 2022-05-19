#pragma once

#include <cmath>
#include <functional>
#include <unordered_map>
#include <vector>

namespace apes {

class Model {
    public:
        Model(std::function<std::vector<int>(int, int)> func,
              std::unordered_map<unsigned int, double> masses = {}) 
            : m_combinable{std::move(func)}, m_masses{std::move(masses)} {}

        double& Mass(int flav) { return m_masses[static_cast<unsigned int>(std::abs(flav))]; }
        const double& Mass(int flav) const { return m_masses.at(static_cast<unsigned int>(std::abs(flav))); }
        void SetMasses(std::unordered_map<unsigned int, double> masses) { m_masses = std::move(masses); }
        std::vector<int> Combinable(int f1, int f2) const { return m_combinable(f1, f2); }

    private:
        std::function<std::vector<int>(int, int)> m_combinable;
        std::unordered_map<unsigned int, double> m_masses;
};

}

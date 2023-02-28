#ifndef INTEGRATOR_HH
#define INTEGRATOR_HH

#include <vector>

namespace chili {

class Integrator {
    public:
        virtual ~Integrator() = default;
        virtual void InitializeTraining() = 0;
        virtual void AddTrainData(double) = 0;
        virtual void Adapt() = 0;
        virtual void GeneratePoint(std::vector<double>&) const = 0;
        virtual double GenerateWeight(const std::vector<double>&) const = 0;
};

}

#endif

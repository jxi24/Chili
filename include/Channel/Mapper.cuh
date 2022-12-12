#ifndef MAPPER_CUH
#define MAPPER_CUH

#include <memory>

namespace apes::cuda {

template<typename T>
class MCFMMapper {
    public:
        // Functions
        __device__
        void GeneratePoint(T*, const double*);
        __device__
        void GenerateWeight(const T*, double*, double*);

};

}

#endif

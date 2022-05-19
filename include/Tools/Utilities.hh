#pragma once

#include <cmath>

namespace apes {

template<typename T>
bool IsClose(const T &a, const T &b, const T &eps = T{1e-8}) {
    return std::abs(a - b) < eps;
}

}

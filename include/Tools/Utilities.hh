#pragma once

#include <cmath>

namespace apes {

template<typename T>
bool IsClose(const T &a, const T &b, const T &eps = T{1e-8}) {
    return std::abs(a - b) < eps;
}

template<typename T>
T Sign(const T &a) {
  return a > 0 ? 1 : ( a < 0 ? -1 : 0 );
}

template <typename T> inline T sqr(const T &x) { return x*x; }

template<typename T> inline bool IsNan(const T& x);
template<typename T> inline bool IsBad(const T& x);

template<> inline bool IsNan<double>(const double& x) {
  return std::isnan(x)||std::isnan(-x);
}
template<> inline bool IsBad<double>(const double& x) {
  return IsNan(x)||std::isinf(x)||std::isinf(-x);
}

}

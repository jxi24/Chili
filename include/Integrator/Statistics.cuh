#ifndef STATISTICS_CUH
#define STATISTICS_CUH

#include <limits>

// Summary statistics from thrust/examples/summary_statistics.cu
template<typename T>
struct summary_stats_data {
    T n, min, max, mean, m2;

    void initialize() {
        n = mean = m2 = 0;
        min = std::numeric_limits<T>::max();
        max = std::numeric_limits<T>::min();
    }

    T variance() { return (m2 - mean*mean) / (n-1); }
    T variance_n() { return m2 / n; }
    T error() { return std::sqrt(variance() / n); }
};

template<typename T>
struct summary_stats_unary_op {
    __host__ __device__
    summary_stats_data<T> operator()(const T &x) const {
        summary_stats_data<T> result;
        result.n = 1;
        result.min = x;
        result.max = x;
        result.mean = x;
        result.m2 = 0;

        return result;
    }
};

template<typename T>
struct summary_stats_binary_op 
        : thrust::binary_function<const summary_stats_data<T>&,
                                  const summary_stats_data<T>&,
                                  summary_stats_data<T> > { 
    __host__ __device__
    summary_stats_data<T> operator()(const summary_stats_data<T> &x, const summary_stats_data<T> &y) const {
        summary_stats_data<T> result;

        T delta = y.mean - x.mean;
        T delta2 = delta * delta;
        
        result.n = x.n + y.n;
        result.min = thrust::min(x.min, y.min);
        result.max = thrust::max(x.max, y.max);

        result.mean = x.mean + delta * y.n / result.n;
        result.m2 = x.m2 + y.m2;
        result.m2 += delta2 * x.n * y.n / result.n;

        return result;
    }
};

#endif

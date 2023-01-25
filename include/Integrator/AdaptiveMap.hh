#ifndef ADAPTIVE_MAP_HH
#define ADAPTIVE_MAP_HH

#include <array>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <iosfwd>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include "yaml-cpp/yaml.h"
#pragma GCC diagnostic pop

namespace apes {

using lim = std::numeric_limits<double>;

enum class AdaptiveMapSplit {
    half,
    third,
    quarter
};

class AdaptiveMap {
    public:
        AdaptiveMap() = default;
        AdaptiveMap(size_t dims, size_t bins) 
            : m_dims(std::move(dims)), m_bins(std::move(bins)) {
                m_hist.resize(dims * (bins+1));
                for(size_t i = 0; i < m_bins + 1; ++i) {
                    m_hist[i] = static_cast<double>(i)/static_cast<double>(bins);
                }

                for(size_t i = 1; i < m_dims; ++i) {
                    std::copy(m_hist.begin(), m_hist.begin() + static_cast<int>(bins + 1),
                              m_hist.begin() + static_cast<int>(i * (bins + 1)));
                }
            }

        // Serialization
        bool Deserialize(std::istream &in);
        bool Serialize(std::ostream &out) const;

        // Bin locations
        double lower_edge(size_t dim, size_t bin) const { return m_hist[dim*(m_bins+1) + bin]; }
        double upper_edge(size_t dim, size_t bin) const { return m_hist[dim*(m_bins+1) + bin + 1]; }
        double width(size_t dim, size_t bin) const { return upper_edge(dim, bin) - lower_edge(dim, bin); }
        size_t FindBin(size_t, double) const;

        // Map information
        std::vector<double>::const_iterator Edges(size_t dim) const { 
            return m_hist.cbegin() + static_cast<int>(dim*(m_bins+1));
        }
        size_t Bins() const { return m_bins; }
        size_t Dims() const { return m_dims; }
        // Used for testing purposes
        const std::vector<double>& Hist() const { return m_hist; }
        std::vector<double>& Hist() { return m_hist; }

        // Generate random numbers
        double operator()(std::vector<double>&);
        double GenerateWeight(const std::vector<double>&) const;

        // Update histograms
        void Adapt(const double&, const std::vector<double>&);
        void Split(AdaptiveMapSplit split = AdaptiveMapSplit::half);

    private:
        std::vector<double> m_hist;
        size_t m_dims{}, m_bins{};
};

}

#endif

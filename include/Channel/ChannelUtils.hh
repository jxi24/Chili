#pragma once

#include <array>
#include <vector>

namespace chili {

// Bit operations taken from: https://graphics.stanford.edu/~seander/bithacks.html
inline unsigned int NextPermutation(unsigned int inp) {
    unsigned int t = inp | (inp - 1);
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(inp) + 1));
}

inline bool BitIsSet(unsigned int inp, unsigned int pos) {
  return inp & (1u << pos);
}

inline std::vector<unsigned int> BitsAreSet(unsigned int inp, unsigned int size) {
    std::vector<unsigned int> set;
    for(unsigned int i = 0; i < size; ++i) {
        if(BitIsSet(inp, i)) set.push_back(inp & (1u << i));
    }
    return set;
}

inline unsigned int NSetBits(unsigned int inp) {
    unsigned int count;
    for(count = 0; inp; count++) inp &= inp - 1; // Clear the least significant bit set
    return count;
}

inline bool HaveCommonBitSet(unsigned int i, unsigned int j) {
    return i & j;
}

inline bool IsPower2(unsigned int val) {
    return (val & (val - 1)) == 0;
}

inline unsigned int Log2(unsigned int val) {
    static const std::array<unsigned int, 5> b = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 
                                                  0xFF00FF00, 0xFFFF0000};
    unsigned int r = (val & b[0]) != 0;
    r |= static_cast<unsigned int>((val & b[4]) != 0) << 4;
    r |= static_cast<unsigned int>((val & b[3]) != 0) << 3;
    r |= static_cast<unsigned int>((val & b[2]) != 0) << 2;
    r |= static_cast<unsigned int>((val & b[1]) != 0) << 1;
    return r;
}

}

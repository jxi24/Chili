#include "Channel/ChannelUtils.hh"
#include <iostream>
#include <stack>
#include <vector>

struct DataFrame {
    unsigned int current;
    size_t schans{};
    std::vector<unsigned int> currents;
    std::vector<unsigned int> avail_currents;
};

bool contained(unsigned int i, unsigned int j) {
    return i & j;
}

int main() {
    std::stack<DataFrame> s;
    std::vector<unsigned int> currents = {1, 2, 4, 8, 16};
    size_t npart = currents.size();
    size_t smax = 2;
    std::vector<unsigned int> new_currents;
    for(size_t i = 0; i < currents.size(); ++i) {
        new_currents.push_back(currents[i]);
        for(size_t j = i+1; j < currents.size(); ++j) {
            if(j < 2 || i < 2) continue;
            new_currents.push_back(currents[i] + currents[j]); 
            for(size_t k = j+1; k < currents.size(); ++k) {
                new_currents.push_back(currents[i] + currents[j] + currents[k]); 
                for(size_t l = k+1; l < currents.size(); ++l) {
                    new_currents.push_back(currents[i] + currents[j] + currents[k] + currents[l]); 
                    for(size_t m = l+1; m < currents.size(); ++m) {
                        new_currents.push_back(currents[i] + currents[j] + currents[k] + currents[l] + currents[m]); 
                        for(size_t n = m+1; n < currents.size(); ++n) {
                            new_currents.push_back(currents[i] + currents[j] + currents[k] + currents[l] + currents[m] + currents[n]); 
                        }
                    }
                }
            }
        }
    }
    std::cout << "currents: ";
    for(const auto &c : new_currents) {
        std::cout  << c << " ";
    }
    std::cout << "\n";
    size_t max_id = (1 << npart) - 1;
    s.push({ 0, 0, {}, new_currents });
    int channels = 0;
    while(!s.empty()) {
        DataFrame top = s.top();
        s.pop();
        if(top.current > max_id) continue;
        if(top.current == max_id) {
            ++channels;
            // std::cout << "Channel: ";
            // for(const auto &c : top.currents) {
            //     std::cout  << c << " ";
            // }
            // std::cout << "\n";
            continue;
        }
        if(top.avail_currents.empty()) continue;
        for(size_t i = 0; i < top.avail_currents.size(); ++i) {
            DataFrame d = top;
            d.schans += apes::BitsAreSet(top.avail_currents[i], static_cast<unsigned int>(npart)).size() - 1;
            if(d.schans > smax) {
                d.schans = top.schans;
                d.avail_currents.erase(d.avail_currents.begin()+static_cast<int>(i));
                continue;
            }
            if(!d.currents.empty()) {
                if(d.currents.back() > top.avail_currents[i]) continue;
            }
            if(!contained(d.current, top.avail_currents[i])) {
                d.current += top.avail_currents[i];
                d.currents.push_back(top.avail_currents[i]);
                d.avail_currents.erase(d.avail_currents.begin()+static_cast<int>(i));
                s.push(d);
            }
        }
    }
    std::cout << npart << " " << channels << std::endl;

    return 0;
}

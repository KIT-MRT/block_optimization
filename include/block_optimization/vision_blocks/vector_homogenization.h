#pragma once

#include <limits>
#include "../block.h"

namespace block_optimization {

template <typename T>
class VectorHomogenization : public Block<T> {
public:
    VectorHomogenization(std::string name) : Block<T>(name, "VectorHomogenization", 0, 3, 2) {
        ;
    }

    virtual void computeOutput(T const* const* params, T const* input, T* output) const override {
        T inv = 1. / (std::max(input[2], std::numeric_limits<T>::epsilon()));
        output[0] = input[0] * inv;
        output[1] = input[1] * inv;
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        T inv = 1. / (std::max(input[2], std::numeric_limits<T>::epsilon()));

        //    1/z 0 -x/z²
        //    0 1/z -y/z²
        if (jacobianWRTinput) {
            jacobianWRTinput[0] = inv;
            jacobianWRTinput[1] = T(0.);
            jacobianWRTinput[2] = -input[0] * std::pow(inv, 2);
            jacobianWRTinput[3] = T(0.);
            jacobianWRTinput[4] = inv;
            jacobianWRTinput[5] = -input[1] * std::pow(inv, 2);
        }
    }


protected:
};

} // namespace block_optimization

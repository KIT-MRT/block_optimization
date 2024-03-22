#pragma once

#include <iostream>
#include "../block.h"

namespace block_optimization {

template <typename T, int dims>
class DataBlock : public block_optimization::Block<T> {
public:
    DataBlock(std::string name) : Block<T>(name, "DataBlock", dims, 0, dims) {
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const override {
        for (int i = 0; i < dims; i++) {
            ouput[i] = params[0][i];
        }
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        if (jacobianWRTinput) {
            *jacobianWRTinput = T(0.);
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < dims; j++) {
                    if (i == j) {
                        jacobianWRTparams[0][i * dims + j] = T(1.);
                    } else {
                        jacobianWRTparams[0][i * dims + j] = T(0.);
                    }
                }
            }
        }
    }
};

} // namespace block_optimization

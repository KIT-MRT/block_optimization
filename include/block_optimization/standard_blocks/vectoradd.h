#pragma once

#include "../block.h"

namespace block_optimization {

template <typename T, int dims>
class VectorAdd : public Block<T> {
public:
    VectorAdd(std::string name) : Block<T>(name, "VectorAdd", dims, dims, dims) {
        ;
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const {
        for (int i = 0; i < dims; i++) {
            ouput[i] = input[i] + params[0][i];
        }
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const {
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < dims; j++) {
                    if (i != j) {
                        jacobianWRTparams[0][i * dims + j] = T(0.);
                    } else {
                        jacobianWRTparams[0][i * dims + j] = T(1.);
                    }
                }
            }
        }
        if (jacobianWRTinput) {
            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < dims; j++) {
                    if (i != j) {
                        jacobianWRTinput[i * dims + j] = T(0.);
                    } else {
                        jacobianWRTinput[i * dims + j] = T(1.);
                    }
                }
            }
        }
    }
};

} // namespace block_optimization

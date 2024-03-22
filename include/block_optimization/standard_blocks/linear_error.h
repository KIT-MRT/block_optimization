#pragma once

#include "../block.h"

namespace block_optimization {

template <typename T, int dims>
class LinearError : public block_optimization::Block<T> {
public:
    LinearError(std::string name) : Block<T>(name, "LinearError", dims, dims, dims) {
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const override {
        //    std::cout << "err: " << std::endl;
        for (int i = 0; i < dims; i++) {
            ouput[i] = input[i] - params[0][i];
            //      std::cout << ouput[i] << " = " << input[i] << " - " <<  params[0][i] << std::endl;
        }
        //    std::cout << std::endl;
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        if (jacobianWRTinput) {
            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < dims; j++) {
                    if (i == j) {
                        jacobianWRTinput[i * dims + j] = T(1.);
                    } else {
                        jacobianWRTinput[i * dims + j] = T(0.);
                    }
                }
            }
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < dims; j++) {
                    if (i == j) {
                        jacobianWRTparams[0][i * dims + j] = -T(1.);
                    } else {
                        jacobianWRTparams[0][i * dims + j] = T(0.);
                    }
                }
            }
        }
    }
};

} // namespace block_optimization

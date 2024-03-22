#pragma once

#include "block_optimization.h"

namespace line_test {

template <typename T>
class AdditiveConstant : public block_optimization::Block<T> {
public:
    AdditiveConstant() : block_optimization::Block<T>("a", "AdditiveConstant", 1, 1, 1) {
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const override {
        *ouput = *input + *params[0];
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        if (jacobianWRTinput) {
            *jacobianWRTinput = T(1.);
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            *jacobianWRTparams[0] = T(1.);
        }
    }
};

template <typename T>
class MultiplicativeConstant : public block_optimization::Block<T> {
public:
    MultiplicativeConstant() : block_optimization::Block<T>("a", "MultiplicativeConstant", 1, 1, 1) {
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const override {
        *ouput = *params[0] * (*input);
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        if (jacobianWRTinput) {
            *jacobianWRTinput = *params[0];
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            *jacobianWRTparams[0] = *input;
        }
    }
};

template <typename T>
class LinearError : public block_optimization::Block<T> {
public:
    LinearError() : block_optimization::Block<T>("a", "LinearError", 1, 1, 1) {
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const override {
        *ouput = *input - (*params[0]); //(T(10.)*(*params[0]) + T(5.));
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        if (jacobianWRTinput) {
            *jacobianWRTinput = T(1.);
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            *jacobianWRTparams[0] = T(-10.);
        }
    }
};

template <typename T>
class DataBlock : public block_optimization::Block<T> {
public:
    DataBlock() : block_optimization::Block<T>("a", "DataBlock", 1, 0, 1) {
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const override {
        *ouput = *params[0];
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        if (jacobianWRTinput) {
            *jacobianWRTinput = T(0.);
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            *jacobianWRTparams[0] = T(1.);
        }
    }
};

} // namespace line_test

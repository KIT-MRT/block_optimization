#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../block.h"

namespace block_optimization {

template <typename T>
class PolyDistortion : public Block<T> {
public:
    PolyDistortion(std::string name) : Block<T>(name, "PolyDistortion", 5, 2, 2) {
        ;
    }

    virtual void computeOutput(T const* const* params, T const* input, T* output) const {
        T k1 = params[0][0];
        T k2 = params[0][1];
        T p1 = params[0][2];
        T p2 = params[0][3];
        T k3 = params[0][4];

        T x = input[0];
        T y = input[1];

        T rsq = input[0] * input[0] + input[1] * input[1];
        T dist = ((rsq * k3 + k2) * rsq + k1) * rsq + T(1.);

        T tangential[2];
        tangential[0] = 2. * p1 * x * y + p2 * (rsq + 2 * std::pow(x, 2));
        tangential[1] = p1 * (rsq + 2 * std::pow(y, 2)) + 2. * p2 * x * y;

        // TODO tangential distortion, k3

        output[0] = dist * input[0] + tangential[0];
        output[1] = dist * input[1] + tangential[1];
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const {

        T k1 = params[0][0];
        T k2 = params[0][1];
        T p1 = params[0][2];
        T p2 = params[0][3];
        T k3 = T(0.); // params[0][4];
        T rsq = input[0] * input[0] + input[1] * input[1];
        T x = input[0];
        T y = input[1];

        T dist = ((rsq * k3 + k2) * rsq + k1) * rsq + T(1.);
        if (jacobianWRTinput) {
            jacobianWRTinput[0] =
                dist + x * (k1 * T(2.) * x + k2 * T(2.) * rsq * T(2.) * x + T(3.) * k3 * std::pow(rsq, 2) * T(2.) * x);
            jacobianWRTinput[1] = y * (k1 * 2. * x + k2 * 2. * rsq * 2. * x + k3 * std::pow(rsq, 2) * 2. * x);
            jacobianWRTinput[2] = x * (k1 * 2. * y + k2 * 2. * rsq * 2. * y + k3 * std::pow(rsq, 2) * 2. * y);
            jacobianWRTinput[3] =
                dist + y * (k1 * 2. * y + k2 * 2. * rsq * 2. * y + 3 * k3 * std::pow(rsq, 2) * 2. * y);
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            jacobianWRTparams[0][0] = x * rsq;
            jacobianWRTparams[0][1] = x * std::pow(rsq, 2);
            jacobianWRTparams[0][2] = 2 * x * y;
            jacobianWRTparams[0][3] = rsq + 2. * std::pow(x, 2);
            jacobianWRTparams[0][4] = x * std::pow(rsq, 3);

            jacobianWRTparams[0][5] = y * rsq;
            jacobianWRTparams[0][6] = y * std::pow(rsq, 2);
            jacobianWRTparams[0][7] = rsq + 2. * std::pow(y, 2);
            jacobianWRTparams[0][8] = 2 * x * y;
            jacobianWRTparams[0][9] = y * std::pow(rsq, 3);
        }
    }

    virtual void getDistortionCoeffs(cv::Mat& d) {
        if (!this->params_[0]) {
            throw std::runtime_error("No internal memory available for block " + this->getName());
        }
        d = cv::Mat(1, 5, CV_64FC1, cv::Scalar(0.));
        for (int i = 0; i < 5; i++) {
            d.at<double>(i) = this->params_[0].get()[i];
        }
    }
};

} // namespace block_optimization

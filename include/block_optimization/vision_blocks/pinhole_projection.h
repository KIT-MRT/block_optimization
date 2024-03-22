#pragma once

#include <opencv2/opencv.hpp>
#include "../block.h"

namespace block_optimization {

template <typename T>
class PinholeProjection : public Block<T> {
public:
    PinholeProjection(std::string name) : Block<T>(name, "PinholeProjection", 4, 2, 2) {
        ;
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const {
        ouput[0] = params[0][0] * input[0] + params[0][1];
        ouput[1] = params[0][2] * input[1] + params[0][3];
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const {
        if (jacobianWRTinput) {
            jacobianWRTinput[0] = params[0][0];
            jacobianWRTinput[1] = T(0.);
            jacobianWRTinput[2] = T(0.);
            jacobianWRTinput[3] = params[0][2];
        }
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            jacobianWRTparams[0][0] = input[0];
            jacobianWRTparams[0][1] = T(1.);
            jacobianWRTparams[0][2] = T(0.);
            jacobianWRTparams[0][3] = T(0.);

            jacobianWRTparams[0][4] = T(0.);
            jacobianWRTparams[0][5] = T(0.);
            jacobianWRTparams[0][6] = input[1];
            jacobianWRTparams[0][7] = T(1.);
        }
    }

    virtual void getCameraMatrix(cv::Mat& K) {
        if (!this->params_[0]) {
            throw std::runtime_error("No internal memory available for block " + this->getName());
        }
        K = cv::Mat::eye(3, 3, CV_64FC1);
        K.at<double>(0, 0) = this->params_[0].get()[0];
        K.at<double>(0, 2) = this->params_[0].get()[1];
        K.at<double>(1, 1) = this->params_[0].get()[2];
        K.at<double>(1, 2) = this->params_[0].get()[3];
    }
};

} // namespace block_optimization

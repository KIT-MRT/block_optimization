#pragma once

#include "../block.h"
#include "../processing_chain.h"

#include "pinhole_camera.h"
#include "vector_homogenization.h"
#include "../standard_blocks/transform_6dof.h"

#include <opencv2/opencv.hpp>

namespace block_optimization {


// this block is a metablock for two independent
// processing chains of a stereo camera setup.
// This kind of defeats the sparse processing
// characteristic of the problem, so for performance reasons,
// you shouldn't really do this. For convenience purposes on the other
// hand, here you have a simple way to optimize a stereo setup

template <typename T>
class StereoCamera : public Block<T> {
public:
    StereoCamera(std::string name) : Block<T>(name, "StereoCamera", 0, 3, 4) {
        c1_.reset(new PinholeCamera<T>(this->getName() + "_c1"));
        c2_.reset(new PinholeCamera<T>(this->getName() + "_c2"));
        transform_.reset(new Transform6DOF<T>(this->getName() + "_t"));
        homogenization_.reset(new VectorHomogenization<T>(this->getName() + "_h"));

        c1Chain_.reset(new ProcessingChain<T>());
        c1Chain_->appendBlock(homogenization_);
        c1Chain_->appendBlock(c1_);
        c2Chain_.reset(new ProcessingChain<T>());
        c2Chain_->appendBlock(transform_);
        c2Chain_->appendBlock(homogenization_);
        c2Chain_->appendBlock(c2_);

        this->paramCount_.clear();
        this->paramCount_.push_back(5); // c1 distortion
        this->paramCount_.push_back(4); // c1 projection
        this->paramCount_.push_back(3); // c2 rotation
        this->paramCount_.push_back(3); // c2 translation
        this->paramCount_.push_back(5); // c2 distortion
        this->paramCount_.push_back(4); // c2 projection
    }

    virtual void computeOutput(T const* const* params, T const* input, T* output) const override {
        T const** c1Params = new T const*[c1Chain_->getParameterBlockSizes().size()];
        T const** c2Params = new T const*[c2Chain_->getParameterBlockSizes().size()];

        this->augmentParams(params, true, c1Params);
        c1Chain_->forwardChain(c1Params, input, &output[0]);
        this->augmentParams(params, false, c2Params);
        c2Chain_->forwardChain(c2Params, input, &output[2]);

        delete[] c1Params;
        c1Params = NULL;
        delete[] c2Params;
        c2Params = NULL;
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        // set jacobians w.r.t. params to zero
        for (int i = 0; i < this->paramCount_.size(); i++) {
            for (int k = 0; k < 4 * this->paramCount_[i]; k++) {

                jacobianWRTparams[i][k] = T(0.);
            }
        }
        T** c1JacobiansWRTparams = new T*[c1Chain_->getParameterBlockSizes().size()];
        T** c2JacobiansWRTparams = new T*[c2Chain_->getParameterBlockSizes().size()];

        this->augmentJacobians(jacobianWRTparams, true, c1JacobiansWRTparams);
        this->augmentJacobians(jacobianWRTparams, false, c2JacobiansWRTparams);

        T* c1JacobianWRTinput = &jacobianWRTinput[0];
        T* c2JacobianWRTinput = &jacobianWRTinput[4];

        c1Chain_->backwardChain(c1JacobiansWRTparams, c1JacobianWRTinput);
        c2Chain_->backwardChain(c2JacobiansWRTparams, c2JacobianWRTinput);

        delete[] c1JacobiansWRTparams;
        delete[] c2JacobiansWRTparams;
    }

    virtual void setInternalMemory() {
        c1_->setInternalMemory();
        c2_->setInternalMemory();
        transform_->setInternalMemory();
    }

    virtual std::shared_ptr<T> getInternalMemory(int index) {
        if (index < 0) {
            throw std::runtime_error("Negative index for internal memory requested for " + this->getName());
        } else if (index < 2) {
            return c1_->getInternalMemory(index);
        } else if (index < 4) {
            return c2_->getInternalMemory(index - 2);
        } else if (index < 6) {
            return transform_->getInternalMemory(index - 4);
        } else {
            throw std::runtime_error("Index exceeds internal memory of " + this->getName());
        }
        return nullptr;
    }

    virtual std::vector<std::shared_ptr<T>> getInternalMemory() {
        throw std::runtime_error("Vector internal memory access not implemented for " + this->getType() + " yet");
    }

    virtual void getDistortionCoeffs(cv::Mat& d, bool c1 = true) {
        if (c1) {
            c1_->distortion_->getDistortionCoeffs(d);
        } else {
            c2_->distortion_->getDistortionCoeffs(d);
        }
    }
    virtual void getCameraMatrix(cv::Mat& K, bool c1 = true) {
        if (c1) {
            c1_->projection_->getCameraMatrix(K);
        } else {
            c2_->projection_->getCameraMatrix(K);
        }
    }

    virtual void getRotation(Eigen::Matrix<T, 3, 3>& R) {
    }
    virtual void getRotation(cv::Mat& R) {
    }
    virtual void getTranslation(Eigen::Matrix<T, 3, 1>& t) {
    }
    virtual void getTranslation(cv::Mat& t) {
    }


    std::shared_ptr<PinholeCamera<T>> c1_;
    std::shared_ptr<PinholeCamera<T>> c2_;
    std::shared_ptr<Transform6DOF<T>> transform_;
    std::shared_ptr<VectorHomogenization<T>> homogenization_;

    std::shared_ptr<ProcessingChain<T>> c1Chain_;
    std::shared_ptr<ProcessingChain<T>> c2Chain_;

protected:
    void augmentParams(T const* const* inputParams, const bool c1, T const** newParams) const {
        if (c1) {
            newParams[0] = inputParams[0]; // distortion
            newParams[1] = inputParams[1]; // projection
        } else {
            newParams[0] = inputParams[2]; // rotation
            newParams[1] = inputParams[3]; // translation
            newParams[2] = inputParams[4]; // distortion
            newParams[3] = inputParams[5]; // projection
        }
    }

    void augmentJacobians(T** inputJacobianWRTparams, const bool c1, T** newInputJacobianWRTparams) const {
        if (c1) {
            // first two rows of both, first two parameter blocks
            newInputJacobianWRTparams[0] = &inputJacobianWRTparams[0][0];
            newInputJacobianWRTparams[1] = &inputJacobianWRTparams[1][0];
        } else {
            // rotation, second two rows
            newInputJacobianWRTparams[0] = &inputJacobianWRTparams[2][6];
            // translation, second two rows
            newInputJacobianWRTparams[1] = &inputJacobianWRTparams[3][6];
            // distortion, second two rows
            newInputJacobianWRTparams[2] = &inputJacobianWRTparams[5][10];
            // projection, second two rows
            newInputJacobianWRTparams[3] = &inputJacobianWRTparams[2][8];
        }
    }
};

} // namespace block_optimization

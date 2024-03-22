#pragma once

#include "pinhole_projection.h"
#include "poly_distortion.h"
#include "../block.h"
#include "../processing_chain.h"

namespace block_optimization {


template <typename T>
class PinholeCamera : public Block<T> {
public:
    PinholeCamera(std::string name) : Block<T>(name, "PinholeCamera", 0, 2, 2) {
        // create the individual processing blocks
        distortion_.reset(new PolyDistortion<T>(this->getName() + "_distort"));
        projection_.reset(new PinholeProjection<T>(this->getName() + "_project"));

        // create the chain
        processChain_.reset(new ProcessingChain<T>());
        processChain_->appendBlock(distortion_);
        processChain_->appendBlock(projection_);

        // just in case
        this->paramCount_.clear();
        this->paramCount_.push_back(5);
        this->paramCount_.push_back(4);
    }

    ~PinholeCamera() {
        ;
    }

    virtual void computeOutput(T const* const* params, T const* input, T* output) const override {
        // I have computation of jacobians on because I don't know what is going to happen with this
        processChain_->forwardChain(params, input, output);
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const override {
        // jacobian data is memorized internally. This might violate the const assumption?
        processChain_->backwardChain(jacobianWRTparams, jacobianWRTinput);
    }

    virtual void setInternalMemory() {
        distortion_->setInternalMemory();
        projection_->setInternalMemory();
    }


    virtual std::shared_ptr<T> getInternalMemory(int index) {
        if (index < 0) {
            throw std::runtime_error("Negative index for internal memory requested for " + this->getName());
        } else if (index == 0) {
            return distortion_->getInternalMemory(0);
        } else if (index == 1) {
            return projection_->getInternalMemory(0);
        } else {
            throw std::runtime_error("Index exceeds internal memory of " + this->getName());
        }
        return nullptr;
    }

    virtual std::vector<std::shared_ptr<T>> getInternalMemory() {
        throw std::runtime_error("Vector internal memory access not implemented for " + this->getType() + " yet");
    }

    std::shared_ptr<PolyDistortion<T>> distortion_;
    std::shared_ptr<PinholeProjection<T>> projection_;

protected:
    std::shared_ptr<ProcessingChain<T>> processChain_;
};

} // namespace block_optimization

#pragma once

#pragma once

#include "rotation.h"
#include "vectoradd.h"
#include "../block.h"

namespace block_optimization {


template <typename T>
class Transform6DOF : public Block<T> {
public:
    Transform6DOF(std::string name) : Block<T>(name, "Transform6DOF", std::vector<int>(2, 3), 3, 3) {
        rotation_.reset(new Rotation<T>(this->getName() + "_rot"));
        translation_.reset(new VectorAdd<T, 3>(this->getName() + "_trans"));
    }

    ~Transform6DOF() {
        ;
    }

    virtual void computeOutput(T const* const* params, T const* input, T* output) const {
        std::shared_ptr<T> tempOut(new T[3], std::default_delete<T[]>());
        rotation_->computeOutput(&params[0], input, tempOut.get());
        translation_->computeOutput(&params[1], tempOut.get(), output);
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const {
        // translation doesn't play a role here
        // b.c. derivative of R*x + t is R which, by the way, is the
        // same as pure rotation
        if (!jacobianWRTparams && !jacobianWRTinput) {
            return;
        } else if (!jacobianWRTparams) {
            rotation_->computeJacobians(&params[0], input, NULL, jacobianWRTinput);
            translation_->computeJacobians(&params[1], input, NULL, NULL);
        } else {
            rotation_->computeJacobians(&params[0], input, &jacobianWRTparams[0], jacobianWRTinput);
            translation_->computeJacobians(&params[1], input, &jacobianWRTparams[1], NULL);
        }
    }


    virtual void setInternalMemory() {
        rotation_->setInternalMemory();
        translation_->setInternalMemory();
    }

    virtual std::shared_ptr<T> getInternalMemory(int index) {
        if (index == 0) {
            return rotation_->getInternalMemory(0);
        } else if (index == 1) {
            return translation_->getInternalMemory(0);
        } else {
            throw std::runtime_error("Index exceeds internal memory of " + this->getType());
        }
    }

    virtual std::vector<std::shared_ptr<T>> getInternalMemory() {
        return std::vector<std::shared_ptr<T>>{rotation_->getInternalMemory(0), translation_->getInternalMemory(0)};
    }

    virtual void getRotationMatrix(Eigen::Matrix<T, 3, 3>& R) {
        if (!rotation_->getInternalMemory(0)) {
            throw std::runtime_error("Internal memory not set");
        } else {
            rotation_->getRotationMatrix(rotation_->getInternalMemory(0).get(), R);
        }
    }
    virtual void getTranslation(Eigen::Matrix<T, 3, 1>& t) {
        if (!translation_->getInternalMemory(0)) {
            throw std::runtime_error("Internal memory not set");
        } else {
            T* data = translation_->getInternalMemory(0).get();
            t(0, 0) = data[0];
            t(1, 0) = data[1];
            t(2, 0) = data[2];
        }
    }
    // this is not modeled as a chain because the backward jacobian
    // of the translation is identity, thus more efficient
    std::shared_ptr<Rotation<T>> rotation_;
    std::shared_ptr<VectorAdd<T, 3>> translation_;

protected:
};

} // namespace block_optimization

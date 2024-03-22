#pragma once

#include <eigen3/Eigen/Dense>
#include "processing_chain.h"


namespace block_optimization {


template <typename T>
class SGDSolver {
    struct ChainSetup {
        ProcessingChainPtr<T> chain;
        std::vector<T*> parameterBlocks;
        int inputSize;
        int outputSize;
        std::vector<int> paramCount;
        std::shared_ptr<T> output;
        std::shared_ptr<EigenMap<T>> outputVec;
        std::vector<std::shared_ptr<T>> jacobians;
        double maxAllowedResiduum;
    };

    struct GradientSetup {
        bool toOptimize;
        Eigen::Matrix<T, 1, Eigen::Dynamic> gradient;
        std::shared_ptr<Eigen::Matrix<T, 1, Eigen::Dynamic>> subParametrization;

        GradientSetup() {
            toOptimize = true;
            subParametrization = nullptr;
        }
    };

public:
    SGDSolver() {
        currentSampleIdx_ = 0;
        learningRate_ = 1e-3;
        batchSize_ = 10;
    }

    virtual void addChain(ProcessingChainPtr<T> chain, std::vector<T*>& parameterBlocks) {
        ChainSetup thisSetup;
        thisSetup.maxAllowedResiduum = 10;
        thisSetup.chain = chain;
        thisSetup.parameterBlocks = parameterBlocks;
        thisSetup.inputSize = chain->getNumInputs();
        thisSetup.outputSize = chain->getNumResiduals();
        chain->getParameterBlockSizes(thisSetup.paramCount);
        if (parameterBlocks.size() != thisSetup.paramCount.size()) {
            // TODO: more verbose
            throw std::runtime_error("Parameter block count in chain and supplied parameters block count differ");
        }

        thisSetup.output = std::shared_ptr<T>(new T[thisSetup.outputSize], std::default_delete<T[]>());
        // transposed already
        thisSetup.outputVec =
            std::shared_ptr<EigenMap<T>>(new EigenMap<T>(thisSetup.output.get(), 1, thisSetup.outputSize));
        thisSetup.jacobians.resize(parameterBlocks.size());
        for (unsigned int i = 0; i < parameterBlocks.size(); i++) {
            // TODO: check if optimization is desired, if no, set NULL
            if (i == parameterBlocks.size() - 1) {
                thisSetup.jacobians[i] = nullptr;
            } else {

                thisSetup.jacobians[i] = std::shared_ptr<T>(new T[thisSetup.outputSize * thisSetup.paramCount[i]],
                                                            std::default_delete<T[]>());
            }
        }

        for (unsigned int i = 0; i < thisSetup.paramCount.size(); i++) {
            if (i == thisSetup.paramCount.size() - 1 || !thisSetup.paramCount[i] || !parameterBlocks[i]) {
                continue;
            }
            int64_t idx = reinterpret_cast<int64_t>(parameterBlocks[i]);
            GradientSetup thisGradSetup;
            thisGradSetup.gradient = Eigen::Matrix<T, 1, Eigen::Dynamic>(thisSetup.paramCount[i]);
            gradients_[idx] = thisGradSetup;
        }

        chainSetups_.push_back(thisSetup);
    }

    virtual double step(std::vector<std::vector<T*>>& data, std::vector<std::vector<T*>>& target) {
        if (data.empty() || target.empty()) {
            //      std::cout << "data or target empty" << std::endl;
            return 0;
        }
        double error = 0;
        int measCount = 0;
        // set gradients to zero
        this->resetGradients();
        // go through all observations in one batch
        int samplesInBatch = batchSize_ > 0 ? batchSize_ : data.size();
        for (int sampleCount = 0; sampleCount < samplesInBatch; sampleCount++) {
            int observIdx = currentSampleIdx_ % data.size();
            // go through all chains in observation
            for (unsigned int chainIdx = 0; chainIdx < chainSetups_.size(); chainIdx++) {
                // check if observed
                if (!target[observIdx][chainIdx]) {
                    continue;
                }
                // for sake of legibility
                ChainSetup& thisSetup = chainSetups_[chainIdx];

                // get input pointer
                T* input = NULL;
                if (thisSetup.inputSize > 0) {
                    input = data[observIdx][chainIdx];
                } else {
                    // this might be a new parameter block
                    int64_t idx = reinterpret_cast<int64_t>(data[observIdx][chainIdx]);
                    if (gradients_.find(idx) == gradients_.end()) {
                        // if doesn't exist in map yet, add
                        GradientSetup thisGradSetup;
                        thisGradSetup.gradient = Eigen::Matrix<T, 1, Eigen::Dynamic>::Zero(thisSetup.paramCount[0]);
                        gradients_[idx] = thisGradSetup;
                    }
                    thisSetup.parameterBlocks[0] = data[observIdx][chainIdx];
                }

                // set the last parameters block to be the target vector
                thisSetup.parameterBlocks[thisSetup.paramCount.size() - 1] = target[observIdx][chainIdx];

                // copy the jacobian pointers
                T** jacobians = new T*[thisSetup.paramCount.size()];
                for (unsigned int i = 0; i < thisSetup.paramCount.size(); i++) {
                    jacobians[i] = thisSetup.jacobians[i].get();
                }

                // for sake of legibility
                ProcessingChain<T>& thisChain = *thisSetup.chain;

                // do full processing of the chain
                thisChain.process(thisSetup.parameterBlocks.data(), input, thisSetup.output.get(), jacobians, NULL);

                // TODO: conditional error add: outlier rejection
                if (thisSetup.outputVec->norm() > thisSetup.maxAllowedResiduum) {
                    delete[] jacobians;
                    jacobians = NULL;
                    continue;
                }

                error += thisSetup.outputVec->norm();
                measCount++;

                for (unsigned int jacobianIdx = 0; jacobianIdx < thisSetup.paramCount.size(); jacobianIdx++) {
                    // the pointer to the parameter vector uniquely identifies the parameter block
                    int64_t dataPtr = reinterpret_cast<int64_t>(thisSetup.parameterBlocks[jacobianIdx]);
                    if (!dataPtr || !jacobians[jacobianIdx]) {
                        // if this pointer doesn't exist, I probably don't want to optimize this
                        continue;
                    }
                    EigenMap<T> jacobian(
                        jacobians[jacobianIdx], thisSetup.outputSize, thisSetup.paramCount[jacobianIdx]);
                    bool isnormal = true;
                    for (int i = 0; i < jacobian.rows(); i++) {
                        for (int j = 0; j < jacobian.cols(); j++) {
                            isnormal &= std::isnormal(jacobian(i, j)) || jacobian(i, j) == 0.0;
                        }
                    }
                    for (int i = 0; i < gradients_[dataPtr].gradient.cols(); i++) {
                        isnormal &= std::isnormal(gradients_[dataPtr].gradient.coeff(i)) ||
                                    gradients_[dataPtr].gradient.coeff(i) == 0.0;
                    }
                    // do not add inf or nan values to gradient
                    if (!isnormal) {
                        continue;
                    }
                    // TODO: this is valid exclusively for least squares problems.
                    gradients_[dataPtr].gradient += *thisSetup.outputVec * jacobian;
                }
                delete[] jacobians;
                jacobians = NULL;
            }
            ++currentSampleIdx_;
        }
        // perform the step
        this->updateParameters();
        //    std::cout << "Error " << error/double(measCount) << std::endl;
        return error / double(measCount);
    }

    virtual void resetGradients() {
        // TODO: this might also be non-zero, i.e. momentum
        for (std::pair<const int64_t, GradientSetup>& thisGradient : gradients_) {
            thisGradient.second.gradient.setZero();
        }
    }

    virtual void updateParameters() {
        // performs w = w - weight*grad
        for (std::pair<const int64_t, GradientSetup>& thisGradient : gradients_) {
            if (!thisGradient.first || !thisGradient.second.toOptimize) {
                continue;
            }

            //      std::cout << "Update: " << learningRate_*thisGradient.second << std::endl;
            T* params = reinterpret_cast<T*>(thisGradient.first);

            if (thisGradient.second.subParametrization) {
                for (int i = 0; i < thisGradient.second.gradient.cols(); i++) {
                    params[i] -= learningRate_ * thisGradient.second.subParametrization->coeff(i) *
                                 thisGradient.second.gradient(0, i);
                }
            } else {
                for (int i = 0; i < thisGradient.second.gradient.cols(); i++) {
                    params[i] -= learningRate_ * thisGradient.second.gradient(0, i);
                }
            }
        }
    }

    void setBatchSize(int batchSize) {
        batchSize_ = batchSize;
    }
    void setLearningRate(double learningRate) {
        if (learningRate < 0) {
            throw std::runtime_error("Negative learning rate not allowed.");
        }
        learningRate_ = learningRate;
    }

    void setSubParametrization(const T* paramBlock, const std::vector<int> subparam) {
        GradientSetup& thisGradient = gradients_.at(reinterpret_cast<int64_t>(paramBlock));
        if (!thisGradient.subParametrization) {
            thisGradient.subParametrization = std::shared_ptr<Eigen::Matrix<T, 1, Eigen::Dynamic>>(
                new Eigen::Matrix<T, 1, Eigen::Dynamic>(1, thisGradient.gradient.cols()));
        }
        thisGradient.subParametrization->setOnes();
        for (int i = 0; i < subparam.size(); i++) {
            thisGradient.subParametrization->coeffRef(subparam[i]) = 0;
        }
    }

    void setParameterBlockConstant(const T* paramBlock) {
        GradientSetup& thisGradient = gradients_.at(reinterpret_cast<int64_t>(paramBlock));
        thisGradient.toOptimize = false;
    }

protected:
    int currentSampleIdx_;
    int batchSize_;
    double learningRate_;
    std::vector<ChainSetup> chainSetups_;
    std::map<int64_t, GradientSetup> gradients_;
};

} // namespace block_optimization

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace block_optimization {

template <typename T>
class Block {
public:
    Block(std::string name, std::string type, std::vector<int> paramCount, int inputDimensions, int outputDimensions)
            : name_(name), type_(type), paramCount_(paramCount), inputDimension_(inputDimensions),
              outputDimension_(outputDimensions), doOptimization_(true), internalMemorySet_(false) {
        if (!paramCount_.empty()) {
            std::vector<int>::iterator paramIt = paramCount_.begin();
            while (paramIt != paramCount_.end()) {
                if (*paramIt == 0) {
                    paramIt = paramCount_.erase(paramIt);
                } else {
                    ++paramIt;
                }
            }
        }
    }

    Block(std::string name, std::string type, int paramCount, int inputDimensions, int outputDimensions)
            : Block(name, type, std::vector<int>{paramCount}, inputDimensions, outputDimensions) {
        ;
    }


    virtual void process(
        T const* const* params, T const* input, T* ouput, T* jacobianWRTinput, T** jacobianWRTparams) const {
        this->computeOutput(params, input, ouput);
        this->computeJacobians(params, input, jacobianWRTparams, jacobianWRTinput);
    }

    virtual void computeOutput(T const* const* params, T const* input, T* ouput) const = 0;

    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const = 0;

    virtual std::vector<int> getParamCount() const {
        return paramCount_;
    }
    virtual int getParamBlockCount() const {
        return paramCount_.size();
    }
    virtual int getInputDimension() const {
        return inputDimension_;
    }
    virtual int getOutputDimension() const {
        return outputDimension_;
    }
    virtual bool doOptimization() const {
        return doOptimization_;
    }

    std::string getName() {
        return name_;
    }

    std::string getType() {
        return type_;
    }

    void enableOptimization() {
        doOptimization_ = true;
    }
    void disableOptimization() {
        doOptimization_ = false;
    }

    virtual void setInternalMemory() {
        params_.clear();
        for (int paramCount : paramCount_) {
            // init memory
            params_.push_back(std::shared_ptr<T>(new T[paramCount], std::default_delete<T[]>()));
            // set zero
            for (int i = 0; i < paramCount; i++) {
                params_.back().get()[i] = T(0.);
            }
        }
        internalMemorySet_ = true;
    }

    bool hasInternalMemory() {
        return internalMemorySet_;
    }

    virtual std::shared_ptr<T> getInternalMemory(int index) {
        return params_.at(index);
    }

    virtual std::vector<std::shared_ptr<T>> getInternalMemory() {
        return params_;
    }

protected:
    std::vector<int> paramCount_;
    int inputDimension_;
    int outputDimension_;
    bool doOptimization_;
    std::vector<std::shared_ptr<T>> params_;
    bool internalMemorySet_;

private:
    std::string name_;
    std::string type_;
};


template <typename T>
using BlockPtr = std::shared_ptr<Block<T>>;

} // namespace block_optimization

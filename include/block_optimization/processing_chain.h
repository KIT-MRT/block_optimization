#pragma once

#include "block.h"

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Core>

namespace block_optimization {

template <typename T>
using EigenMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<1>>;

template <typename T>
struct BlockSetup {
    std::shared_ptr<T> input_;
    std::shared_ptr<T> output_;

    std::shared_ptr<T> jacobiansWRTinput_;
    std::shared_ptr<EigenMap<T>> jacobiansWRTinputEigen_;

    std::vector<std::shared_ptr<T>> jacobiansWRTparams_;
    std::vector<std::shared_ptr<EigenMap<T>>> jacobiansWRTparamsEigen_;

    std::map<int, int> blockParamsToChainParamsMap;

    int paramBlockCount;
};

template <typename T>
class ProcessingChain : public std::list<std::pair<BlockPtr<T>, BlockSetup<T>>> {
public:
    ProcessingChain() : numResiduals_(0), numParamBlocks_(0) {
        ;
    }

    virtual int getNumResiduals() {
        if (this->empty()) {
            return 0;
        }
        if (!this->back().first) {
            throw std::runtime_error("NULL ptr block in chain");
        }
        return numResiduals_;
    }

    virtual int getNumInputs() {
        if (this->empty()) {
            return 0;
        }
        if (!this->front().first) {
            throw std::runtime_error("NULL ptr block in chain");
        }
        return this->front().first->getInputDimension();
    }

    virtual std::vector<int32_t> getParameterBlockSizes() {
        return chainParamBlockSizes_;
    }


    virtual void getParameterBlockSizes(std::vector<int32_t>& paramBlockSizes) {
        paramBlockSizes.resize(chainParamBlockSizes_.size());
        std::copy(chainParamBlockSizes_.begin(), chainParamBlockSizes_.end(), paramBlockSizes.begin());
    }

    virtual void getParameters(std::vector<T*>& parameters) {
        parameters.resize(chainParamBlockSizes_.size());
        typename ProcessingChain<T>::iterator blockIt = this->begin();
        int paramIdx = 0;
        while (blockIt != this->end()) {
            BlockPtr<T>& thisBlock = blockIt->first;
            int thisParamBlockCount = thisBlock->getParamBlockCount();
            if (thisBlock->hasInternalMemory()) {
                for (int i = 0; i < thisParamBlockCount; i++) {
                    parameters[paramIdx++] = thisBlock->getInternalMemory(i).get();
                }
            } else {
                for (int i = 0; i < thisParamBlockCount; i++) {
                    parameters[paramIdx++] = NULL;
                }
            }
        }
        ++blockIt;
    }

    virtual void appendBlock(BlockPtr<T> block) {
        if (!block) {
            throw std::runtime_error("Tried to append NULL block");
        }

        // check whether this is the first
        if (!this->empty()) {
            // if not so, check output-input dimension match
            if (this->back().first->getOutputDimension() != block->getInputDimension()) {
                throw std::runtime_error("Input and output dimension missmatch");
            }
        }

        BlockSetup<T> thisBlockSetup;
        thisBlockSetup.paramBlockCount = 0;

        int outputSize = block->getOutputDimension();
        int inputSize = block->getInputDimension();


        // check whether input exists. This might not be the case for data blocks
        // with optimized data
        if (!this->empty()) {
            if (inputSize > 0) {
                // connect output of last block to input of this block
                thisBlockSetup.input_ = this->back().second.output_;
            } else {
                throw std::runtime_error("Block " + block->getName() + " has to be connected to preceeding block " +
                                         this->back().first->getName());
            }
        } else {
            if (inputSize > 0) {
                // connect output of last block to input of this block
                thisBlockSetup.input_.reset(new T[inputSize], std::default_delete<T[]>());
            } else {
                // no input: empty data
                thisBlockSetup.input_.reset();
            }
        }

        // a block has to have outputs
        if (outputSize > 0) {
            thisBlockSetup.output_.reset(new T[outputSize], std::default_delete<T[]>());
        } else {
            throw std::runtime_error("block has no output");
        }

        // create the corresponding jacobians w.r.t. input
        if (inputSize > 0) {
            thisBlockSetup.jacobiansWRTinput_.reset(new T[outputSize * inputSize], std::default_delete<T[]>());
            thisBlockSetup.jacobiansWRTinputEigen_.reset(
                new EigenMap<T>(thisBlockSetup.jacobiansWRTinput_.get(), outputSize, inputSize));
        } else {
            thisBlockSetup.jacobiansWRTinput_.reset();
            thisBlockSetup.jacobiansWRTinputEigen_.reset();
        }


        // create memory for jacobians
        std::vector<int> paramCount = block->getParamCount();
        for (unsigned int i = 0; i < paramCount.size(); i++) {
            if (paramCount[i] > 0) {
                thisBlockSetup.jacobiansWRTparams_.push_back(
                    std::shared_ptr<T>(new T[outputSize * paramCount[i]], std::default_delete<T[]>()));
                thisBlockSetup.jacobiansWRTparamsEigen_.push_back(std::shared_ptr<EigenMap<T>>(
                    new EigenMap<T>(thisBlockSetup.jacobiansWRTparams_.back().get(), outputSize, paramCount[i])));
                thisBlockSetup.blockParamsToChainParamsMap[i] = numParamBlocks_;
                ++numParamBlocks_;
                ++thisBlockSetup.paramBlockCount;
                chainParamBlockSizes_.push_back(paramCount[i]);
            } else {
                thisBlockSetup.jacobiansWRTparams_.push_back(std::shared_ptr<T>(nullptr));
                thisBlockSetup.jacobiansWRTparamsEigen_.push_back(std::shared_ptr<EigenMap<T>>(nullptr));
            }
        }

        numResiduals_ = outputSize;

        // add the block to the list
        this->push_back(std::make_pair(block, thisBlockSetup));
    }

    virtual void process(T const* const* parameters, T const* input, T* output, T** jacobians, T* jacobianWRTinput) {
        // check for jacobian pointer
        bool computeJacobians = bool(jacobians);
        // forward pass
        this->forwardChain(parameters, input, output, computeJacobians);
        // backward pass only if desired
        if (computeJacobians) {
            this->backwardChain(jacobians, jacobianWRTinput);
        }
    }


    void forwardChain(T const* const* parameters, T const* input, T* output, bool computeJacobians = true) {
        bool first = true;
        for (std::pair<BlockPtr<T>, BlockSetup<T>>& block : *this) {
            BlockPtr<T> thisBlockPtr = block.first;
            BlockSetup<T>& thisSetup = block.second;

            T const** thisBlocksParameters = new T const*[thisSetup.paramBlockCount];
            for (int i = 0; i < thisSetup.paramBlockCount; i++) {
                thisBlocksParameters[i] = parameters[thisSetup.blockParamsToChainParamsMap.at(i)];
            }

            T const* inputData = NULL;
            if (input && first) {
                if (!thisBlockPtr->getInputDimension()) {
                    throw std::runtime_error("Input for block " + thisBlockPtr->getName() +
                                             "supplied, but none expected");
                }
                inputData = input;
            } else {
                inputData = thisSetup.input_.get();
            }

            //      std::cout << thisBlockPtr->getName() << " in: " <<std::endl;
            //      for(int i = 0; i< thisBlockPtr->getInputDimension(); i++) {
            //        std::cout << inputData[i] << " ";
            //      }
            //      std::cout << std::endl;

            if (computeJacobians) {
                T** thisBlocksParamJacobians = new T*[thisSetup.jacobiansWRTparams_.size()];
                for (unsigned int i = 0; i < thisSetup.jacobiansWRTparams_.size(); i++) {
                    thisBlocksParamJacobians[i] = thisSetup.jacobiansWRTparams_[i].get();
                }
                thisBlockPtr->process(thisBlocksParameters,
                                      inputData,
                                      thisSetup.output_.get(),
                                      thisSetup.jacobiansWRTinput_.get(),
                                      thisBlocksParamJacobians);
                delete[] thisBlocksParamJacobians;
                thisBlocksParamJacobians = NULL;
            } else {
                thisBlockPtr->process(thisBlocksParameters, inputData, thisSetup.output_.get(), NULL, NULL);
            }
            delete[] thisBlocksParameters;
            thisBlocksParameters = NULL;
            first = false;

            //      std::cout << thisBlockPtr->getName() << " out: " <<std::endl;
            //      for(int i = 0; i< thisBlockPtr->getOutputDimension(); i++) {
            //        std::cout << thisSetup.output_.get()[i] << " ";
            //      }
            //      std::cout << std::endl;
        }
        // copy residuals
        for (int i = 0; i < numResiduals_; i++) {
            output[i] = this->back().second.output_.get()[i];
        }
    }


    void backwardChain(T** jacobians, T* jacobianWRTinput) {
        // if jacobians are not desired, stop here
        if (!jacobians && !jacobianWRTinput) {
            return;
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bwdJacobian =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(numResiduals_, numResiduals_);
        typename ProcessingChain::reverse_iterator blockIt = this->rbegin();
        while (blockIt != this->rend()) {
            BlockPtr<T> thisBlockPtr = blockIt->first;
            BlockSetup<T>& thisSetup = blockIt->second;

            for (int i = 0; i < thisSetup.paramBlockCount; i++) {
                // only compute if desired
                int chainParamIndex = thisSetup.blockParamsToChainParamsMap.at(i);
                if (jacobians[chainParamIndex]) {
                    if (!thisSetup.jacobiansWRTparamsEigen_[i]) {
                        throw std::runtime_error("Jacobian of block " + thisBlockPtr->getName() +
                                                 " desired but not availble");
                    }

                    EigenMap<T> thisJacobi(jacobians[chainParamIndex], numResiduals_, thisBlockPtr->getParamCount()[i]);
                    thisJacobi = bwdJacobian * (*thisSetup.jacobiansWRTparamsEigen_[i]);

                    //          std::cout << thisBlockPtr->getName() << std::endl <<
                    //          *thisSetup.jacobiansWRTparamsEigen_[i] << std::endl; std::cout <<
                    //          thisBlockPtr->getName() << std::endl << thisJacobi << std::endl;
                }
            }

            if (thisSetup.jacobiansWRTinputEigen_) {
                // compute the backward property of the chain rule
                bwdJacobian = bwdJacobian * (*thisSetup.jacobiansWRTinputEigen_);
            } else if (thisBlockPtr != this->begin()->first) {
                throw std::runtime_error("Block " + thisBlockPtr->getName() + "does not supply backward jaconbian");
            }


            ++blockIt;
        }

        if (jacobianWRTinput) {
            // copy input jacobian
            for (int i = 0; i < bwdJacobian.rows(); i++) {
                for (int j = 0; j < bwdJacobian.cols(); j++) {
                    jacobianWRTinput[i * bwdJacobian.rows() + j] = bwdJacobian(i, j);
                }
            }
        }
    }

protected:
    std::vector<int> chainParamBlockSizes_;
    int numResiduals_;
    int numParamBlocks_;
};

template <typename T>
using ProcessingChainPtr = std::shared_ptr<ProcessingChain<T>>;

} // namespace block_optimization

#include "ceres_cost_function.h"

namespace block_optimization {

CeresCostFunctionAdapter::CeresCostFunctionAdapter() : chain_(nullptr) {
}

CeresCostFunctionAdapter::CeresCostFunctionAdapter(ProcessingChainPtr<double> chain) : CeresCostFunctionAdapter() {
    this->makeCostFunction(chain);
}

bool CeresCostFunctionAdapter::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    if (!chain_) {
        return false;
    }

    chain_->process(parameters, NULL, residuals, jacobians, NULL);

    return true;
}

void CeresCostFunctionAdapter::makeCostFunction(ProcessingChainPtr<double> chain) {
    this->chain_ = chain;
    if (!chain_) {
        throw std::runtime_error("Tried to create cost function from empty chain pointer");
    }

    int numResiduals = chain_->getNumResiduals();
    if (!numResiduals) {
        throw std::runtime_error("Chain has to compute a residual");
    }
    std::vector<int32_t> parameterBlockSizes;
    chain_->getParameterBlockSizes(parameterBlockSizes);

    if (parameterBlockSizes.empty()) {
        throw std::runtime_error("empty parameters");
    }

    this->mutable_parameter_block_sizes()->resize(parameterBlockSizes.size());
    std::copy(parameterBlockSizes.begin(), parameterBlockSizes.end(), this->mutable_parameter_block_sizes()->begin());

    this->set_num_residuals(numResiduals);
}

} // namespace block_optimization

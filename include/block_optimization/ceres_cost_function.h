#pragma once

#include <ceres/cost_function.h>
#include "processing_chain.h"

namespace block_optimization {

class CeresCostFunctionAdapter : public ceres::CostFunction {
public:
    CeresCostFunctionAdapter();
    CeresCostFunctionAdapter(ProcessingChainPtr<double> chain);

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

    virtual void makeCostFunction(ProcessingChainPtr<double> chain);

protected:
    ProcessingChainPtr<double> chain_;
};

} // namespace block_optimization

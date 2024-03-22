#include "ceres_solver.h"
#include "ceres_cost_function.h"


namespace block_optimization {

void CeresSolver::solve(Problem<double>& problem) {
    if (!problem_) {
        throw std::runtime_error("Problem not initialized");
    }
}

void CeresSolver::makeCeresProblem(Problem<double>& problem) {
}

void CeresSolver::addChain(ProcessingChainPtr<double> chain,
                           std::vector<double*> params,
                           std::vector<double*>& data,
                           std::vector<double*>& target,
                           ceres::LossFunction* lossFcn) {
    if (data.size() != target.size()) {
        throw std::runtime_error("data and target count has to be same");
    }

    if (!problem_) {
        problem_ = std::make_shared<ceres::Problem>();
    }

    bool dataConst = !chain->front().first->doOptimization();
    bool targetConst = !chain->back().first->doOptimization();


    for (unsigned int i = 0; i < data.size(); i++) {
        if (!data[i] || !target[i]) {
            continue;
        }
        ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);
        params.front() = data[i];
        params.back() = target[i];

        problem_->AddResidualBlock(costFcn, lossFcn, params);
        if (dataConst) {
            problem_->SetParameterBlockConstant(params.front());
        } else {
            problem_->SetParameterBlockVariable(params.front());
        }
        if (targetConst) {
            problem_->SetParameterBlockConstant(params.back());
        } else {
            problem_->SetParameterBlockVariable(params.back());
        }
    }

    ProcessingChain<double>::iterator elemIt = chain->begin();
    unsigned int paramIdx = 0;
    while (elemIt != chain->end()) {
        if (elemIt->first->getParamBlockCount() != 0) {
            if (!elemIt->first->doOptimization()) {
                for (int i = 0; i < elemIt->first->getParamBlockCount(); i++) {
                    if (paramIdx != 0 && paramIdx != params.size() - 1) {
                        problem_->SetParameterBlockConstant(params[paramIdx]);
                    }
                    ++paramIdx;
                }
            } else {
                for (int i = 0; i < elemIt->first->getParamBlockCount(); i++) {
                    if (paramIdx != 0 && paramIdx != params.size() - 1) {
                        problem_->SetParameterBlockVariable(params[paramIdx]);
                    }
                    ++paramIdx;
                }
            }
        }
        ++elemIt;
    }
}


} // namespace block_optimization

#pragma once

#include "problem.h"
#include "solver.h"

#include <ceres/loss_function.h>
#include <ceres/problem.h>

namespace block_optimization {

class CeresSolver : public Solver<double> {
public:
    virtual void solve(Problem<double>& problem);
    virtual void makeCeresProblem(Problem<double>& problem);

    virtual void addChain(ProcessingChainPtr<double>,
                          std::vector<double*> params,
                          std::vector<double*>& data,
                          std::vector<double*>& target,
                          ceres::LossFunction* lossFcn = NULL);

    std::shared_ptr<ceres::Problem> problem_;
    std::shared_ptr<ceres::Solver> solver_;
};

} // namespace block_optimization

#pragma once

#include "problem.h"

namespace block_optimization {

template <typename T>
class Solver {
public:
    virtual void solve(Problem<T>& problem) = 0;

protected:
private:
};

} // namespace block_optimization

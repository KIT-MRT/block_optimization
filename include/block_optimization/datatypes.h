#pragma once

#include <memory>

#include <Eigen/Core>

namespace block_optimization {

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using MatrixPtr = std::shared_ptr<Matrix<T>>;
template <typename T>
using MatrixConstPtr = std::shared_ptr<const Matrix<T>>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorPtr = std::shared_ptr<Vector<T>>;
template <typename T>
using VectorConstPtr = std::shared_ptr<const Vector<T>>;

} // namespace block_optimization

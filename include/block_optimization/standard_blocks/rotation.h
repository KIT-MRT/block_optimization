#pragma once

//#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../block.h"


namespace block_optimization {


template <typename T>
class Rotation : public Block<T> {
    using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, 3, 1>, 0, Eigen::InnerStride<1>>;

public:
    Rotation(std::string name) : Block<T>(name, "Rotation", 3, 3, 3) {
        ;
    }
    virtual void computeOutput(T const* const* params, T const* input, T* output) const {
        Eigen::Matrix<T, 3, 1> inVec;
        for (int i = 0; i < 3; i++) {
            inVec(i) = input[i];
        }
        EigenVectorMap outVec(output);
        Eigen::Matrix<T, 3, 3> rotMat;
        this->getRotationMatrix(params[0], rotMat);
        outVec = rotMat * inVec;
    }
    virtual void computeJacobians(T const* const* params,
                                  T const* input,
                                  T** jacobianWRTparams,
                                  T* jacobianWRTinput) const {
        // compute jacobians w.r.t. params
        if (jacobianWRTparams && jacobianWRTparams[0]) {
            // import input to vector
            Eigen::Matrix<T, 3, 1> in;
            for (int i = 0; i < 3; i++) {
                in(i) = input[i];
            }

            // this will be the final output
            Eigen::Matrix<T, 3, 3> fullJacobian;

            // read parameters
            T rx = params[0][0];
            T ry = params[0][1];
            T rz = params[0][2];

            // compute angle
            T theta = std::sqrt(std::pow(rx, 2) + std::pow(ry, 2) + std::pow(rz, 2));
            T itheta = theta > std::numeric_limits<T>::epsilon() ? T(1.) / theta : T(0.);
            // normalize axis
            rx *= itheta;
            ry *= itheta;
            rz *= itheta;

            // helpers
            T s = std::sin(theta);
            T c = std::cos(theta);
            T c1 = 1 - c;

            // matrix form of cross product
            Eigen::Matrix<T, 3, 3> kx;
            kx << 0, -rz, ry, rz, 0, -rx, -ry, rx, 0;

            // derivatives of the above
            Eigen::Matrix<T, 3, 3> dkxdr[3];
            dkxdr[0] << 0, 0, 0, 0, 0, -1, 0, 1, 0;
            dkxdr[1] << 0, 0, 1, 0, 0, 0, -1, 0, 0;
            dkxdr[2] << 0, -1, 0, 1, 0, 0, 0, 0, 0;

            // outer product of axis with itself
            Eigen::Matrix<T, 3, 3> rrt;
            rrt << rx * rx, rx * ry, rx * rz, ry * rx, ry * ry, ry * rz, rz * rx, rz * ry, rz * rz;


            // derivatives of the above
            Eigen::Matrix<T, 3, 3> drrt[3];
            drrt[0] << 2 * rx, ry, rz, ry, 0, 0, rz, 0, 0;
            drrt[1] << 0, rx, 0, rx, 2 * ry, rz, 0, rz, 0;
            drrt[2] << 0, 0, rx, 0, 0, ry, rx, ry, 2 * rz;

            // compute the derivative matrices of the formula
            // R = I*cos(theta) + r_x*sin(theta) + rrT(1-cos(theta)) (1)
            // note: the r_[] have all been normalized, that's why I have
            // few normalizations by theta in here
            for (int paramI = 0; paramI < 3; paramI++) {
                Eigen::Matrix<T, 3, 3> jacobian;
                T rn = params[0][paramI] * itheta;
                // derivative of the first term of (1)
                jacobian = -Eigen::Matrix<T, 3, 3>::Identity() * s * rn;
                // add derivative of second term
                jacobian += (c + s * itheta) * rn * kx;
                if (theta > std::numeric_limits<T>::epsilon()) {
                    jacobian += s * itheta * dkxdr[paramI];
                } else { // small angle approximation
                    jacobian += dkxdr[paramI];
                }
                // add derivative of third term
                jacobian += drrt[paramI] * c1 * itheta + (s - 2 * c1 * itheta) * rn * rrt;
                // multiply with vector for result
                fullJacobian.col(paramI) = jacobian * in;
            }

            // copy values
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    jacobianWRTparams[0][i * 3 + j] = fullJacobian(i, j);
                }
            }
        }

        // compute jacobian w.r.t. input
        if (jacobianWRTinput) {
            // todo: this might also be solved with a matrix map or so
            Eigen::Matrix<T, 3, 3> rotMat;
            this->getRotationMatrix(params[0], rotMat);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    jacobianWRTinput[i * 3 + j] = rotMat(i, j);
                }
            }
        }
    }

    virtual void getRotationMatrix(T const* params, Eigen::Matrix<T, 3, 3>& R) const {
        Eigen::Matrix<T, 3, 1> axis;
        for (int i = 0; i < 3; i++) {
            axis(i) = params[i];
        }
        T angle = axis.norm();
        axis = axis * T(1.) / std::max(std::numeric_limits<T>::epsilon(), angle);
        R = Eigen::AngleAxis<T>(angle, axis);
    }

protected:
};
} // namespace block_optimization

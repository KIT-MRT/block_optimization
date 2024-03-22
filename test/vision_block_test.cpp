// google test docs
// wiki page: https://code.google.com/p/googletest/w/list
// primer: https://code.google.com/p/googletest/wiki/V1_7_Primer
// FAQ: https://code.google.com/p/googletest/wiki/FAQ
// advanced guide: https://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide
// samples: https://code.google.com/p/googletest/wiki/V1_7_Samples
//
// List of some basic tests fuctions:
// Fatal assertion                      Nonfatal assertion                   Verifies / Description
//-------------------------------------------------------------------------------------------------------------------------------------------------------
// ASSERT_EQ(expected, actual);         EXPECT_EQ(expected, actual);         expected == actual
// ASSERT_NE(val1, val2);               EXPECT_NE(val1, val2);               val1 != val2
// ASSERT_LT(val1, val2);               EXPECT_LT(val1, val2);               val1 < val2
// ASSERT_LE(val1, val2);               EXPECT_LE(val1, val2);               val1 <= val2
// ASSERT_GT(val1, val2);               EXPECT_GT(val1, val2);               val1 > val2
// ASSERT_GE(val1, val2);               EXPECT_GE(val1, val2);               val1 >= val2
//
// ASSERT_FLOAT_EQ(expected, actual);   EXPECT_FLOAT_EQ(expected, actual);   the two float values are almost equal (4
// ULPs) ASSERT_DOUBLE_EQ(expected, actual);  EXPECT_DOUBLE_EQ(expected, actual);  the two double values are almost
// equal (4 ULPs) ASSERT_NEAR(val1, val2, abs_error);  EXPECT_NEAR(val1, val2, abs_error);  the difference between val1
// and val2 doesn't exceed the given absolute error
//
// Note: more information about ULPs can be found here:
// http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
//
// Example of two unit test:
// TEST(Math, Add) {
//    ASSERT_EQ(10, 5+ 5);
//}
//
// TEST(Math, Float) {
//	  ASSERT_FLOAT_EQ((10.0f + 2.0f) * 3.0f, 10.0f * 3.0f + 2.0f * 3.0f)
//}
//=======================================================================================================================================================
#include "gtest/gtest.h"


#include <ceres/ceres.h>
#include "block_optimization.h"
#include "ceres_cost_function.h"
#include "sgd_solver.h"
#include "standard_blocks.h"
#include "vision_blocks.h"


TEST(VisionBlocks, PinholeProjection) {
    const int numDataPoints = 50;
    const double thresh = 1e-6;
    // prepare data and target values
    double targetParams[4];
    targetParams[0] = 100;
    targetParams[1] = 50;
    targetParams[2] = 100;
    targetParams[3] = 25;

    std::vector<double[3]> data(numDataPoints);
    std::vector<double[2]> target(numDataPoints);
    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand() % 1000) / 100. - 5.;
        data[i][1] = double(rand() % 1000) / 100. - 5.;
        data[i][2] = double(rand() % 1000) / 100.;

        target[i][0] = data[i][0] / data[i][2] * targetParams[0] + targetParams[1];
        target[i][1] = data[i][1] / data[i][2] * targetParams[2] + targetParams[3];
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> homogenization(
        new block_optimization::VectorHomogenization<double>("homogen"));
    block_optimization::BlockPtr<double> projection(new block_optimization::PinholeProjection<double>("project"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 2>("e"));

    projection->setInternalMemory();

    std::shared_ptr<double> params = projection->getInternalMemory(0);

    ASSERT_NE(reinterpret_cast<int64_t>(params.get()), 0);

    params.get()[0] = 1.;
    params.get()[1] = 0.;
    params.get()[2] = 1.;
    params.get()[3] = 0.;

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(homogenization);
    chain->appendBlock(projection);
    chain->appendBlock(residuumBlock);

    std::vector<double*> paramVec(3);
    paramVec[1] = params.get();

    ceres::Problem problem;

    for (int i = 0; i < numDataPoints; i++) {
        ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);

        paramVec[0] = data[i];
        paramVec[2] = target[i];

        problem.AddResidualBlock(costFcn, NULL, paramVec);
        problem.SetParameterBlockConstant(paramVec[0]);
        problem.SetParameterBlockConstant(paramVec[2]);
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(targetParams[i], params.get()[i], 1e-6);
    }
}

TEST(VisionBlocks, PinholeDistortion) {
    const int numDataPoints = 100;
    const double thresh = 1e-6;
    // prepare data and target values
    double targetParams[9];
    targetParams[0] = 100;
    targetParams[1] = 50;
    targetParams[2] = 100;
    targetParams[3] = 25;

    targetParams[4] = -0.2;
    targetParams[5] = 0.1;
    targetParams[6] = 0.;
    targetParams[7] = 0.;
    targetParams[8] = 0.;


    std::vector<double[3]> data(numDataPoints);
    std::vector<double[2]> target(numDataPoints);

    double k1 = targetParams[4];
    double k2 = targetParams[5];
    double k3 = targetParams[8];


    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand() % 1000) / 100. - 5.;
        data[i][1] = double(rand() % 1000) / 100. - 5.;
        data[i][2] = 0;
        while (data[i][2] < 1e-2) {
            data[i][2] = double(rand() % 1000) / 10.;
        }

        double u = data[i][0] / data[i][2];
        double v = data[i][1] / data[i][2];
        double rsq = u * u + v * v;

        double distort = 1. + rsq * (k1 + rsq * (k2 + rsq * k3));

        target[i][0] = u * distort * targetParams[0] + targetParams[1];
        target[i][1] = v * distort * targetParams[2] + targetParams[3];
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> homogenization(
        new block_optimization::VectorHomogenization<double>("homogen"));
    block_optimization::BlockPtr<double> distortion(new block_optimization::PolyDistortion<double>("distort"));
    block_optimization::BlockPtr<double> projection(new block_optimization::PinholeProjection<double>("project"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 2>("e"));

    projection->setInternalMemory();
    distortion->setInternalMemory();

    std::shared_ptr<double> projectParams = projection->getInternalMemory(0);
    std::shared_ptr<double> distortionParams = distortion->getInternalMemory(0);

    ASSERT_NE(reinterpret_cast<int64_t>(projectParams.get()), 0);
    ASSERT_NE(reinterpret_cast<int64_t>(distortionParams.get()), 0);

    projectParams.get()[0] = 100.;
    projectParams.get()[1] = 50.;
    projectParams.get()[2] = 100.;
    projectParams.get()[3] = 25.;

    distortionParams.get()[0] = 0;
    distortionParams.get()[1] = 0.;
    distortionParams.get()[2] = 0.;
    distortionParams.get()[3] = 0.;
    distortionParams.get()[4] = 0.;

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(homogenization);
    chain->appendBlock(distortion);
    chain->appendBlock(projection);
    chain->appendBlock(residuumBlock);

    std::vector<double*> paramVec(4);
    paramVec[1] = distortionParams.get();
    paramVec[2] = projectParams.get();

    ceres::Problem problem;

    for (int i = 0; i < numDataPoints; i++) {
        ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);

        paramVec[0] = data[i];
        paramVec[3] = target[i];

        problem.AddResidualBlock(costFcn, NULL, paramVec);
        problem.SetParameterBlockConstant(paramVec[0]);
        problem.SetParameterBlockConstant(paramVec[2]);
        problem.SetParameterBlockConstant(paramVec[3]);
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(targetParams[i], projectParams.get()[i], 1e-7);
    }
    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(targetParams[i + 4], distortionParams.get()[i], 1e-7);
    }
}

TEST(VisionBlocks, VisualOdometry) {

    const int numDataPoints = 100;
    const double thresh = 1e-6;
    // prepare data and target values
    double targetParams[6];
    targetParams[0] = 0.02;
    targetParams[1] = -0.01;
    targetParams[2] = 0.05;

    targetParams[3] = 0.2;
    targetParams[4] = -0.1;
    targetParams[5] = 0.5;


    double camParams[9];
    camParams[0] = 100;
    camParams[1] = 50;
    camParams[2] = 100;
    camParams[3] = 25;

    camParams[4] = -0.2;
    camParams[5] = 0.1;
    camParams[6] = 0.;
    camParams[7] = 0.;
    camParams[8] = 0.;

    std::vector<double[3]> data(numDataPoints);
    std::vector<double[2]> target(numDataPoints);

    double k1 = camParams[4];
    double k2 = camParams[5];
    double k3 = camParams[8];

    Eigen::Vector3d axis;
    axis(0) = targetParams[0];
    axis(1) = targetParams[1];
    axis(2) = targetParams[2];

    double angle = axis.norm();
    if (angle > 1e-7) {
        axis.normalize();
    }

    Eigen::Matrix3d R;
    R = Eigen::AngleAxis<double>(angle, axis);


    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand() % 1000) / 100. - 5.;
        data[i][1] = double(rand() % 1000) / 100. - 5.;
        data[i][2] = 0;
        while (data[i][2] < 1e-2) {
            data[i][2] = double(rand() % 1000) / 10.;
        }


        Eigen::Vector3d vec;
        vec << data[i][0], data[i][1], data[i][2];

        vec = R * vec;

        vec(0) += targetParams[3];
        vec(1) += targetParams[4];
        vec(2) += targetParams[5];


        double u = vec(0) / vec(2);
        double v = vec(1) / vec(2);
        double rsq = u * u + v * v;

        double distort = 1. + rsq * (k1 + rsq * (k2 + rsq * k3));

        target[i][0] = u * distort * camParams[0] + camParams[1];
        target[i][1] = v * distort * camParams[2] + camParams[3];
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> transformBlock(new block_optimization::Transform6DOF<double>("trafo"));
    block_optimization::BlockPtr<double> homogenization(
        new block_optimization::VectorHomogenization<double>("homogen"));
    block_optimization::BlockPtr<double> distortion(new block_optimization::PolyDistortion<double>("distort"));
    block_optimization::BlockPtr<double> projection(new block_optimization::PinholeProjection<double>("project"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 2>("e"));

    transformBlock->setInternalMemory();

    std::shared_ptr<double> rotParams = transformBlock->getInternalMemory(0);
    std::shared_ptr<double> transParams = transformBlock->getInternalMemory(1);

    ASSERT_NE(reinterpret_cast<int64_t>(transParams.get()), 0);
    ASSERT_NE(reinterpret_cast<int64_t>(rotParams.get()), 0);

    projection->setInternalMemory();
    distortion->setInternalMemory();

    std::shared_ptr<double> projectParams = projection->getInternalMemory(0);
    std::shared_ptr<double> distortionParams = distortion->getInternalMemory(0);

    ASSERT_NE(reinterpret_cast<int64_t>(projectParams.get()), 0);
    ASSERT_NE(reinterpret_cast<int64_t>(distortionParams.get()), 0);


    projectParams.get()[0] = camParams[0];
    projectParams.get()[1] = camParams[1];
    projectParams.get()[2] = camParams[2];
    projectParams.get()[3] = camParams[3];

    distortionParams.get()[0] = camParams[4];
    distortionParams.get()[1] = camParams[5];
    distortionParams.get()[2] = camParams[6];
    distortionParams.get()[3] = camParams[7];
    distortionParams.get()[4] = camParams[8];

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(transformBlock);
    chain->appendBlock(homogenization);
    chain->appendBlock(distortion);
    chain->appendBlock(projection);
    chain->appendBlock(residuumBlock);

    std::vector<double*> paramVec(6);
    paramVec[1] = rotParams.get();
    paramVec[2] = transParams.get();
    paramVec[3] = distortionParams.get();
    paramVec[4] = projectParams.get();

    ceres::Problem problem;

    for (int i = 0; i < numDataPoints; i++) {
        ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);

        paramVec[0] = data[i];
        paramVec[5] = target[i];

        problem.AddResidualBlock(costFcn, NULL, paramVec);
        problem.SetParameterBlockConstant(paramVec[0]);
        problem.SetParameterBlockConstant(paramVec[3]);
        problem.SetParameterBlockConstant(paramVec[4]);
        problem.SetParameterBlockConstant(paramVec[5]);
    }


    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);


    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[i], rotParams.get()[i], 1e-7);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[i + 3], transParams.get()[i], 1e-7);
    }
}

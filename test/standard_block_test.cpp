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

TEST(StandardBlocks, VectorAdd) {
    const int numDataPoints = 15;

    // prepare data and target values
    double targetParams[3];
    targetParams[0] = 2;
    targetParams[1] = -4;
    targetParams[2] = 0;

    std::vector<double[3]> data(numDataPoints);
    std::vector<double[3]> target(numDataPoints);
    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand());
        data[i][1] = double(rand());
        data[i][2] = double(rand());

        target[i][0] = data[i][0] + targetParams[0];
        target[i][1] = data[i][1] + targetParams[1];
        target[i][2] = data[i][2] + targetParams[2];
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> addBlock(new block_optimization::VectorAdd<double, 3>("add"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 3>("e"));


    addBlock->setInternalMemory();

    std::shared_ptr<double> params = addBlock->getInternalMemory(0);
    ASSERT_NE(reinterpret_cast<int64_t>(params.get()), 0);

    params.get()[0] = double(rand());
    params.get()[1] = double(rand());
    params.get()[2] = double(rand());

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(addBlock);
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

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[i], params.get()[i], 1e-7);
    }
}

TEST(StandardBlocks, Rotation) {
    const int numDataPoints = 15;

    // prepare data and target values
    double targetParams[3];
    targetParams[0] = 0.3;
    targetParams[1] = -0.1;
    targetParams[2] = 1.3;

    Eigen::Vector3d axis;
    axis(0) = targetParams[0];
    axis(1) = targetParams[1];
    axis(2) = targetParams[2];

    double angle = axis.norm();
    axis.normalize();

    Eigen::Matrix3d R;
    R = Eigen::AngleAxis<double>(angle, axis);

    std::vector<double[3]> data(numDataPoints);
    std::vector<double[3]> target(numDataPoints);
    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand() % 1000) / 100.;
        data[i][1] = double(rand() % 1000) / 100.;
        data[i][2] = double(rand() % 1000) / 100.;

        Eigen::Vector3d vec;
        vec << data[i][0], data[i][1], data[i][2];

        vec = R * vec;

        target[i][0] = vec(0);
        target[i][1] = vec(1);
        target[i][2] = vec(2);
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> rotationBlock(new block_optimization::Rotation<double>("rotate"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 3>("e"));


    rotationBlock->setInternalMemory();

    std::shared_ptr<double> params = rotationBlock->getInternalMemory(0);
    ASSERT_NE(reinterpret_cast<int64_t>(params.get()), 0);

    params.get()[0] = 0.;
    params.get()[1] = 0.;
    params.get()[2] = 0.;

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(rotationBlock);
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

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[i], params.get()[i], 1e-7);
    }
}

TEST(StandardBlocks, Transform6DOF) {

    const double thresh = 1e-6;

    const int numDataPoints = 15;

    // prepare data and target values
    double targetParams[6];
    targetParams[0] = 0.3;
    targetParams[1] = -0.1;
    targetParams[2] = 1.3;

    targetParams[3] = 5;
    targetParams[4] = -1;
    targetParams[5] = 1;


    Eigen::Vector3d axis;
    axis(0) = targetParams[0];
    axis(1) = targetParams[1];
    axis(2) = targetParams[2];

    double angle = axis.norm();
    axis.normalize();

    Eigen::Matrix3d R;
    R = Eigen::AngleAxis<double>(angle, axis);

    std::vector<double[3]> data(numDataPoints);
    std::vector<double[3]> target(numDataPoints);
    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand() % 1000) / 100.;
        data[i][1] = double(rand() % 1000) / 100.;
        data[i][2] = double(rand() % 1000) / 100.;

        Eigen::Vector3d vec;
        vec << data[i][0], data[i][1], data[i][2];

        vec = R * vec;

        target[i][0] = vec(0) + targetParams[3];
        target[i][1] = vec(1) + targetParams[4];
        target[i][2] = vec(2) + targetParams[5];
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> transformBlock(new block_optimization::Transform6DOF<double>("trafo"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 3>("e"));


    transformBlock->setInternalMemory();

    std::shared_ptr<double> rotParams = transformBlock->getInternalMemory(0);
    std::shared_ptr<double> transParams = transformBlock->getInternalMemory(1);

    ASSERT_NE(reinterpret_cast<int64_t>(rotParams.get()), 0);
    ASSERT_NE(reinterpret_cast<int64_t>(transParams.get()), 0);

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(transformBlock);
    chain->appendBlock(residuumBlock);


    std::vector<double*> paramVec(4);
    paramVec[1] = rotParams.get();
    paramVec[2] = transParams.get();

    ceres::Problem problem;

    for (int i = 0; i < numDataPoints; i++) {
        ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);

        paramVec[0] = data[i];
        paramVec[3] = target[i];

        problem.AddResidualBlock(costFcn, NULL, paramVec);
        problem.SetParameterBlockConstant(paramVec[0]);
        problem.SetParameterBlockConstant(paramVec[3]);
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[i], rotParams.get()[i], 1e-6);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[3 + i], transParams.get()[i], 1e-6);
    }
}

TEST(StandardBlocks, ChainedTransforms) {

    const double thresh = 1e-6;

    const int numDataPoints = 15;

    // prepare data and target values
    double targetParams[6];
    targetParams[0] = 0.3;
    targetParams[1] = -0.1;
    targetParams[2] = 1.3;

    targetParams[3] = 5;
    targetParams[4] = -1;
    targetParams[5] = 1;

    Eigen::Vector3d axis;
    axis(0) = targetParams[0];
    axis(1) = targetParams[1];
    axis(2) = targetParams[2];

    double angle = axis.norm();
    axis.normalize();

    Eigen::Matrix3d R;
    R = Eigen::AngleAxis<double>(angle, axis);

    std::vector<double[3]> data(numDataPoints);
    std::vector<double[3]> target(numDataPoints);
    for (int i = 0; i < numDataPoints; i++) {
        data[i][0] = double(rand() % 1000) / 100.;
        data[i][1] = double(rand() % 1000) / 100.;
        data[i][2] = double(rand() % 1000) / 100.;

        Eigen::Vector3d vec;
        vec << data[i][0], data[i][1], data[i][2];

        vec(0) += targetParams[3];
        vec(1) += targetParams[4];
        vec(2) += targetParams[5];


        vec = R * vec;

        target[i][0] = vec(0); // + targetParams[3];
        target[i][1] = vec(1); // + targetParams[4];
        target[i][2] = vec(2); // + targetParams[5];

        //    target[i][0] = vec(0) + targetParams[3];
        //    target[i][1] = vec(1) + targetParams[4];
        //    target[i][2] = vec(2) + targetParams[5];
    }

    block_optimization::BlockPtr<double> dataBlock(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> rotationBlock(new block_optimization::Rotation<double>("rotate"));
    block_optimization::BlockPtr<double> addBlock(new block_optimization::VectorAdd<double, 3>("add"));
    block_optimization::BlockPtr<double> residuumBlock(new block_optimization::LinearError<double, 3>("e"));


    addBlock->setInternalMemory();
    rotationBlock->setInternalMemory();

    std::shared_ptr<double> rotParams = rotationBlock->getInternalMemory(0);
    std::shared_ptr<double> transParams = addBlock->getInternalMemory(0);

    ASSERT_NE(reinterpret_cast<int64_t>(rotParams.get()), 0);
    ASSERT_NE(reinterpret_cast<int64_t>(transParams.get()), 0);

    block_optimization::ProcessingChainPtr<double> chain(new block_optimization::ProcessingChain<double>());

    chain->appendBlock(dataBlock);
    chain->appendBlock(addBlock);
    chain->appendBlock(rotationBlock);
    chain->appendBlock(residuumBlock);


    std::vector<double*> paramVec(4);
    paramVec[2] = rotParams.get();
    paramVec[1] = transParams.get();

    ceres::Problem problem;

    for (int i = 0; i < numDataPoints; i++) {
        ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);

        paramVec[0] = data[i];
        paramVec[3] = target[i];

        problem.AddResidualBlock(costFcn, NULL, paramVec);
        problem.SetParameterBlockConstant(paramVec[0]);
        problem.SetParameterBlockConstant(paramVec[3]);
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[i], rotParams.get()[i], 1e-6);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(targetParams[3 + i], transParams.get()[i], 1e-6);
    }
}

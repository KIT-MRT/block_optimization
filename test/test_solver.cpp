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
#include "line_blocks.h"
#include "sgd_solver.h"


// A google test function (uncomment the next function, add code and
// change the names TestGroupName and TestName)
TEST(Solver, Ceres) {
    double targetM = 10.;
    double targetB = 5.;

    std::shared_ptr<line_test::DataBlock<double>> data(new line_test::DataBlock<double>());
    std::shared_ptr<line_test::AdditiveConstant<double>> add(new line_test::AdditiveConstant<double>());
    std::shared_ptr<line_test::MultiplicativeConstant<double>> multi(new line_test::MultiplicativeConstant<double>());
    std::shared_ptr<line_test::LinearError<double>> error(new line_test::LinearError<double>());


    std::shared_ptr<block_optimization::ProcessingChain<double>> chain(
        new block_optimization::ProcessingChain<double>());
    chain->appendBlock(data);
    chain->appendBlock(multi);
    chain->appendBlock(add);
    chain->appendBlock(error);

    ceres::CostFunction* costFcn = new block_optimization::CeresCostFunctionAdapter(chain);

    double m = 0.;
    double b = 0.;
    double x = 5.;
    double y = targetM * x + targetB;
    double x2 = 10.;
    double y2 = targetM * x2 + targetB;
    std::vector<double*> params(4);
    params[0] = &x;
    params[1] = &m;
    params[2] = &b;
    params[3] = &y;


    ceres::Problem problem;
    problem.AddResidualBlock(costFcn, NULL, params);
    problem.SetParameterBlockConstant(&x);
    problem.SetParameterBlockConstant(&y);

    params[0] = &x2;
    params[1] = &m;
    params[2] = &b;
    params[3] = &y2;
    problem.AddResidualBlock(costFcn, NULL, params);
    problem.SetParameterBlockConstant(&x2);
    problem.SetParameterBlockConstant(&y2);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    ASSERT_NEAR(m, targetM, 1e-7);
    ASSERT_NEAR(b, targetB, 1e-7);
}


TEST(Solver, SGD) {
    double targetM = 10.;
    double targetB = 5.;
    int steps = 1000;

    std::shared_ptr<line_test::AdditiveConstant<double>> add(new line_test::AdditiveConstant<double>());
    std::shared_ptr<line_test::MultiplicativeConstant<double>> multi(new line_test::MultiplicativeConstant<double>());
    std::shared_ptr<line_test::LinearError<double>> error(new line_test::LinearError<double>());


    std::shared_ptr<block_optimization::ProcessingChain<double>> chain(
        new block_optimization::ProcessingChain<double>());
    chain->appendBlock(multi);
    chain->appendBlock(add);
    chain->appendBlock(error);

    double m = 0.;
    double b = 0.;
    double x = 5.;
    double y = x;
    std::vector<double*> params(3);
    params[0] = &m;
    params[1] = &b;
    params[2] = &y;

    block_optimization::SGDSolver<double> solver;
    solver.addChain(chain, params);
    solver.setBatchSize(25);
    solver.setLearningRate(1e-3);


    std::vector<std::vector<double*>> data(100);
    std::vector<std::vector<double*>> target(100);

    for (int i = 0; i < 100; i++) {
        double x = double(rand() % 1000) / 100. - 5.;

        data[i].push_back(new double(x));
        target[i].push_back(new double(targetM * x + targetB));
    }

    for (int k = 0; k < steps; k++) {
        solver.step(data, target);
    }

    for (int i = 0; i < 100; i++) {
        delete data[i][0];
        data[i][0] = NULL;
        delete target[i][0];
        target[i][0] = NULL;
    }

    ASSERT_NEAR(m, targetM, 1e-7);
    ASSERT_NEAR(b, targetB, 1e-7);
}

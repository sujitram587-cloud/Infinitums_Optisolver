#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "../include/qp_model.hpp"
#include "../include/qp_solver.hpp"

using namespace std;


// ============================================================
// TEST UTILITIES
// ============================================================

struct TestCase {
    string name;
    QPModel model;
    string expectedStatus;
    vector<double> expectedX;
    double expectedObjective;
};

const double TOLERANCE = 1e-6;


// ------------------------------------------------------------
// Check whether two floating-point values are approximately equal
// ------------------------------------------------------------
bool nearlyEqual(double a, double b, double tolerance = TOLERANCE) {
    return fabs(a - b) <= tolerance;
}


// ------------------------------------------------------------
// Check solution vector
// ------------------------------------------------------------
bool checkSolution(
    const vector<double>& actual,
    const vector<double>& expected
) {
    if (actual.size() != expected.size())
        return false;

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!nearlyEqual(actual[i], expected[i]))
            return false;
    }

    return true;
}


// ------------------------------------------------------------
// Run one test
// ------------------------------------------------------------
bool runTest(const TestCase& test, int testNumber) {

    QPSolver solver;
    QPResult result = solver.solve(test.model);

    bool statusOK =
        (result.status == test.expectedStatus);

    bool solutionOK = true;
    bool objectiveOK = true;

    // Only check solution/objective when an optimal solution
    // is expected.
    if (test.expectedStatus == "OPTIMAL") {

        solutionOK =
            checkSolution(result.x, test.expectedX);

        objectiveOK =
            nearlyEqual(
                result.objectiveValue,
                test.expectedObjective
            );
    }

    bool passed =
        statusOK &&
        solutionOK &&
        objectiveOK;

    cout << left
         << setw(5)  << testNumber
         << setw(32) << test.name
         << setw(8)  << test.model.n
         << setw(24) << result.status
         << setw(12) << fixed << setprecision(4)
         << result.solveTime
         << (passed ? "PASS" : "FAIL")
         << "\n";

    // Extra information for failed tests
    if (!passed) {

        cout << "     Expected status    : "
             << test.expectedStatus << "\n";

        cout << "     Actual status      : "
             << result.status << "\n";

        if (test.expectedStatus == "OPTIMAL") {

            cout << "     Expected objective : "
                 << test.expectedObjective << "\n";

            cout << "     Actual objective   : "
                 << result.objectiveValue << "\n";

            cout << "     Expected x         : ";

            for (double value : test.expectedX)
                cout << value << " ";

            cout << "\n";

            cout << "     Actual x           : ";

            for (double value : result.x)
                cout << value << " ";

            cout << "\n";
        }
    }

    return passed;
}


// ============================================================
// BASIC QP TESTS
// ============================================================

TestCase testBasic() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    return {
        "Basic Unconstrained QP",
        model,
        "OPTIMAL",
        {2.0, 3.0},
        -13.0
    };
}
TestCase testPositiveDefiniteCrossTermQP() {
    QPModel model(2);

    model.Q = {
        {4.0, 1.0},
        {1.0, 2.0}
    };

    model.c = {
        -9.0,
        -5.0
    };

    // Q is positive definite.
    // Solving Qx + c = 0 gives x = (2, 1).
    // f(x) = 0.5*x^T*Q*x + c^T*x = -12.5

    return {
        "Positive Definite Cross-Term QP",
        model,
        "OPTIMAL",
        {13.0 / 7.0, 11.0 / 7.0},
        -86.0 / 7.0
    };
}
TestCase testZeroLinearTermQP() {
    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 4.0}
    };

    model.c = {
        0.0,
        0.0
    };

    // With positive definite Q and c = 0,
    // the minimum occurs at x = (0, 0).
    // Objective value = 0.

    return {
        "Zero Linear Term QP",
        model,
        "OPTIMAL",
        {0.0, 0.0},
        0.0
    };
}
TestCase testSingleVariableNonZeroQP() {
    QPModel model(1);

    model.Q = {
        {4.0}
    };

    model.c = {
        -8.0
    };

    // 4x - 8 = 0
    // Therefore x = 2
    // f(2) = 0.5(4)(2^2) - 8(2) = -8

    return {
        "Single Variable Non-Zero QP",
        model,
        "OPTIMAL",
        {2.0},
        -8.0
    };
}
TestCase testThreeVariableCrossTermQP() {
    QPModel model(3);

    model.Q = {
        {4.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 2.0}
    };

    model.c = {
        -5.0,
        -5.0,
        -3.0
    };

    // Qx + c = 0 gives x = (1, 1, 1).
    // f(1,1,1) = -6.5

    return {
        "Three Variable Cross-Term QP",
        model,
        "OPTIMAL",
        {1.0, 1.0, 1.0},
        -6.5
    };
}



// ============================================================
// EQUALITY CONSTRAINT TESTS
// ============================================================

TestCase testEquality() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.A_eq = {
        {1.0, 1.0}
    };

    model.b_eq = {
        6.0
    };

    return {
        "Equality Constrained QP",
        model,
        "OPTIMAL",
        {2.5, 3.5},
        -12.5
    };
}
TestCase testSingleVariableEqualityQP() {
    QPModel model(1);

    model.Q = {
        {2.0}
    };

    model.c = {
        -4.0
    };

    model.A_eq = {
        {1.0}
    };

    model.b_eq = {
        3.0
    };

    // Equality constraint forces x = 3.
    // f(3) = 0.5(2)(3^2) - 4(3)
    //      = 9 - 12
    //      = -3

    return {
        "Single Variable Equality QP",
        model,
        "OPTIMAL",
        {3.0},
        -3.0
    };
}
TestCase testThreeVariableEqualityCrossTermQP() {
    QPModel model(3);

    model.Q = {
        {4.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 2.0}
    };

    model.c = {
        -5.0,
        -5.0,
        -3.0
    };

    model.A_eq = {
        {1.0, 1.0, 1.0}
    };

    model.b_eq = {
        3.0
    };

    // Equality constraint: x1 + x2 + x3 = 3
    // The unconstrained optimum is already (1,1,1),
    // which satisfies the equality constraint.
    // Therefore the optimum remains (1,1,1).
    // Objective value = -6.5

    return {
        "Three Variable Equality Cross-Term QP",
        model,
        "OPTIMAL",
        {1.0, 1.0, 1.0},
        -6.5
    };
}
TestCase testTwoVariableEqualityCrossTermQP() {
    QPModel model(2);

    model.Q = {
        {4.0, 1.0},
        {1.0, 2.0}
    };

    model.c = {
        -9.0,
        -5.0
    };

    model.A_eq = {
        {1.0, 1.0}
    };

    model.b_eq = {
        3.0
    };

// Solving the constrained problem gives:
// x = (7/4, 5/4)
//
// Objective = -97/8

    return {
        "Two Variable Equality Cross-Term QP",
        model,
        "OPTIMAL",
        {7.0 / 4.0, 5.0 / 4.0},
        -97.0 / 8.0
    };
}
// ------------------------------------------------------------
// Multiple equality constraints
// ------------------------------------------------------------
TestCase testMultipleEqualities() {

    QPModel model(3);

    model.Q = {
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0}
    };

    model.c = {
        -2.0,
        -4.0,
        -6.0
    };

    // Constraints:
    // x1 + x2 = 3
    // x2 + x3 = 5
    model.A_eq = {
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0}
    };

    model.b_eq = {
        3.0,
        5.0
    };

    return {
        "Multiple Equality QP",
        model,
        "OPTIMAL",
        {1.0, 2.0, 3.0},
        -14.0
    };
}
TestCase testEqualityWithNegativeCoefficient() {
    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -2.0
    };

    model.A_eq = {
        {1.0, -1.0}
    };

    model.b_eq = {
        1.0
    };

    // Equality: x1 - x2 = 1
    //
    // Unconstrained optimum: (2,1)
    // Since 2 - 1 = 1, the equality is satisfied.
    // Therefore the optimum remains (2,1).
    //
    // f(2,1) = (4 + 1) - 8 - 2 = -5

    return {
        "Equality With Negative Coefficient",
        model,
        "OPTIMAL",
        {2.0, 1.0},
        -5.0
    };
}

// ------------------------------------------------------------
// Generate an equality-constrained QP
// ------------------------------------------------------------
TestCase generateEqualityTest(
    int n,
    const string& name
) {
    QPModel model(n);

    vector<double> expectedX(n);

    // Positive-definite diagonal Q.
    for (int i = 0; i < n; ++i) {

        model.Q[i][i] = 2.0 + i;

        // Known optimal solution.
        expectedX[i] =
            static_cast<double>(i + 1);

        // Qx + c = 0 at the optimum.
        model.c[i] =
            -model.Q[i][i] * expectedX[i];
    }

    // Equality constraint:
    // x1 + x2 + ... + xn = sum(expectedX)
    vector<double> equalityRow(n, 1.0);

    double equalityValue = 0.0;

    for (double value : expectedX)
        equalityValue += value;

    model.A_eq = {
        equalityRow
    };

    model.b_eq = {
        equalityValue
    };

    // Calculate expected objective.
    double expectedObjective = 0.0;

    for (int i = 0; i < n; ++i) {

        expectedObjective +=
            0.5 *
            model.Q[i][i] *
            expectedX[i] *
            expectedX[i];

        expectedObjective +=
            model.c[i] *
            expectedX[i];
    }

    return {
        name,
        model,
        "OPTIMAL",
        expectedX,
        expectedObjective
    };
}


// ============================================================
// INEQUALITY CONSTRAINT TESTS
// ============================================================

TestCase testInequality() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.A_ineq = {
        {1.0, 0.0}
    };

    model.b_ineq = {
        1.0
    };

    return {
        "Inequality Constrained QP",
        model,
        "OPTIMAL",
        {1.0, 3.0},
        -12.0
    };
}
TestCase testSingleVariableInequalityQP() {
    QPModel model(1);

    model.Q = {
        {2.0}
    };

    model.c = {
        -4.0
    };

    model.A_ineq = {
        {1.0}
    };

    model.b_ineq = {
        1.0
    };

    // Unconstrained optimum: x = 2
    // Constraint: x <= 1
    // Therefore optimum is x = 1.
    // f(1) = 0.5(2)(1^2) - 4(1) = -3

    return {
        "Single Variable Inequality QP",
        model,
        "OPTIMAL",
        {1.0},
        -3.0
    };
}
TestCase testThreeVariableInequalityCrossTermQP() {
    QPModel model(3);

    model.Q = {
        {4.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 2.0}
    };

    model.c = {
        -5.0,
        -5.0,
        -3.0
    };

    model.A_ineq = {
        {1.0, 1.0, 1.0}
    };

    model.b_ineq = {
        2.0
    };

  // Constraint: x1 + x2 + x3 <= 2
// The constraint is active at the optimum.
//
// KKT solution:
// x = (5/7, 6/7, 3/7)
//
// Objective = -41/7
    return {
        "Three Variable Inequality Cross-Term QP",
        model,
        "OPTIMAL",
       {5.0 / 7.0, 6.0 / 7.0, 3.0 / 7.0},
       -41.0 / 7.0
    };
}
// ------------------------------------------------------------
// Multiple inequality constraints
// ------------------------------------------------------------
TestCase testMultipleInequalities() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    // Constraints:
    // x1 <= 2
    // x2 <= 2
    model.A_ineq = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    model.b_ineq = {
        2.0,
        2.0
    };

    return {
        "Multiple Inequality QP",
        model,
        "OPTIMAL",
        {2.0, 2.0},
        -12.0
    };
}

// ------------------------------------------------------------
// Multiple active inequality constraints
// ------------------------------------------------------------
TestCase testMultipleActiveInequalities() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -6.0,
        -8.0
    };

    // Unconstrained optimum: (3, 4)
    // Both constraints are active:
    // x1 <= 2
    // x2 <= 3
    model.A_ineq = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    model.b_ineq = {
        2.0,
        3.0
    };

    // At x = (2,3):
    // f = (2^2 + 3^2) - 6(2) - 8(3)
    //   = 13 - 12 - 24
    //   = -23
    return {
        "Multiple Active Inequalities",
        model,
        "OPTIMAL",
        {2.0, 3.0},
        -23.0
    };
}


// ------------------------------------------------------------
// Generate an inequality-constrained QP
// ------------------------------------------------------------
TestCase generateInequalityTest(
    int n,
    const string& name
) {
    QPModel model(n);

    vector<double> expectedX(n);

    // Positive-definite diagonal Q.
    for (int i = 0; i < n; ++i) {

        model.Q[i][i] = 2.0 + i;

        // Known unconstrained optimum.
        expectedX[i] =
            static_cast<double>(i + 1);

        model.c[i] =
            -model.Q[i][i] * expectedX[i];
    }

    // Add one inequality:
    //
    // x1 + x2 + ... + xn <= sum(expectedX)
    //
    // The known optimum lies exactly on this boundary.
    vector<double> inequalityRow(n, 1.0);

    double inequalityValue = 0.0;

    for (double value : expectedX)
        inequalityValue += value;

    model.A_ineq = {
        inequalityRow
    };

    model.b_ineq = {
        inequalityValue
    };

    // Calculate expected objective.
    double expectedObjective = 0.0;

    for (int i = 0; i < n; ++i) {

        expectedObjective +=
            0.5 *
            model.Q[i][i] *
            expectedX[i] *
            expectedX[i];

        expectedObjective +=
            model.c[i] *
            expectedX[i];
    }

    return {
        name,
        model,
        "OPTIMAL",
        expectedX,
        expectedObjective
    };
}


// ============================================================
// BOUNDS TESTS
// ============================================================

TestCase testLowerBound() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.lowerBound = {
        3.0,
        -100.0
    };

    model.upperBound = {
        100.0,
        100.0
    };

    return {
        "Lower Bound QP",
        model,
        "OPTIMAL",
        {3.0, 3.0},
        -12.0
    };
}
TestCase testSingleVariableUpperBound() {
    QPModel model(1);

    model.Q = {
        {4.0}
    };

    model.c = {
        -8.0
    };

    model.lowerBound = {-100.0};
    model.upperBound = {1.0};

    // Unconstrained optimum: x = 2
    // Upper bound forces x = 1
    // f(1) = 0.5(4)(1^2) - 8(1) = -6

    return {
        "Single Variable Upper Bound",
        model,
        "OPTIMAL",
        {1.0},
        -6.0
    };
}
TestCase testSingleVariableBothBounds() {
    QPModel model(1);

    model.Q = {
        {4.0}
    };

    model.c = {
        -8.0
    };

    model.lowerBound = {1.0};
    model.upperBound = {2.0};

    // Unconstrained optimum: x = 2
    // Bounds: 1 <= x <= 2
    // Therefore optimum remains x = 2.
    // f(2) = 0.5(4)(2^2) - 8(2)
    //      = 8 - 16
    //      = -8

    return {
        "Single Variable Both Bounds",
        model,
        "OPTIMAL",
        {2.0},
        -8.0
    };
}


// ------------------------------------------------------------
// Upper bound test
// ------------------------------------------------------------
TestCase testUpperBound() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.lowerBound = {
        -100.0,
        -100.0
    };

    model.upperBound = {
        1.0,
        100.0
    };

    return {
        "Upper Bound QP",
        model,
        "OPTIMAL",
        {1.0, 3.0},
        -12.0
    };
}


// ------------------------------------------------------------
// Active upper-bound test
// ------------------------------------------------------------
TestCase testActiveUpperBound() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    // Unconstrained optimum is (2, 3).
    // Force x1 to its upper boundary: x1 <= 1.
    model.upperBound = {
        1.0,
        100.0
    };

    model.lowerBound = {
        -100.0,
        -100.0
    };

    return {
        "Active Upper Boundary",
        model,
        "OPTIMAL",
        {1.0, 3.0},
        -12.0
    };
}

// ------------------------------------------------------------
// Both lower and upper bounds
// ------------------------------------------------------------
TestCase testLowerAndUpperBounds() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    // Unconstrained optimum: (2, 3)
    //
    // Bounds:
    // 1 <= x1 <= 2
    // 2 <= x2 <= 3
    //
    // The optimum is exactly on the upper boundaries.
    model.lowerBound = {
        1.0,
        2.0
    };

    model.upperBound = {
        2.0,
        3.0
    };

    // f(2,3) = (4 + 9) - 8 - 18 = -13
    return {
        "Lower and Upper Bounds",
        model,
        "OPTIMAL",
        {2.0, 3.0},
        -13.0
    };
}

// ------------------------------------------------------------
// Both variables at their lower bounds
// ------------------------------------------------------------
TestCase testMultipleActiveLowerBounds() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    // Unconstrained optimum: (2, 3)
    // Force both variables below their unconstrained optimum.
    model.lowerBound = {
        3.0,
        4.0
    };

    model.upperBound = {
        100.0,
        100.0
    };

    // f(3,4)
    // = (9 + 16) - 12 - 24
    // = -11
    return {
        "Multiple Active Lower Bounds",
        model,
        "OPTIMAL",
        {3.0, 4.0},
        -11.0
    };
}
TestCase testSingleVariableLowerBound() {
    QPModel model(1);
    model.Q = {{2}};
    model.c = {-4};
    model.lowerBound = {3};
    model.upperBound = {100};

    // Unconstrained optimum: x = 2
    // Lower bound forces x = 3
    // f(3) = 3² - 4(3) = -3
    return {"Single Variable Lower Bound", model, "OPTIMAL", {3}, -3.0};
}


// ============================================================
// MIXED CONSTRAINT TESTS
// ============================================================

// ------------------------------------------------------------
// Generate a QP with both equality and inequality constraints
// ------------------------------------------------------------
TestCase generateMixedConstraintTest(
    int n,
    const string& name
) {
    QPModel model(n);

    vector<double> expectedX(n);

    // Positive-definite diagonal Q
    for (int i = 0; i < n; ++i) {

        model.Q[i][i] = 2.0 + i;

        expectedX[i] =
            static_cast<double>(i + 1);

        model.c[i] =
            -model.Q[i][i] * expectedX[i];
    }

    // --------------------------------------------------------
    // Equality:
    // x1 + x2 + ... + xn = sum(expectedX)
    // --------------------------------------------------------
    vector<double> equalityRow(n, 1.0);

    double equalityValue = 0.0;

    for (double value : expectedX)
        equalityValue += value;

    model.A_eq = {
        equalityRow
    };

    model.b_eq = {
        equalityValue
    };

    // --------------------------------------------------------
    // Inequality:
    // x1 + x2 + ... + xn <= sum(expectedX)
    // --------------------------------------------------------
    vector<double> inequalityRow(n, 1.0);

    model.A_ineq = {
        inequalityRow
    };

    model.b_ineq = {
        equalityValue
    };

    // --------------------------------------------------------
    // Calculate expected objective
    // --------------------------------------------------------
    double expectedObjective = 0.0;

    for (int i = 0; i < n; ++i) {

        expectedObjective +=
            0.5 *
            model.Q[i][i] *
            expectedX[i] *
            expectedX[i];

        expectedObjective +=
            model.c[i] *
            expectedX[i];
    }

    return {
        name,
        model,
        "OPTIMAL",
        expectedX,
        expectedObjective
    };
}
// ------------------------------------------------------------
// Multiple equality and inequality constraints
// ------------------------------------------------------------
TestCase testMultipleMixedConstraints() {

    QPModel model(3);

    model.Q = {
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0}
    };

    model.c = {
        -2.0,
        -4.0,
        -6.0
    };

    // Equality constraints:
    // x1 + x2 = 3
    // x2 + x3 = 5
    model.A_eq = {
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0}
    };

    model.b_eq = {
        3.0,
        5.0
    };

    // Inequality:
    // x1 <= 1
    //
    // The equality constraints give:
    // x = (1, 2, 3)
    // so the inequality is active.
    model.A_ineq = {
        {1.0, 0.0, 0.0}
    };

    model.b_ineq = {
        1.0
    };

    // f(1,2,3)
    // = (1 + 4 + 9) - (2 + 8 + 18)
    // = 14 - 28
    // = -14
    return {
        "Multiple Mixed Constraints",
        model,
        "OPTIMAL",
        {1.0, 2.0, 3.0},
        -14.0
    };
}
TestCase testEqualityWithActiveLowerBound() {
    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.A_eq = {
        {1.0, 1.0}
    };

    model.b_eq = {
        5.0
    };

    model.lowerBound = {
        3.0,
        -100.0
    };

    model.upperBound = {
        100.0,
        100.0
    };

    // Equality: x1 + x2 = 5
    // Lower bound: x1 >= 3
    //
    // Unconstrained optimum: (2,3)
    // Along x1+x2=5, the minimum would be (2,3),
    // but x1 >= 3 forces the solution to (3,2).
    //
    // f(3,2) = (9 + 4) - 12 - 12 = -11

    return {
        "Equality With Active Lower Bound",
        model,
        "OPTIMAL",
        {3.0, 2.0},
        -11.0
    };
}
TestCase testEqualityWithActiveUpperBound() {
    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.A_eq = {
        {1.0, 1.0}
    };

    model.b_eq = {
        4.0
    };

    model.lowerBound = {
        -100.0,
        -100.0
    };

    model.upperBound = {
        1.0,
        100.0
    };

    // Equality: x1 + x2 = 4
    // Unconstrained optimum: (2,3)
    // Upper bound forces x1 <= 1.
    // Therefore x1 = 1 and x2 = 3.
    //
    // f(1,3) = (1 + 9) - 4 - 18 = -12

    return {
        "Equality With Active Upper Bound",
        model,
        "OPTIMAL",
        {1.0, 3.0},
        -12.0
    };
}

// ============================================================
// CONVEXITY TESTS
// ============================================================

TestCase testNonConvex() {

    QPModel model(2);

    // Q has one negative eigenvalue.
    // Therefore the QP is non-convex.
    model.Q = {
        {-2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        0.0,
        0.0
    };

    return {
        "Non-Convex QP",
        model,
        "NON_CONVEX_QP",
        {},
        0.0
    };
}

// ------------------------------------------------------------
// Positive semidefinite Q matrix
// ------------------------------------------------------------
TestCase testPositiveSemidefiniteQ() {

    QPModel model(2);

    // Singular PSD matrix:
    // eigenvalues are 0 and 2.
    model.Q = {
        {1.0, -1.0},
        {-1.0, 1.0}
    };

    model.c = {
        0.0,
        0.0
    };

    return {
        "Positive Semidefinite Q",
        model,
        "SINGULAR_OR_UNSTABLE_Q",
        {},
        0.0
    };
}

// ------------------------------------------------------------
// Asymmetric Q matrix
// ------------------------------------------------------------
TestCase testAsymmetricQ() {

    QPModel model(2);

    model.Q = {
        {2.0, 1.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    return {
        "Asymmetric Q Matrix",
        model,
        "Q_NOT_SYMMETRIC",
        {},
        0.0
    };
}


// ============================================================
// EDGE CASE TESTS
// ============================================================

TestCase testInfeasible() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    model.A_eq = {
        {1.0, 1.0}
    };

    model.b_eq = {
        6.0
    };

    model.A_ineq = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    model.b_ineq = {
        2.0,
        2.0
    };

    return {
        "Infeasible QP",
        model,
        "INFEASIBLE_CONSTRAINTS",
        {},
        0.0
    };
}
TestCase testSingleVariableInfeasibleBounds() {
    QPModel model(1);

    model.Q = {
        {2.0}
    };

    model.c = {
        -4.0
    };

    model.lowerBound = {
        5.0
    };

    model.upperBound = {
        2.0
    };

    // Lower bound x >= 5 conflicts with upper bound x <= 2.
    // Therefore the model is invalid.

    return {
        "Single Variable Infeasible Bounds",
        model,
        "INVALID_MODEL",
        {},
        0.0
    };
}
// ------------------------------------------------------------
// Invalid bounds: lower bound greater than upper bound
// ------------------------------------------------------------
TestCase testInvalidBounds() {

    QPModel model(2);

    model.Q = {
        {2.0, 0.0},
        {0.0, 2.0}
    };

    model.c = {
        -4.0,
        -6.0
    };

    // Invalid:
    // x1 >= 5
    // x1 <= 1
    model.lowerBound = {
        5.0,
        -100.0
    };

    model.upperBound = {
        1.0,
        100.0
    };

    return {
        "Invalid Bounds QP",
        model,
        "INVALID_MODEL",
        {},
        0.0
    };
}

// ------------------------------------------------------------
// Invalid model: zero variables
// ------------------------------------------------------------
TestCase testZeroVariables() {

    QPModel model(0);

    return {
        "Zero Variable QP",
        model,
        "INVALID_MODEL",
        {},
        0.0
    };
}


// ============================================================
// STRESS / MULTI-VARIABLE TESTS
// ============================================================

// ------------------------------------------------------------
// Generate an unconstrained QP for different variable counts
// ------------------------------------------------------------
TestCase generateUnconstrainedTest(
    int n,
    const string& name
) {
    QPModel model(n);

    vector<double> expectedX(n);

    // Create a positive-definite diagonal Q matrix.
    for (int i = 0; i < n; ++i) {

        model.Q[i][i] = 2.0 + i;

        // Known optimal solution:
        expectedX[i] =
            static_cast<double>(i + 1);

        // For an unconstrained QP:
        // Qx + c = 0
        //
        // Therefore:
        // c = -Qx
        model.c[i] =
            -model.Q[i][i] * expectedX[i];
    }

    // Calculate expected objective:
    // f(x) = 0.5*x^T*Q*x + c^T*x
    double expectedObjective = 0.0;

    for (int i = 0; i < n; ++i) {

        expectedObjective +=
            0.5 *
            model.Q[i][i] *
            expectedX[i] *
            expectedX[i];

        expectedObjective +=
            model.c[i] *
            expectedX[i];
    }

    return {
        name,
        model,
        "OPTIMAL",
        expectedX,
        expectedObjective
    };
}
TestCase test100VariableUnconstrainedQP() {
    const int n = 100;

    QPModel model(n);
    vector<double> expectedX(n, 1.0);

    // Q = 2I and c = -2 for every variable.
    // Therefore:
    //   Qx + c = 0
    //   2x - 2 = 0
    //   x = 1
    for (int i = 0; i < n; ++i) {
        model.Q[i][i] = 2.0;
        model.c[i] = -2.0;
    }

    // Objective at x = (1,1,...,1):
    // 0.5 * 100 * 2 - 2 * 100 = -100
    return {
        "100 Variable Unconstrained QP",
        model,
        "OPTIMAL",
        expectedX,
        -100.0
    };
}

// ============================================================
// TEST RUNNER
// ============================================================

int main() {

    cout << "\n";
    cout << "==========================================================================\n";
    cout << "                  SOVEREIGN QP SOLVER TEST SUITE\n";
    cout << "==========================================================================\n";

    cout << left
         << setw(5)  << "ID"
         << setw(32) << "TEST"
         << setw(8)  << "N"
         << setw(24) << "STATUS"
         << setw(12) << "TIME(ms)"
         << "RESULT\n";

    cout << "--------------------------------------------------------------------------\n";

    vector<TestCase> tests = {

        // --------------------------------------------------------
        // BASIC QP TESTS
        // --------------------------------------------------------
        testBasic(),
        testPositiveDefiniteCrossTermQP(),
        testZeroLinearTermQP(),
        testSingleVariableNonZeroQP(),
        testThreeVariableCrossTermQP(),

        // --------------------------------------------------------
        // EQUALITY CONSTRAINT TESTS
        // --------------------------------------------------------
        testEquality(),
        testMultipleEqualities(),
        testSingleVariableEqualityQP(),
        testThreeVariableEqualityCrossTermQP(),
        testTwoVariableEqualityCrossTermQP(),
        testEqualityWithNegativeCoefficient(),
        generateEqualityTest(3, "3 Variable Equality QP"),
        generateEqualityTest(5, "5 Variable Equality QP"),
        generateEqualityTest(10, "10 Variable Equality QP"),
        generateEqualityTest(20, "20 Variable Equality QP"),


        // --------------------------------------------------------
        // INEQUALITY CONSTRAINT TESTS
        // --------------------------------------------------------
        testInequality(),
        testMultipleInequalities(),
        testMultipleActiveInequalities(),
        testSingleVariableInequalityQP(),
        testThreeVariableInequalityCrossTermQP(),
        generateInequalityTest(3, "3 Variable Inequality QP"),
        generateInequalityTest(5, "5 Variable Inequality QP"),
        generateInequalityTest(10, "10 Variable Inequality QP"),
        generateInequalityTest(20, "20 Variable Inequality QP"),

        // --------------------------------------------------------
        // BOUNDS TESTS
        // --------------------------------------------------------
        testLowerBound(),
        testUpperBound(),
        testActiveUpperBound(),
        testLowerAndUpperBounds(),
        testMultipleActiveLowerBounds(),
        testSingleVariableUpperBound(),
        testSingleVariableBothBounds(),

        // --------------------------------------------------------
        // MIXED CONSTRAINT TESTS
        // --------------------------------------------------------
        generateMixedConstraintTest(3, "3 Variable Mixed QP"),
        testMultipleMixedConstraints(),
        testEqualityWithActiveLowerBound(),
        testEqualityWithActiveUpperBound(),
        generateMixedConstraintTest(5, "5 Variable Mixed QP"),
        generateMixedConstraintTest(10, "10 Variable Mixed QP"),
        generateMixedConstraintTest(20, "20 Variable Mixed QP"),

        // --------------------------------------------------------
        // CONVEXITY TESTS
        // --------------------------------------------------------
        testNonConvex(),
        testAsymmetricQ(),
        testPositiveSemidefiniteQ(),

        // --------------------------------------------------------
        // EDGE CASE TESTS
        // --------------------------------------------------------
        testInfeasible(),
        testInvalidBounds(),
        testZeroVariables(),
        testSingleVariableInfeasibleBounds(),

        // --------------------------------------------------------
        // STRESS / MULTI-VARIABLE TESTS
        // --------------------------------------------------------
        generateUnconstrainedTest(1, "1 Variable QP"),
        generateUnconstrainedTest(3, "3 Variable QP"),
        generateUnconstrainedTest(5, "5 Variable QP"),
        generateUnconstrainedTest(10, "10 Variable QP"),
        generateUnconstrainedTest(20, "20 Variable QP"),
        test100VariableUnconstrainedQP()
    };


    int passed = 0;

    for (int i = 0; i < static_cast<int>(tests.size()); ++i) {

        if (runTest(tests[i], i + 1))
            passed++;
    }


    cout << "--------------------------------------------------------------------------\n";

    cout << "Tests Passed : "
         << passed
         << " / "
         << tests.size()
         << "\n";

    if (passed == static_cast<int>(tests.size()))
        cout << "Overall Result : ALL TESTS PASSED\n";
    else
        cout << "Overall Result : SOME TESTS FAILED\n";

    cout << "==========================================================================\n";

    return 0;
}
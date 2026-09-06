#include <iostream>
#include <vector>
#include "../include/qp_model.hpp"
#include "../include/qp_solver.hpp"

int main() {

    std::cout << "====================================\n";
    std::cout << "       SOVEREIGN QP SOLVER\n";
    std::cout << "====================================\n\n";

    // Create a QP model with 2 variables
    QPModel model(2);

    // Quadratic objective:
    // f(x) = 1/2 x^T Q x + c^T x
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
    4.0,
    5.0
};

    // Variable bounds:
    // -100 <= x1 <= 100
    // -100 <= x2 <= 100
    model.lowerBound = {
        -100.0,
        -100.0
    };

    model.upperBound = {
        100.0,
        100.0
    };

    // Create solver
    QPSolver solver;

    // Solve the quadratic programming problem
    QPResult result = solver.solve(model);

    // Display result
    std::cout << "Status: "
              << result.status << "\n\n";

    if (!result.x.empty()) {

        std::cout << "Optimal solution:\n";

        for (size_t i = 0; i < result.x.size(); ++i) {
            std::cout << "x[" << i << "] = "
                      << result.x[i] << "\n";
        }

        std::cout << "\nObjective value = "
                  << result.objectiveValue << "\n";

        std::cout << "Solve time = "
                  << result.solveTime
                  << " ms\n";
    }

    std::cout << "\n====================================\n";

    return 0;
}
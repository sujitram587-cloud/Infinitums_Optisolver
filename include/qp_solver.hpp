#ifndef QP_SOLVER_HPP
#define QP_SOLVER_HPP

#include "qp_model.hpp"
#include <vector>
#include <string>

struct QPResult {
    std::string status;
    std::vector<double> x;
    double objectiveValue;
    double solveTime;
};

class QPSolver {
public:

    QPResult solve(const QPModel& model);

private:

    // Solve Ax = rhs using Gaussian elimination
    bool solveLinearSystem(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& rhs,
        std::vector<double>& x
    );
};

#endif
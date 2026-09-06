#include "../include/qp_solver.hpp"

#include <cmath>
#include <chrono>
#include <algorithm>

QPResult QPSolver::solve(const QPModel& model) {

    QPResult result;

    auto start = std::chrono::high_resolution_clock::now();

    // -----------------------------
    // Basic validation
    // -----------------------------

    if (!model.isValid()) {
        result.status = "INVALID_MODEL";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

  

    // Q must be symmetric for our convex QP prototype.
    if (!model.isSymmetric()) {
        result.status = "Q_NOT_SYMMETRIC";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }
    int n = model.n;
    std::vector<double> x;

    // Handle equality constraints using KKT system
if (!model.A_eq.empty()) {

    int m = static_cast<int>(model.A_eq.size());

    if (model.b_eq.size() != static_cast<size_t>(m)) {
        result.status = "INVALID_EQUALITY_CONSTRAINTS";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

    // KKT system:
    //
    // [ Q   A^T ] [ x      ] = [ -c    ]
    // [ A    0  ] [ lambda ]   [ b_eq  ]

    int total = n + m;

    std::vector<std::vector<double>> K(
        total, std::vector<double>(total, 0.0)
    );

    std::vector<double> rhs(total, 0.0);

    // Q block
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            K[i][j] = model.Q[i][j];
        }
        rhs[i] = -model.c[i];
    }

    // A^T and A blocks
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            K[j][n + i] = model.A_eq[i][j];
            K[n + i][j] = model.A_eq[i][j];
        }

        rhs[n + i] = model.b_eq[i];
    }

    // Solve KKT system
    std::vector<double> solution;

    bool success = solveLinearSystem(K, rhs, solution);

    if (!success) {
        result.status = "KKT_SYSTEM_SINGULAR";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

    // Extract x from the KKT solution
    x.assign(solution.begin(), solution.begin() + n);
}
else {
    std::vector<double> rhs(n);

    for (int i = 0; i < n; ++i) {
        rhs[i] = -model.c[i];
    }

    bool success = solveLinearSystem(model.Q, rhs, x);

    if (!success) {
        result.status = "SINGULAR_OR_UNSTABLE_Q";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }
}



    // -----------------------------
    // Check variable bounds
    // -----------------------------

    const double tolerance = 1e-9;

    for (int i = 0; i < n; i++) {

        if (x[i] < model.lowerBound[i] - tolerance ||
            x[i] > model.upperBound[i] + tolerance) {

            result.status = "SOLUTION_OUTSIDE_BOUNDS";
            result.x = x;
            result.objectiveValue =
                model.objectiveValue(x);

            return result;
        }
    }

// --------------------------------
// Optimality check
// --------------------------------

if (model.A_eq.empty()) {

    std::vector<double> gradient =
        model.gradient(x);

    double maxGradient = 0.0;

    for (double value : gradient) {
        maxGradient =
            std::max(maxGradient, std::abs(value));
    }

    if (maxGradient > 1e-7) {
        result.status = "OPTIMALITY_CHECK_FAILED";
        result.x = x;
        result.objectiveValue =
            model.objectiveValue(x);
        return result;
    }
}

    // -----------------------------
    // Successful solution
    // -----------------------------

    result.status = "OPTIMAL";
    result.x = x;
    result.objectiveValue =
        model.objectiveValue(x);

    auto end = std::chrono::high_resolution_clock::now();

    result.solveTime =
        std::chrono::duration<double, std::milli>
        (end - start).count();

    return result;
}


// --------------------------------------------------
// Gaussian elimination with partial pivoting
// --------------------------------------------------

bool QPSolver::solveLinearSystem(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x) {

    int n = static_cast<int>(A.size());

    if (n == 0)
        return false;

    if (rhs.size() != static_cast<size_t>(n))
        return false;

    std::vector<std::vector<double>> M = A;
    std::vector<double> b = rhs;

    const double tolerance = 1e-10;

    // Forward elimination
    for (int k = 0; k < n; k++) {

        int pivot = k;

        for (int i = k + 1; i < n; i++) {

            if (std::abs(M[i][k]) >
                std::abs(M[pivot][k])) {

                pivot = i;
            }
        }

        if (std::abs(M[pivot][k]) < tolerance)
            return false;

        std::swap(M[k], M[pivot]);
        std::swap(b[k], b[pivot]);

        for (int i = k + 1; i < n; i++) {

            double factor =
                M[i][k] / M[k][k];

            for (int j = k; j < n; j++) {
                M[i][j] -= factor * M[k][j];
            }

            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {

        double sum = b[i];

        for (int j = i + 1; j < n; j++) {
            sum -= M[i][j] * x[j];
        }

        if (std::abs(M[i][i]) < tolerance)
            return false;

        x[i] = sum / M[i][i];
    }

    return true;
}
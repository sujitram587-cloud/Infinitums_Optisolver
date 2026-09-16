#include "../include/qp_solver.hpp"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>

QPResult QPSolver::solve(const QPModel& model) {

    QPResult result;

    auto start = std::chrono::high_resolution_clock::now();

    // --------------------------------
    // Basic validation
    // --------------------------------

    if (!model.isValid()) {
        result.status = "INVALID_MODEL";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

    if (!model.isSymmetric()) {
        result.status = "Q_NOT_SYMMETRIC";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }
    if (!model.isPositiveSemidefinite()) {
    result.status = "NON_CONVEX_QP";
    result.objectiveValue = 0.0;
    result.solveTime = 0.0;
    return result;
}

    const int n = model.n;
    const double tolerance = 1e-8;

    // --------------------------------
    // Validate equality constraints
    // --------------------------------

    if (model.A_eq.size() != model.b_eq.size()) {
        result.status = "INVALID_EQUALITY_CONSTRAINTS";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

    // --------------------------------
    // Validate inequality constraints
    // --------------------------------

    if (model.A_ineq.size() != model.b_ineq.size()) {
        result.status = "INVALID_INEQUALITY_CONSTRAINTS";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

    for (const auto& row : model.A_eq) {
        if (row.size() != static_cast<size_t>(n)) {
            result.status = "INVALID_EQUALITY_CONSTRAINTS";
            result.objectiveValue = 0.0;
            result.solveTime = 0.0;
            return result;
        }
    }

    for (const auto& row : model.A_ineq) {
        if (row.size() != static_cast<size_t>(n)) {
            result.status = "INVALID_INEQUALITY_CONSTRAINTS";
            result.objectiveValue = 0.0;
            result.solveTime = 0.0;
            return result;
        }
    }

    // --------------------------------
    // Build inequality list
    //
    // A_ineq x <= b_ineq
    //
    // Also convert variable bounds:
    //
    // x[i] <= upperBound[i]
    // -x[i] <= -lowerBound[i]
    // --------------------------------

    std::vector<std::vector<double>> Aineq = model.A_ineq;
    std::vector<double> bineq = model.b_ineq;

// Upper bounds
for (int i = 0; i < n; ++i) {

    // Ignore the default "no upper bound" value.
    if (model.upperBound[i] < 1e19) {

        std::vector<double> row(n, 0.0);
        row[i] = 1.0;

        Aineq.push_back(row);
        bineq.push_back(model.upperBound[i]);
    }
}

// Lower bounds
for (int i = 0; i < n; ++i) {

    // Ignore the default "no lower bound" value.
    if (model.lowerBound[i] > -1e19) {

        std::vector<double> row(n, 0.0);
        row[i] = -1.0;

        Aineq.push_back(row);
        bineq.push_back(-model.lowerBound[i]);
    }
}

    const int meq = static_cast<int>(model.A_eq.size());
    const int mineq = static_cast<int>(Aineq.size());

    // --------------------------------
    // Safety limit for active-set
    // enumeration
    // --------------------------------

    if (mineq > 20) {
        result.status = "TOO_MANY_INEQUALITY_CONSTRAINTS";
        result.objectiveValue = 0.0;
        result.solveTime = 0.0;
        return result;
    }

    // --------------------------------
    // If there are no inequalities,
    // solve equality-constrained QP
    // directly.
    // --------------------------------

    if (mineq == 0) {

        std::vector<double> x;

        if (meq == 0) {

            // Qx = -c
            std::vector<double> rhs(n, 0.0);

            for (int i = 0; i < n; ++i) {
                rhs[i] = -model.c[i];
            }

            bool success =
                solveLinearSystem(model.Q, rhs, x);

            if (!success) {
                result.status = "SINGULAR_OR_UNSTABLE_Q";
                result.objectiveValue = 0.0;
                result.solveTime = 0.0;
                return result;
            }

        } else {

            // --------------------------------
            // Equality KKT system
            //
            // [ Q   A^T ] [ x      ] = [ -c   ]
            // [ A    0  ] [ lambda ]   [ b_eq ]
            // --------------------------------

            const int total = n + meq;

            std::vector<std::vector<double>> K(
                total,
                std::vector<double>(total, 0.0)
            );

            std::vector<double> rhs(
                total,
                0.0
            );

            // Q block
            for (int i = 0; i < n; ++i) {

                for (int j = 0; j < n; ++j) {
                    K[i][j] = model.Q[i][j];
                }

                rhs[i] = -model.c[i];
            }

            // A^T and A blocks
            for (int i = 0; i < meq; ++i) {

                for (int j = 0; j < n; ++j) {

                    K[j][n + i] =
                        model.A_eq[i][j];

                    K[n + i][j] =
                        model.A_eq[i][j];
                }

                rhs[n + i] =
                    model.b_eq[i];
            }

            std::vector<double> solution;

            bool success =
                solveLinearSystem(K, rhs, solution);

            if (!success) {
                result.status = "KKT_SYSTEM_SINGULAR";
                result.objectiveValue = 0.0;
                result.solveTime = 0.0;
                return result;
            }

            x.assign(
                solution.begin(),
                solution.begin() + n
            );
        }

        result.status = "OPTIMAL";
        result.x = x;
        result.objectiveValue =
            model.objectiveValue(x);

        auto end =
            std::chrono::high_resolution_clock::now();

        result.solveTime =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        return result;
    }

    // --------------------------------
    // Active-set enumeration
    // --------------------------------
    //
    // We try different combinations
    // of active inequality constraints.
    //
    // For an active inequality:
    //
    // A_active x = b_active
    //
    // KKT system:
    //
    // [ Q   Aeq^T   Aactive^T ] [x     ] = [-c   ]
    // [ Aeq   0       0       ] [lambda ]   [beq  ]
    // [ Aact  0       0       ] [mu     ]   [bact ]
    //
    // For minimization:
    //
    // mu >= 0
    // --------------------------------

    const std::uint64_t combinations =
        (std::uint64_t(1) << mineq);

    bool found = false;

    std::vector<double> bestX;
    double bestObjective = 0.0;

    for (std::uint64_t mask = 0;
         mask < combinations;
         ++mask) {

        int activeCount = 0;

        for (int i = 0; i < mineq; ++i) {
            if (mask & (std::uint64_t(1) << i)) {
                ++activeCount;
            }
        }

        const int total =
            n + meq + activeCount;

        std::vector<std::vector<double>> K(
            total,
            std::vector<double>(total, 0.0)
        );

        std::vector<double> rhs(
            total,
            0.0
        );

        // --------------------------------
        // Q block
        // --------------------------------

        for (int i = 0; i < n; ++i) {

            for (int j = 0; j < n; ++j) {
                K[i][j] = model.Q[i][j];
            }

            rhs[i] = -model.c[i];
        }

        // --------------------------------
        // Equality constraints
        // --------------------------------

        for (int i = 0; i < meq; ++i) {

            for (int j = 0; j < n; ++j) {

                K[j][n + i] =
                    model.A_eq[i][j];

                K[n + i][j] =
                    model.A_eq[i][j];
            }

            rhs[n + i] =
                model.b_eq[i];
        }

        // --------------------------------
        // Active inequality constraints
        // --------------------------------

        int activeIndex = 0;

        for (int i = 0; i < mineq; ++i) {

            if (!(mask &
                  (std::uint64_t(1) << i))) {
                continue;
            }

            int row = n + meq + activeIndex;

            for (int j = 0; j < n; ++j) {

                K[j][row] =
                    Aineq[i][j];

                K[row][j] =
                    Aineq[i][j];
            }

            rhs[row] =
                bineq[i];

            ++activeIndex;
        }

        // --------------------------------
        // Solve KKT system
        // --------------------------------

        std::vector<double> solution;

        bool success =
            solveLinearSystem(K, rhs, solution);

        if (!success) {
            continue;
        }

        // --------------------------------
        // Extract x
        // --------------------------------

        std::vector<double> x(
            solution.begin(),
            solution.begin() + n
        );

        // --------------------------------
        // Check equality constraints
        // --------------------------------

        bool feasible = true;

        for (int i = 0; i < meq; ++i) {

            double value = 0.0;

            for (int j = 0; j < n; ++j) {
                value +=
                    model.A_eq[i][j] * x[j];
            }

            if (std::abs(
                    value - model.b_eq[i]
                ) > tolerance) {

                feasible = false;
                break;
            }
        }

        if (!feasible) {
            continue;
        }

        // --------------------------------
        // Check ALL inequalities
        // --------------------------------

        for (int i = 0; i < mineq; ++i) {

            double value = 0.0;

            for (int j = 0; j < n; ++j) {
                value +=
                    Aineq[i][j] * x[j];
            }

            if (value > bineq[i] + tolerance) {
                feasible = false;
                break;
            }
        }

        if (!feasible) {
            continue;
        }

        // --------------------------------
        // Check active inequality
        // multipliers
        //
        // mu >= 0
        // --------------------------------

        activeIndex = 0;

        for (int i = 0; i < mineq; ++i) {

            if (!(mask &
                  (std::uint64_t(1) << i))) {
                continue;
            }

            double mu =
                solution[n + meq + activeIndex];

            if (mu < -tolerance) {
                feasible = false;
                break;
            }

            ++activeIndex;
        }

        if (!feasible) {
            continue;
        }

        // --------------------------------
        // Candidate is feasible and
        // satisfies KKT conditions.
        // --------------------------------

        double objective =
            model.objectiveValue(x);

        if (!found ||
            objective < bestObjective) {

            found = true;
            bestX = x;
            bestObjective = objective;
        }
    }

    // --------------------------------
    // No feasible solution
    // --------------------------------

    if (!found) {

        result.status =
            "INFEASIBLE_CONSTRAINTS";

        result.objectiveValue = 0.0;

        auto end =
            std::chrono::high_resolution_clock::now();

        result.solveTime =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        return result;
    }

    // --------------------------------
    // Successful solution
    // --------------------------------

    result.status = "OPTIMAL";
    result.x = bestX;
    result.objectiveValue = bestObjective;

    auto end =
        std::chrono::high_resolution_clock::now();

    result.solveTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    return result;
}


// --------------------------------------------------
// Gaussian elimination with partial pivoting
// --------------------------------------------------

bool QPSolver::solveLinearSystem(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x) {

    int n =
        static_cast<int>(A.size());

    if (n == 0)
        return false;

    if (rhs.size() !=
        static_cast<size_t>(n))
        return false;

    // Check matrix dimensions
    for (const auto& row : A) {

        if (row.size() !=
            static_cast<size_t>(n)) {

            return false;
        }
    }

    std::vector<std::vector<double>> M = A;
    std::vector<double> b = rhs;

    const double tolerance = 1e-10;

    // --------------------------------
    // Forward elimination
    // --------------------------------

    for (int k = 0; k < n; ++k) {

        int pivot = k;

        for (int i = k + 1; i < n; ++i) {

            if (std::abs(M[i][k]) >
                std::abs(M[pivot][k])) {

                pivot = i;
            }
        }

        if (std::abs(M[pivot][k]) <
            tolerance) {

            return false;
        }

        std::swap(M[k], M[pivot]);
        std::swap(b[k], b[pivot]);

        for (int i = k + 1; i < n; ++i) {

            double factor =
                M[i][k] / M[k][k];

            for (int j = k; j < n; ++j) {

                M[i][j] -=
                    factor * M[k][j];
            }

            b[i] -=
                factor * b[k];
        }
    }

    // --------------------------------
    // Back substitution
    // --------------------------------

    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {

        double sum = b[i];

        for (int j = i + 1; j < n; ++j) {

            sum -=
                M[i][j] * x[j];
        }

        if (std::abs(M[i][i]) <
            tolerance) {

            return false;
        }

        x[i] =
            sum / M[i][i];
    }

    return true;
}
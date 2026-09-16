#include "../include/qp_model.hpp"
#include <cmath>

QPModel::QPModel(int variables) {
    n = variables;

    Q.resize(n, std::vector<double>(n, 0.0));
    c.resize(n, 0.0);

    lowerBound.resize(n, -1e20);
    upperBound.resize(n, 1e20);
}

bool QPModel::isValid() const {

    if (n <= 0)
        return false;

    if (Q.size() != n)
        return false;

    for (const auto& row : Q) {
        if (row.size() != n)
            return false;
    }

    if (c.size() != n)
        return false;

    if (lowerBound.size() != n ||
        upperBound.size() != n)
        return false;

    for (int i = 0; i < n; i++) {
        if (lowerBound[i] > upperBound[i])
            return false;
    }

    return true;

}
bool QPModel::isPositiveSemidefinite(double tolerance) const {

    // Q must be symmetric for a standard convex QP.
    if (!isSymmetric(tolerance))
        return false;

    // Cholesky-style PSD check.
    // For a positive semidefinite matrix, all required
    // diagonal terms remain non-negative.
    std::vector<std::vector<double>> L(
        n,
        std::vector<double>(n, 0.0)
    );

    for (int i = 0; i < n; ++i) {

        for (int j = 0; j <= i; ++j) {

            double sum = 0.0;

            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }

            double value =
                Q[i][j] - sum;

            if (i == j) {

                // A significantly negative diagonal
                // means Q is not positive semidefinite.
                if (value < -tolerance)
                    return false;

                // Small negative values can occur because
                // of floating-point round-off.
                L[i][j] =
                    (value > 0.0)
                    ? std::sqrt(value)
                    : 0.0;

            } else {

                if (std::abs(L[j][j]) > tolerance) {

                    L[i][j] =
                        value / L[j][j];

                } else {

                    // If the pivot is effectively zero,
                    // the corresponding value must also
                    // be approximately zero for PSD.
                    if (std::abs(value) > tolerance)
                        return false;

                    L[i][j] = 0.0;
                }
            }
        }
    }

    return true;
}

bool QPModel::isSymmetric(double tolerance) const {

    if (!isValid())
        return false;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            if (std::abs(Q[i][j] - Q[j][i]) > tolerance)
                return false;
        }
    }

    return true;
}

double QPModel::objectiveValue(
    const std::vector<double>& x
) const {

    double quadratic = 0.0;
    double linear = 0.0;

    // 1/2 x^T Q x
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            quadratic += x[i] * Q[i][j] * x[j];
        }
    }

    quadratic *= 0.5;

    // c^T x
    for (int i = 0; i < n; i++) {
        linear += c[i] * x[i];
    }

    return quadratic + linear;
}

std::vector<double> QPModel::gradient(
    const std::vector<double>& x
) const {

    std::vector<double> grad(n, 0.0);

    // grad f(x) = Qx + c
    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {
            grad[i] += Q[i][j] * x[j];
        }

        grad[i] += c[i];
    }

    return grad;
}
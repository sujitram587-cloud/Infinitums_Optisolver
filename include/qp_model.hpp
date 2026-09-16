#ifndef QP_MODEL_HPP
#define QP_MODEL_HPP

#include <vector>

class QPModel {
public:

    // Number of variables
    int n;

    // Quadratic objective matrix Q
    // Objective = 1/2 x^T Q x + c^T x
    std::vector<std::vector<double>> Q;

    // Linear objective vector c
    std::vector<double> c;

    // Equality constraints:
    // A_eq x = b_eq
    std::vector<std::vector<double>> A_eq;
    std::vector<double> b_eq;

    // Inequality constraints:
    // A_ineq x <= b_ineq
    std::vector<std::vector<double>> A_ineq;
    std::vector<double> b_ineq;

    // Variable bounds
    // lowerBound <= x <= upperBound
    std::vector<double> lowerBound;
    std::vector<double> upperBound;

    // Constructor
    QPModel(int variables);

    // Check whether the model dimensions are valid
    bool isValid() const;

    // Check whether Q is symmetric
    bool isSymmetric(double tolerance = 1e-9) const;
    bool isPositiveSemidefinite(double tolerance = 1e-9) const;

    // Calculate objective value
    double objectiveValue(
        const std::vector<double>& x
    ) const;

    // Calculate gradient:
    // grad f(x) = Qx + c
    std::vector<double> gradient(
        const std::vector<double>& x
    ) const;
};

#endif
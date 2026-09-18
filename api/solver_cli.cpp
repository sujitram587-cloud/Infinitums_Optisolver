#include <iostream>
#include <vector>

#include "../include/qp_model.hpp"
#include "../include/qp_solver.hpp"

using namespace std;

int main() {

    int n;
    cin >> n;

    QPModel model(n);

    // Read Q matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> model.Q[i][j];
        }
    }

    // Read c vector
    for (int i = 0; i < n; ++i) {
        cin >> model.c[i];
    }

    // Read equality constraints
    int eqCount;
    cin >> eqCount;

    if (eqCount > 0) {
        model.A_eq.resize(eqCount, vector<double>(n));
        model.b_eq.resize(eqCount);

        for (int i = 0; i < eqCount; ++i) {
            for (int j = 0; j < n; ++j) {
                cin >> model.A_eq[i][j];
            }
            cin >> model.b_eq[i];
        }
    }

    // Read inequality constraints
    int ineqCount;
    cin >> ineqCount;

    if (ineqCount > 0) {
        model.A_ineq.resize(ineqCount, vector<double>(n));
        model.b_ineq.resize(ineqCount);

        for (int i = 0; i < ineqCount; ++i) {
            for (int j = 0; j < n; ++j) {
                cin >> model.A_ineq[i][j];
            }
            cin >> model.b_ineq[i];
        }
    }

    // Read lower bounds
    for (int i = 0; i < n; ++i) {
        cin >> model.lowerBound[i];
    }

    // Read upper bounds
    for (int i = 0; i < n; ++i) {
        cin >> model.upperBound[i];
    }

    // Solve
    QPSolver solver;
    QPResult result = solver.solve(model);

    // Return machine-readable output
    cout << "STATUS " << result.status << '\n';

    cout << "X ";
    for (double value : result.x) {
        cout << value << ' ';
    }
    cout << '\n';

    cout << "OBJECTIVE " << result.objectiveValue << '\n';
    cout << "TIME " << result.solveTime << '\n';

    return 0;
}
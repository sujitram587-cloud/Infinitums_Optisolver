# Sovereign QP Solver

A C++17 prototype Quadratic Programming (QP) solver developed as part of the **Infinitums OptiSolver** project.

The solver is designed to solve convex quadratic optimization problems with linear equality constraints, linear inequality constraints, and variable bounds.

## Problem Formulation

The solver addresses problems of the form:

minimize

    1/2 xᵀQx + cᵀx

subject to

    A_eq x = b_eq

    A_ineq x ≤ b_ineq

    lowerBound ≤ x ≤ upperBound

where:

- `Q` is the quadratic coefficient matrix
- `c` is the linear objective vector
- `A_eq`, `b_eq` represent equality constraints
- `A_ineq`, `b_ineq` represent inequality constraints
- `lowerBound` and `upperBound` represent variable bounds

## Features

- Quadratic objective function
- Linear objective terms
- Symmetric matrix validation
- Positive-semidefinite / convexity checking
- Unconstrained QP solving
- Equality-constrained QP solving using KKT systems
- Inequality constraint handling
- Lower and upper variable bounds
- Feasibility checking
- Optimality checking
- Gaussian elimination with partial pivoting
- Solver status reporting
- Objective value calculation
- Solve-time measurement

## Project Structure

```text
Sovereign QP Solver/
├── examples/
│   └── basic_qp.cpp
├── include/
│   ├── qp_model.hpp
│   └── qp_solver.hpp
├── src/
│   ├── qp_model.cpp
│   └── qp_solver.cpp
├── tests/
│   └── qp_tests.cpp
├── CMakeLists.txt
└── README.md
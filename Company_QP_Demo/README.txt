COMPANY QP SOLVER DEMO
========================

Scenario
--------
Example Manufacturing Pvt. Ltd. manufactures three products:

  Product A, Product B, Product C

The company wants to determine production quantities that minimize
its quadratic production objective while respecting two limited
manufacturing resources.

The company sends the optimization model as a JSON input file.

INPUT FILE
----------
input/production_problem.json

The JSON contains:
  - Number of decision variables
  - Quadratic objective matrix Q
  - Linear objective vector c
  - Inequality constraints
  - Lower and upper bounds

BUSINESS INTERPRETATION
-----------------------
x[0] = production quantity of Product A
x[1] = production quantity of Product B
x[2] = production quantity of Product C

Resource constraint 1:
  2A + 1B + 1C <= 8

Resource constraint 2:
  1A + 2B + 1C <= 8

Production quantities must be non-negative.

SOLVER WORKFLOW
---------------
Company input file
        |
        v
Python file interface
        |
        v
Sovereign QP Solver (C++)
        |
        v
Optimization result
        |
        v
output/production_problem_output.json

HOW TO RUN
----------
From the Sovereign QP Solver project folder:

  python api/app.py "Company_QP_Demo/input/production_problem.json"

The generated result is written to:

  output/production_problem_output.json

If the demo folder is kept outside the main project, copy the input
file into the project's input folder or provide the appropriate path.

EXPECTED RESULT
---------------
The solver should report:

  status: OPTIMAL

  x[0] = 2.181818...
  x[1] = 2.181818...
  x[2] = 1.454545...

  objectiveValue = approximately -36.363636

The exact solve time will vary between runs.

DEMO EXPLANATION
----------------
"The company does not enter individual values into the solver.
It provides the complete mathematical optimization model as a JSON
file. Our Python interface validates and forwards that model to the
C++ QP engine. The solver finds the optimal production quantities
subject to the resource constraints and writes the result as JSON."

NOTE
----
This is a demonstration scenario for the prototype. The company name
and production data are illustrative.

import json
import subprocess
import sys
import os


# Check command-line argument
if len(sys.argv) not in [2, 3]:
    print("Error: Invalid number of arguments.")
    print("Usage: python api/app.py <input_file.json> [output_folder]")
    sys.exit(1)

INPUT_FILE = sys.argv[1]

if len(sys.argv) == 3:
    OUTPUT_DIR = sys.argv[2]
else:
    OUTPUT_DIR = "output"

# Check input file
if not os.path.isfile(INPUT_FILE):
    print("Error: Input file not found:")
    print(INPUT_FILE)
    sys.exit(1)


try:
    # Read JSON input
    with open(INPUT_FILE, "r") as file:
        data = json.load(file)

except json.JSONDecodeError:
    print("Error: The input file is not valid JSON.")
    sys.exit(1)

except OSError as error:
    print("Error: Could not read input file.")
    print(error)
    sys.exit(1)


# Required fields
required_fields = [
    "n",
    "Q",
    "c",
    "A_eq",
    "b_eq",
    "A_ineq",
    "b_ineq",
    "lowerBound",
    "upperBound"
]

missing_fields = [field for field in required_fields if field not in data]

if missing_fields:
    print("Error: Missing required field(s):")
    for field in missing_fields:
        print("-", field)
    sys.exit(1)


try:
    n = int(data["n"])

    if n <= 0:
        print("Error: 'n' must be a positive integer.")
        sys.exit(1)

    Q = data["Q"]
    c = data["c"]
    A_eq = data["A_eq"]
    b_eq = data["b_eq"]
    A_ineq = data["A_ineq"]
    b_ineq = data["b_ineq"]
    lower_bound = data["lowerBound"]
    upper_bound = data["upperBound"]

    # Validate Q
    if len(Q) != n or any(len(row) != n for row in Q):
        print("Error: Q must be an n x n matrix.")
        sys.exit(1)

    # Validate c
    if len(c) != n:
        print("Error: c must contain exactly n values.")
        sys.exit(1)

    # Validate equality constraints
    if len(A_eq) != len(b_eq):
        print("Error: A_eq and b_eq must have the same number of constraints.")
        sys.exit(1)

    for row in A_eq:
        if len(row) != n:
            print("Error: Every row in A_eq must contain exactly n values.")
            sys.exit(1)

    # Validate inequality constraints
    if len(A_ineq) != len(b_ineq):
        print("Error: A_ineq and b_ineq must have the same number of constraints.")
        sys.exit(1)

    for row in A_ineq:
        if len(row) != n:
            print("Error: Every row in A_ineq must contain exactly n values.")
            sys.exit(1)

    # Validate bounds
    if len(lower_bound) != n or len(upper_bound) != n:
        print("Error: lowerBound and upperBound must contain exactly n values.")
        sys.exit(1)

    for i in range(n):
        if lower_bound[i] > upper_bound[i]:
            print("Error: lowerBound cannot be greater than upperBound.")
            print("Problem at variable:", i)
            sys.exit(1)

except (TypeError, ValueError):
    print("Error: Invalid data type in the input JSON.")
    sys.exit(1)


values = []

# Q matrix
for row in Q:
    values.extend(row)

# c vector
values.extend(c)

# Equality constraints
values.append(len(A_eq))

for i in range(len(A_eq)):
    values.extend(A_eq[i])
    values.append(b_eq[i])

# Inequality constraints
values.append(len(A_ineq))

for i in range(len(A_ineq)):
    values.extend(A_ineq[i])
    values.append(b_ineq[i])

# Bounds
values.extend(lower_bound)
values.extend(upper_bound)

input_data = str(n) + "\n" + " ".join(map(str, values))


# Run C++ solver
result = subprocess.run(
    ["api/solver_cli.exe"],
    input=input_data,
    text=True,
    capture_output=True
)

if result.returncode != 0:
    print("Error: QP solver failed.")
    print(result.stderr)
    sys.exit(1)


# Read solver output
lines = result.stdout.strip().splitlines()

if len(lines) < 4:
    print("Error: Unexpected output from the QP solver.")
    sys.exit(1)

try:
    status = lines[0].replace("STATUS ", "")
    x = [float(v) for v in lines[1].replace("X ", "").split()]
    objective = float(lines[2].replace("OBJECTIVE ", ""))
    solve_time = float(lines[3].replace("TIME ", ""))
except (ValueError, IndexError):
    print("Error: Could not read the solver result.")
    sys.exit(1)


# Create output
output = {
    "status": status,
    "x": x,
    "objectiveValue": objective,
    "solveTime": solve_time
}


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

input_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    input_name + "_output.json"
)


# Write output JSON
with open(OUTPUT_FILE, "w") as file:
    json.dump(output, file, indent=2)


print("QP solved successfully.")
print("Input file:", INPUT_FILE)
print("Output written to:", OUTPUT_FILE)
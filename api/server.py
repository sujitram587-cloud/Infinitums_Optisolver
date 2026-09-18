from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)


@app.route("/solve", methods=["POST"])
def solve():

    data = request.get_json()

    n = data["n"]

    values = []

    # Q matrix
    for row in data["Q"]:
        values.extend(row)

    # c vector
    values.extend(data["c"])

    # Equality constraints
    eq = data["A_eq"]
    values.append(len(eq))

    for i in range(len(eq)):
        values.extend(eq[i])
        values.append(data["b_eq"][i])

    # Inequality constraints
    ineq = data["A_ineq"]
    values.append(len(ineq))

    for i in range(len(ineq)):
        values.extend(ineq[i])
        values.append(data["b_ineq"][i])

    # Bounds
    values.extend(data["lowerBound"])
    values.extend(data["upperBound"])

    input_data = str(n) + "\n" + " ".join(map(str, values))

    result = subprocess.run(
        ["api/solver_cli.exe"],
        input=input_data,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        return jsonify({
            "status": "SOLVER_ERROR",
            "error": result.stderr
        }), 500

    lines = result.stdout.strip().splitlines()

    status = lines[0].replace("STATUS ", "")
    x = [float(v) for v in lines[1].replace("X ", "").split()]
    objective = float(lines[2].replace("OBJECTIVE ", ""))
    solve_time = float(lines[3].replace("TIME ", ""))

    return jsonify({
        "status": status,
        "x": x,
        "objectiveValue": objective,
        "solveTime": solve_time
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sovereign QP Solver API is running",
        "endpoint": "/solve",
        "method": "POST"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
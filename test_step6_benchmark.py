import time
import pyomo.environ as pyo
from sovereign_pyomo import SovereignPyomoSolver

def run_benchmark():
    print("=== Step 6: Sovereign Solver Performance Benchmarking ===")
    
    # 1. Build Pyomo Model
    model = pyo.ConcreteModel(name="Benchmark_Refinery")
    model.crude_a = pyo.Var(bounds=(0, 5000))
    model.crude_b = pyo.Var(bounds=(0, 3000))
    model.cost = pyo.Objective(expr=45.0 * model.crude_a + 60.0 * model.crude_b, sense=pyo.minimize)
    model.sulfur = pyo.Constraint(expr=1.2 * model.crude_a + 0.8 * model.crude_b <= 7000.0)
    model.capacity = pyo.Constraint(expr=1.0 * model.crude_a + 1.0 * model.crude_b <= 6000.0)
    model.demand = pyo.Constraint(expr=1.0 * model.crude_a + 1.0 * model.crude_b >= 4000.0)

    # 2. Measure Sovereign GPU Execution Time
    solver = SovereignPyomoSolver()
    
    start_gpu = time.perf_counter()
    solution = solver.solve(model, max_iters=20000, learning_rate=0.00005)
    gpu_time = (time.perf_counter() - start_gpu) * 1000  # Convert to ms

    print("\n------------------------------------------------")
    print("  SOVEREIGN GPU SOLVER (NVIDIA RTX 4060)")
    print("------------------------------------------------")
    print(f"  Execution Time  : {gpu_time:.2f} ms")
    print(f"  CRUDE_A Yield   : {solution.get('crude_a', 0.0):.2f} barrels")
    print(f"  CRUDE_B Yield   : {solution.get('crude_b', 0.0):.2f} barrels")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    run_benchmark()
import pyomo.environ as pyo
from sovereign_pyomo import SovereignPyomoSolver

def main():
    print("=== Testing Step 5: Sovereign Pyomo Framework Integration ===")
    
    # Define standard Pyomo Concrete Model
    model = pyo.ConcreteModel(name="Pyomo_Refinery_Optimization")
    
    # Decision Variables
    model.crude_a = pyo.Var(bounds=(0, 5000))
    model.crude_b = pyo.Var(bounds=(0, 3000))
    
    # Objective Function: Minimize cost
    model.cost = pyo.Objective(expr=45.0 * model.crude_a + 60.0 * model.crude_b, sense=pyo.minimize)
    
    # Constraints
    model.sulfur = pyo.Constraint(expr=1.2 * model.crude_a + 0.8 * model.crude_b <= 7000.0)
    model.capacity = pyo.Constraint(expr=1.0 * model.crude_a + 1.0 * model.crude_b <= 6000.0)
    model.demand = pyo.Constraint(expr=1.0 * model.crude_a + 1.0 * model.crude_b >= 4000.0)

    # Solve via Sovereign GPU Plugin
    solver = SovereignPyomoSolver()
    solution = solver.solve(model, max_iters=10000, learning_rate=0.00005)
    
    print("\n=== PYOMO MODEL SOLVED VIA SOVEREIGN GPU ENGINE ===")
    for var_name, val in solution.items():
        print(f"  -> {var_name}: {val:.2f} barrels")

if __name__ == "__main__":
    main()
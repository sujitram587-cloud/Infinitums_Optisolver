from mps_parser import MPSParser

def main():
    print("=== Testing Step 4: Sovereign GPU Solver Engine Execution ===")
    
    # Parse MPS Refinery Optimization Problem
    parser = MPSParser("sample_model.mps")
    model, name = parser.parse()
    print(f"[SUCCESS] Loaded Model: '{name}'")
    
    # Offload to GPU VRAM
    model.offload_to_gpu()
    
    # Run GPU Solver Kernels
    print("[INFO] Launching CUDA Kernels on RTX 4060...")
    solution = model.solve(max_iters=10000, learning_rate=0.00005)
    
    print("\n=== OPTIMAL SOLUTION CALCULATED ON GPU ===")
    for idx, name in enumerate(parser.col_names):
        print(f"  -> Variable {name}: {solution[idx]:.4f} barrels")

    # Compute total objective cost
    obj_cost = sum(solution[i] * parser.obj_coeffs.get(i, 0.0) for i in range(len(solution)))
    print(f"  -> Optimal Objective Cost: ₹{obj_cost:,.2f}")

    # Memory cleanup
    model.free()
    print("\n[SUCCESS] Execution finished and VRAM released.")

if __name__ == "__main__":
    main()
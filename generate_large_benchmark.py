import time
import random
from sovereign import SovereignModel

def generate_and_benchmark_10k():
    num_vars = 10000
    num_constraints = 5000
    density = 0.001  # Sparse matrix density (~10 non-zeros per constraint)
    
    print("==========================================================")
    print("   SOVEREIGN GPU SOLVER: 10,000 VARIABLE BENCHMARK ENGINE")
    print("==========================================================")
    print(f"[INFO] Generating Synthetic Scale Dataset...")
    print(f"  -> Decision Variables (N) : {num_vars:,}")
    print(f"  -> Constraints (M)        : {num_constraints:,}")
    
    # 1. Generate Objective Coefficients & Variable Bounds
    random.seed(42)
    c_vector = [random.uniform(10.0, 100.0) for _ in range(num_vars)]
    var_lb = [0.0] * num_vars
    var_ub = [1000.0] * num_vars

    # 2. Build Synthetic CSR Sparse Constraint Matrix
    values = []
    col_indices = []
    row_pointers = [0]
    
    nnz_per_row = int(num_vars * density)
    
    for r in range(num_constraints):
        # Pick random column indices for sparse row entries
        cols = sorted(random.sample(range(num_vars), nnz_per_row))
        for c in cols:
            values.append(random.uniform(0.5, 5.0))
            col_indices.append(c)
        row_pointers.append(len(values))
        
    total_nnz = len(values)
    print(f"  -> Total Non-Zero Entries : {total_nnz:,}")

    # Constraint Bounds (Ax >= Row_LB)
    row_lb = [random.uniform(100.0, 500.0) for _ in range(num_constraints)]
    row_ub = [1e20] * num_constraints

    # 3. Instantiate GPU Model Pointer
    start_setup = time.perf_counter()
    model = SovereignModel(num_vars, num_constraints, total_nnz)
    model.set_objective(c_vector, is_maximize=False)
    model.set_var_bounds(var_lb, var_ub)
    model.set_row_bounds(row_lb, row_ub)
    model.set_csr_matrix(values, col_indices, row_pointers)
    
    # Offload Host CPU Memory -> NVIDIA RTX 4060 VRAM
    model.offload_to_gpu()
    setup_time = (time.perf_counter() - start_setup) * 1000

    # 4. Execute High-Throughput CUDA Optimization Kernels
    max_iters = 20000
    learning_rate = 0.00001
    
    print("\n[INFO] Launching Parallel CUDA Kernels on GPU Cores...")
    start_gpu = time.perf_counter()
    solution = model.solve(max_iters=max_iters, learning_rate=learning_rate)
    gpu_time = (time.perf_counter() - start_gpu) * 1000
    
    # Calculate GFLOPS / Performance Metrics
    total_ops = max_iters * (2 * total_nnz + 5 * num_vars)
    gflops = (total_ops / (gpu_time / 1000)) / 1e9

    print("\n==========================================================")
    print("                GPU PERFORMANCE REPORT                   ")
    print("==========================================================")
    print(f"  Host-to-VRAM Transfer Time : {setup_time:.2f} ms")
    print(f"  GPU Solver Execution Time  : {gpu_time:.2f} ms ({gpu_time/1000:.3f} s)")
    print(f"  CUDA Iteration Speed       : {gpu_time / max_iters:.4f} ms/iter")
    print(f"  Calculated Compute Rate    : {gflops:.2f} GFLOPS")
    print(f"  First 5 Decision Variables : {[round(x, 2) for x in solution[:5]]}")
    print("==========================================================")

    model.free()

if __name__ == "__main__":
    generate_and_benchmark_10k()
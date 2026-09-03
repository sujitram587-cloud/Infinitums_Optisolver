from sovereign import SovereignModel

def main():
    print("=== Testing Step 2: Sovereign Solver Python Library ===")

    # Define a 2-variable, 2-constraint LP model
    # Minimize 3*x1 + 5*x2
    # Constraints:
    #   1*x1 + 2*x2 <= 10
    #   0*x1 + 4*x2 <= 12
    num_vars = 2
    num_constraints = 2
    nnz = 3

    # Instantiate model
    model = SovereignModel(num_vars, num_constraints, nnz)

    # Populate data via Python API
    model.set_objective([3.0, 5.0], is_maximize=False)
    model.set_var_bounds([0.0, 0.0], [100.0, 100.0])
    model.set_row_bounds([-1e20, -1e20], [10.0, 12.0])

    # CSR Representation
    values = [1.0, 2.0, 4.0]
    col_indices = [0, 1, 1]
    row_pointers = [0, 2, 3]
    model.set_csr_matrix(values, col_indices, row_pointers)

    # Offload arrays from Python to C++/CUDA DLL & RTX 4060 VRAM
    success = model.offload_to_gpu()

    if success:
        print("[SUCCESS] Python API passed model data to sovereign.dll and RTX 4060 VRAM!")
    else:
        print("[ERROR] Failed to offload to GPU.")

    # Cleanup memory
    model.free()
    print("[SUCCESS] Memory cleared from Python API successfully.")

if __name__ == "__main__":
    main()
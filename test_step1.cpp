#include "sovereign_solver.h"
#include <iostream>

int main() {
    std::cout << "=== Testing Step 1: Sovereign Core GPU Library Setup ===" << std::endl;

    // Define a 2-variable, 2-constraint model with 3 non-zero terms
    // Minimize 3x1 + 5x2
    // Constraints:
    //   x1 + 2x2 <= 10
    //   0x1 + 4x2 <= 12
    int num_vars = 2;
    int num_constraints = 2;
    int nnz = 3;

    LPModel* model = create_lp_model(num_vars, num_constraints, nnz);

    // Fill Objective Vector (3, 5)
    model->c[0] = 3.0; model->c[1] = 5.0;

    // Variable Bounds: 0 <= x1, x2 <= 100
    model->var_lb[0] = 0.0; model->var_ub[0] = 100.0;
    model->var_lb[1] = 0.0; model->var_ub[1] = 100.0;

    // Constraint Bounds: -INF <= row <= 10 (or 12)
    model->row_lb[0] = -1e20; model->row_ub[0] = 10.0;
    model->row_lb[1] = -1e20; model->row_ub[1] = 12.0;

    // Populate Sparse Matrix (CSR format)
    model->A.values[0] = 1.0; model->A.col_indices[0] = 0;
    model->A.values[1] = 2.0; model->A.col_indices[1] = 1;
    model->A.values[2] = 4.0; model->A.col_indices[2] = 1;

    model->A.row_pointers[0] = 0; // Row 0 starts at index 0
    model->A.row_pointers[1] = 2; // Row 1 starts at index 2
    model->A.row_pointers[2] = 3; // End offset

    // Allocate & Copy to RTX 4060 VRAM
    GPULPModel* gpu_model = allocate_gpu_model(model);
    int status = copy_model_to_gpu(model, gpu_model);

    if (status == 0) {
        std::cout << "[SUCCESS] Model allocated and copied to RTX 4060 VRAM successfully." << std::endl;
    } else {
        std::cout << "[ERROR] GPU Memory Transfer Failed." << std::endl;
    }

    // Cleanup
    free_gpu_model(gpu_model);
    free_lp_model(model);

    std::cout << "[SUCCESS] VRAM and CPU Memory cleared cleanly." << std::endl;
    return 0;
}
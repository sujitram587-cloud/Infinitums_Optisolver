#ifndef SOVEREIGN_SOLVER_H
#define SOVEREIGN_SOLVER_H

#ifdef _WIN32
    #define SOVEREIGN_API __declspec(dllexport)
#else
    #define SOVEREIGN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    double* values;
    int* col_indices;
    int* row_pointers;
} CSRMatrix;

typedef struct {
    int num_vars;
    int num_constraints;
    CSRMatrix A;
    double* c;
    double* var_lb;
    double* var_ub;
    double* row_lb;
    double* row_ub;
    int is_maximize;
} LPModel;

typedef struct {
    int num_vars;
    int num_constraints;
    int nnz;
    double* d_values;
    int* d_col_indices;
    int* d_row_pointers;
    double* d_c;
    double* d_var_lb;
    double* d_var_ub;
    double* d_row_lb;
    double* d_row_ub;
    double* d_x;
    double* d_row_act; // VRAM buffer for row activities (A * x)
} GPULPModel;

SOVEREIGN_API LPModel* create_lp_model(int num_vars, int num_constraints, int nnz);
SOVEREIGN_API void free_lp_model(LPModel* model);
SOVEREIGN_API GPULPModel* allocate_gpu_model(const LPModel* host_model);
SOVEREIGN_API int copy_model_to_gpu(const LPModel* host_model, GPULPModel* gpu_model);
SOVEREIGN_API void free_gpu_model(GPULPModel* gpu_model);

// New Solver Engine API
SOVEREIGN_API int solve_gpu_model(GPULPModel* gpu_model, double* host_x_out, int max_iters, double learning_rate);

#ifdef __cplusplus
}
#endif

#endif // SOVEREIGN_SOLVER_H
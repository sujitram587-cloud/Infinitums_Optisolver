#include "sovereign_solver.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// CUDA Kernel 1: Calculate Sparse Matrix-Vector Product (A * x) on GPU Cores
__global__ void compute_row_activities_kernel(
    int num_constraints,
    const double* __restrict__ d_values,
    const int* __restrict__ d_col_indices,
    const int* __restrict__ d_row_pointers,
    const double* __restrict__ d_x,
    double* __restrict__ d_row_act
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_constraints) {
        double sum = 0.0;
        int start = d_row_pointers[row];
        int end = d_row_pointers[row + 1];
        for (int i = start; i < end; ++i) {
            sum += d_values[i] * d_x[d_col_indices[i]];
        }
        d_row_act[row] = sum;
    }
}

// CUDA Kernel 2: Gradient Step & Projected Bound Clamping on GPU
__global__ void update_variables_kernel(
    int num_vars, int num_constraints,
    const double* __restrict__ d_values,
    const int* __restrict__ d_col_indices,
    const int* __restrict__ d_row_pointers,
    const double* __restrict__ d_c,
    const double* __restrict__ d_var_lb,
    const double* __restrict__ d_var_ub,
    const double* __restrict__ d_row_lb,
    const double* __restrict__ d_row_ub,
    const double* __restrict__ d_row_act,
    double* __restrict__ d_x,
    double learning_rate,
    double penalty_weight
) {
    int var = blockIdx.x * blockDim.x + threadIdx.x;
    if (var >= num_vars) return;

    double grad = d_c[var];

    // Accumulate constraint violation gradient penalty
    for (int row = 0; row < num_constraints; ++row) {
        int start = d_row_pointers[row];
        int end = d_row_pointers[row + 1];
        for (int i = start; i < end; ++i) {
            if (d_col_indices[i] == var) {
                double val = d_values[i];
                double act = d_row_act[row];
                double ub = d_row_ub[row];
                double lb = d_row_lb[row];
                
                if (act > ub) {
                    grad += penalty_weight * val * (act - ub);
                } else if (act < lb) {
                    grad += penalty_weight * val * (act - lb);
                }
            }
        }
    }

    // Gradient descent update step
    double new_x = d_x[var] - learning_rate * grad;

    // Projection onto variable box bounds [var_lb, var_ub]
    if (new_x < d_var_lb[var]) new_x = d_var_lb[var];
    if (new_x > d_var_ub[var]) new_x = d_var_ub[var];

    d_x[var] = new_x;
}

extern "C" LPModel* create_lp_model(int num_vars, int num_constraints, int nnz) {
    LPModel* model = (LPModel*)malloc(sizeof(LPModel));
    model->num_vars = num_vars;
    model->num_constraints = num_constraints;
    model->is_maximize = 0;

    model->A.num_rows = num_constraints;
    model->A.num_cols = num_vars;
    model->A.nnz = nnz;
    model->A.values = (double*)malloc(nnz * sizeof(double));
    model->A.col_indices = (int*)malloc(nnz * sizeof(int));
    model->A.row_pointers = (int*)malloc((num_constraints + 1) * sizeof(int));

    model->c = (double*)malloc(num_vars * sizeof(double));
    model->var_lb = (double*)malloc(num_vars * sizeof(double));
    model->var_ub = (double*)malloc(num_vars * sizeof(double));
    model->row_lb = (double*)malloc(num_constraints * sizeof(double));
    model->row_ub = (double*)malloc(num_constraints * sizeof(double));

    return model;
}

extern "C" void free_lp_model(LPModel* model) {
    if (!model) return;
    free(model->A.values);
    free(model->A.col_indices);
    free(model->A.row_pointers);
    free(model->c);
    free(model->var_lb);
    free(model->var_ub);
    free(model->row_lb);
    free(model->row_ub);
    free(model);
}

extern "C" GPULPModel* allocate_gpu_model(const LPModel* host_model) {
    if (!host_model) return nullptr;

    GPULPModel* gpu_model = (GPULPModel*)malloc(sizeof(GPULPModel));
    gpu_model->num_vars = host_model->num_vars;
    gpu_model->num_constraints = host_model->num_constraints;
    gpu_model->nnz = host_model->A.nnz;

    cudaMalloc((void**)&gpu_model->d_values, gpu_model->nnz * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_col_indices, gpu_model->nnz * sizeof(int));
    cudaMalloc((void**)&gpu_model->d_row_pointers, (gpu_model->num_constraints + 1) * sizeof(int));

    cudaMalloc((void**)&gpu_model->d_c, gpu_model->num_vars * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_var_lb, gpu_model->num_vars * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_var_ub, gpu_model->num_vars * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_row_lb, gpu_model->num_constraints * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_row_ub, gpu_model->num_constraints * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_x, gpu_model->num_vars * sizeof(double));
    cudaMalloc((void**)&gpu_model->d_row_act, gpu_model->num_constraints * sizeof(double));

    return gpu_model;
}

extern "C" int copy_model_to_gpu(const LPModel* host_model, GPULPModel* gpu_model) {
    if (!host_model || !gpu_model) return -1;

    cudaMemcpy(gpu_model->d_values, host_model->A.values, host_model->A.nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_model->d_col_indices, host_model->A.col_indices, host_model->A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_model->d_row_pointers, host_model->A.row_pointers, (host_model->num_constraints + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_model->d_c, host_model->c, host_model->num_vars * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_model->d_var_lb, host_model->var_lb, host_model->num_vars * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_model->d_var_ub, host_model->var_ub, host_model->num_vars * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_model->d_row_lb, host_model->row_lb, host_model->num_constraints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_model->d_row_ub, host_model->row_ub, host_model->num_constraints * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize d_x to initial lower bounds
    cudaMemcpy(gpu_model->d_x, host_model->var_lb, host_model->num_vars * sizeof(double), cudaMemcpyHostToDevice);

    return 0;
}

extern "C" int solve_gpu_model(GPULPModel* gpu_model, double* host_x_out, int max_iters, double learning_rate) {
    if (!gpu_model || !host_x_out) return -1;

    int threads_per_block = 256;
    int blocks_vars = (gpu_model->num_vars + threads_per_block - 1) / threads_per_block;
    int blocks_rows = (gpu_model->num_constraints + threads_per_block - 1) / threads_per_block;

    double penalty_weight = 100.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Step 1: Compute Ax on GPU
        compute_row_activities_kernel<<<blocks_rows, threads_per_block>>>(
            gpu_model->num_constraints,
            gpu_model->d_values,
            gpu_model->d_col_indices,
            gpu_model->d_row_pointers,
            gpu_model->d_x,
            gpu_model->d_row_act
        );

        // Step 2: Compute Gradient & Project on GPU
        update_variables_kernel<<<blocks_vars, threads_per_block>>>(
            gpu_model->num_vars,
            gpu_model->num_constraints,
            gpu_model->d_values,
            gpu_model->d_col_indices,
            gpu_model->d_row_pointers,
            gpu_model->d_c,
            gpu_model->d_var_lb,
            gpu_model->d_var_ub,
            gpu_model->d_row_lb,
            gpu_model->d_row_ub,
            gpu_model->d_row_act,
            gpu_model->d_x,
            learning_rate,
            penalty_weight
        );
    }

    cudaDeviceSynchronize();

    // Copy computed solution back from RTX 4060 VRAM to CPU RAM
    cudaMemcpy(host_x_out, gpu_model->d_x, gpu_model->num_vars * sizeof(double), cudaMemcpyDeviceToHost);

    return 0;
}

extern "C" void free_gpu_model(GPULPModel* gpu_model) {
    if (!gpu_model) return;
    cudaFree(gpu_model->d_values);
    cudaFree(gpu_model->d_col_indices);
    cudaFree(gpu_model->d_row_pointers);
    cudaFree(gpu_model->d_c);
    cudaFree(gpu_model->d_var_lb);
    cudaFree(gpu_model->d_var_ub);
    cudaFree(gpu_model->d_row_lb);
    cudaFree(gpu_model->d_row_ub);
    cudaFree(gpu_model->d_x);
    cudaFree(gpu_model->d_row_act);
    free(gpu_model);
}
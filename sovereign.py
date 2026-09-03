import ctypes
import os

dll_path = os.path.abspath("sovereign.dll")
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Could not find {dll_path}. Build sovereign.dll first.")

sovereign_lib = ctypes.CDLL(dll_path)

class CSRMatrix(ctypes.Structure):
    _fields_ = [
        ("num_rows", ctypes.c_int),
        ("num_cols", ctypes.c_int),
        ("nnz", ctypes.c_int),
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("col_indices", ctypes.POINTER(ctypes.c_int)),
        ("row_pointers", ctypes.POINTER(ctypes.c_int))
    ]

class LPModel(ctypes.Structure):
    _fields_ = [
        ("num_vars", ctypes.c_int),
        ("num_constraints", ctypes.c_int),
        ("A", CSRMatrix),
        ("c", ctypes.POINTER(ctypes.c_double)),
        ("var_lb", ctypes.POINTER(ctypes.c_double)),
        ("var_ub", ctypes.POINTER(ctypes.c_double)),
        ("row_lb", ctypes.POINTER(ctypes.c_double)),
        ("row_ub", ctypes.POINTER(ctypes.c_double)),
        ("is_maximize", ctypes.c_int)
    ]

class GPULPModel(ctypes.Structure):
    _fields_ = [
        ("num_vars", ctypes.c_int),
        ("num_constraints", ctypes.c_int),
        ("nnz", ctypes.c_int),
        ("d_values", ctypes.c_void_p),
        ("d_col_indices", ctypes.c_void_p),
        ("d_row_pointers", ctypes.c_void_p),
        ("d_c", ctypes.c_void_p),
        ("d_var_lb", ctypes.c_void_p),
        ("d_var_ub", ctypes.c_void_p),
        ("d_row_lb", ctypes.c_void_p),
        ("d_row_ub", ctypes.c_void_p),
        ("d_x", ctypes.c_void_p),
        ("d_row_act", ctypes.c_void_p)
    ]

sovereign_lib.create_lp_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
sovereign_lib.create_lp_model.restype = ctypes.POINTER(LPModel)

sovereign_lib.free_lp_model.argtypes = [ctypes.POINTER(LPModel)]
sovereign_lib.free_lp_model.restype = None

sovereign_lib.allocate_gpu_model.argtypes = [ctypes.POINTER(LPModel)]
sovereign_lib.allocate_gpu_model.restype = ctypes.POINTER(GPULPModel)

sovereign_lib.copy_model_to_gpu.argtypes = [ctypes.POINTER(LPModel), ctypes.POINTER(GPULPModel)]
sovereign_lib.copy_model_to_gpu.restype = ctypes.c_int

sovereign_lib.solve_gpu_model.argtypes = [ctypes.POINTER(GPULPModel), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double]
sovereign_lib.solve_gpu_model.restype = ctypes.c_int

sovereign_lib.free_gpu_model.argtypes = [ctypes.POINTER(GPULPModel)]
sovereign_lib.free_gpu_model.restype = None

class SovereignModel:
    def __init__(self, num_vars: int, num_constraints: int, nnz: int):
        self.num_vars = num_vars
        self.num_constraints = num_constraints
        self.nnz = nnz
        self.model_ptr = sovereign_lib.create_lp_model(num_vars, num_constraints, nnz)
        self.gpu_model_ptr = None

    def set_objective(self, c_vector: list, is_maximize: bool = False):
        for i in range(self.num_vars):
            self.model_ptr.contents.c[i] = c_vector[i]
        self.model_ptr.contents.is_maximize = 1 if is_maximize else 0

    def set_var_bounds(self, var_lb: list, var_ub: list):
        for i in range(self.num_vars):
            self.model_ptr.contents.var_lb[i] = var_lb[i]
            self.model_ptr.contents.var_ub[i] = var_ub[i]

    def set_row_bounds(self, row_lb: list, row_ub: list):
        for i in range(self.num_constraints):
            self.model_ptr.contents.row_lb[i] = row_lb[i]
            self.model_ptr.contents.row_ub[i] = row_ub[i]

    def set_csr_matrix(self, values: list, col_indices: list, row_pointers: list):
        for i in range(self.nnz):
            self.model_ptr.contents.A.values[i] = values[i]
            self.model_ptr.contents.A.col_indices[i] = col_indices[i]
        for i in range(self.num_constraints + 1):
            self.model_ptr.contents.A.row_pointers[i] = row_pointers[i]

    def offload_to_gpu(self) -> bool:
        self.gpu_model_ptr = sovereign_lib.allocate_gpu_model(self.model_ptr)
        status = sovereign_lib.copy_model_to_gpu(self.model_ptr, self.gpu_model_ptr)
        return status == 0

    def solve(self, max_iters: int = 5000, learning_rate: float = 0.0001) -> list[float]:
        if not self.gpu_model_ptr:
            raise RuntimeError("Model must be offloaded to GPU before calling solve().")
        
        solution_arr = (ctypes.c_double * self.num_vars)()
        sovereign_lib.solve_gpu_model(self.gpu_model_ptr, solution_arr, max_iters, learning_rate)
        return [solution_arr[i] for i in range(self.num_vars)]

    def free(self):
        if self.gpu_model_ptr:
            sovereign_lib.free_gpu_model(self.gpu_model_ptr)
            self.gpu_model_ptr = None
        if self.model_ptr:
            sovereign_lib.free_lp_model(self.model_ptr)
            self.model_ptr = None
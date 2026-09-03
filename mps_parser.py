import os
from sovereign import SovereignModel

class MPSParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.model_name = ""
        self.obj_name = ""
        self.row_names = []         # Constraint row names (excludes objective)
        self.row_indices = {}       # Map row_name -> index (0..M-1)
        self.row_types = {}         # Map row_name -> 'L', 'G', 'E', or 'N'
        
        self.col_names = []         # Column / Variable names
        self.col_indices = {}       # Map col_name -> index (0..N-1)
        
        self.obj_coeffs = {}        # Map col_idx -> objective coefficient
        self.matrix_entries = {}    # Map (row_idx, col_idx) -> value
        
        self.rhs_values = {}        # Map row_name -> RHS value
        self.var_lb = {}            # Map col_name -> lower bound
        self.var_ub = {}            # Map col_name -> upper bound

    def parse(self) -> tuple[SovereignModel, str]:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"MPS file not found: {self.filepath}")

        current_section = None
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip('\n')
                if not line or line.startswith('*'):
                    continue
                
                # Check for section headers (unindented)
                if not line.startswith(' ') and not line.startswith('\t'):
                    parts = line.split()
                    current_section = parts[0]
                    if current_section == "NAME":
                        self.model_name = parts[1] if len(parts) > 1 else "Unnamed"
                    continue

                tokens = line.split()
                if not tokens:
                    continue

                if current_section == "ROWS":
                    row_type = tokens[0]
                    row_name = tokens[1]
                    self.row_types[row_name] = row_type
                    if row_type == 'N' and not self.obj_name:
                        self.obj_name = row_name
                    elif row_type != 'N':
                        if row_name not in self.row_indices:
                            self.row_indices[row_name] = len(self.row_names)
                            self.row_names.append(row_name)

                elif current_section == "COLUMNS":
                    col_name = tokens[0]
                    if col_name not in self.col_indices:
                        self.col_indices[col_name] = len(self.col_names)
                        self.col_names.append(col_name)
                    
                    col_idx = self.col_indices[col_name]
                    
                    # Parse pairs of (row_name, coefficient)
                    idx = 1
                    while idx < len(tokens):
                        r_name = tokens[idx]
                        val = float(tokens[idx+1])
                        if r_name == self.obj_name:
                            self.obj_coeffs[col_idx] = val
                        elif r_name in self.row_indices:
                            r_idx = self.row_indices[r_name]
                            self.matrix_entries[(r_idx, col_idx)] = val
                        idx += 2

                elif current_section == "RHS":
                    idx = 1
                    while idx < len(tokens):
                        r_name = tokens[idx]
                        val = float(tokens[idx+1])
                        self.rhs_values[r_name] = val
                        idx += 2

                elif current_section == "BOUNDS":
                    b_type = tokens[0]
                    c_name = tokens[2]
                    val = float(tokens[3]) if len(tokens) > 3 else 0.0
                    
                    if b_type == 'LO':
                        self.var_lb[c_name] = val
                    elif b_type == 'UP':
                        self.var_ub[c_name] = val
                    elif b_type == 'FX':
                        self.var_lb[c_name] = val
                        self.var_ub[c_name] = val
                    elif b_type == 'FR':
                        self.var_lb[c_name] = -1e20
                        self.var_ub[c_name] = 1e20

                elif current_section == "ENDATA":
                    break

        num_vars = len(self.col_names)
        num_constraints = len(self.row_names)
        
        c_vector = [self.obj_coeffs.get(i, 0.0) for i in range(num_vars)]
        lb_vector = [self.var_lb.get(name, 0.0) for name in self.col_names]
        ub_vector = [self.var_ub.get(name, 1e20) for name in self.col_names]
        
        row_lb = []
        row_ub = []
        for r_name in self.row_names:
            r_type = self.row_types[r_name]
            rhs = self.rhs_values.get(r_name, 0.0)
            if r_type == 'L':
                row_lb.append(-1e20)
                row_ub.append(rhs)
            elif r_type == 'G':
                row_lb.append(rhs)
                row_ub.append(1e20)
            elif r_type == 'E':
                row_lb.append(rhs)
                row_ub.append(rhs)
            else:
                row_lb.append(-1e20)
                row_ub.append(1e20)

        # Build CSR Matrix vectors
        values = []
        col_indices = []
        row_pointers = [0]

        for r_idx in range(num_constraints):
            for c_idx in range(num_vars):
                if (r_idx, c_idx) in self.matrix_entries:
                    values.append(self.matrix_entries[(r_idx, c_idx)])
                    col_indices.append(c_idx)
            row_pointers.append(len(values))

        nnz = len(values)

        # Instantiate Sovereign Model
        model = SovereignModel(num_vars, num_constraints, nnz)
        model.set_objective(c_vector, is_maximize=False)
        model.set_var_bounds(lb_vector, ub_vector)
        model.set_row_bounds(row_lb, row_ub)
        model.set_csr_matrix(values, col_indices, row_pointers)
        
        return model, self.model_name